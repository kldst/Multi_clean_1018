"""Step 2 of the pipeline: tracking + test-time optimisation for every sequence of an inference folder.

  python -m multismpl.tto.run --infer <infer_dir> --out <fit_dir>          # every dataset: Joint Candidate Tracking + TTO
  python -m multismpl.tto.run --infer <infer_dir> --out <fit_dir> --tracker slots      # legacy slot tracker

Pipeline: the inference picks the top-k slots per frame (the network score) and stores C mask-ranked candidates;
Joint Candidate Tracking links candidates into P person tracks (it may replace a wrongly picked slot); TTO fits them.

Input  <infer_dir>/<seq>.npz from multismpl.inference.{mamma,chi3d,harmony4d} (camera-0 gauge slots, masks,
       K, E; cand_* arrays for --tracker candidates).
Output <fit_dir>/<seq>.npz, WORLD frame (= camera 0 for CHI3D / Harmony4D):
  pose (F,P,72)  beta (F,P,10)  anchor (F,P,3) pelvis          fitted
  init_pose / init_beta / init_anchor                          network start after tracking
  perm (F,P)  ambiguous (F,)  dropped (F,P) missing person  repair (json tracking stats)  w_in_mean / w_cov_mean (V,)
"""
import argparse
import json
import time
from pathlib import Path

import multismpl.paths  # noqa: F401  (sets the torch env flag before torch is imported)
import numpy as np
import torch

from multismpl.body.body_model import init_smplx
from multismpl.tto.fit import SequenceFit
from multismpl.tto.tracking import stats_json, tracks_from_candidates, tracks_from_slots


def build_tracks(d, n, opt, dev):
    if opt.tracker == 'candidates':
        if 'cand_trans' not in d:
            raise SystemExit('--tracker candidates needs an inference run with --candidates C')
        P = opt.people or d['pose'].shape[1]
        return tracks_from_candidates(d, n, P, drop_outlier_views=opt.triangulation == 'outlier', up_axis=opt.up_axis, dev=dev,
                                      up_shift=opt.up_shift, prefer_consistent=opt.prefer_consistent,
                                      topk_bonus=opt.topk_bonus, w_pose=opt.w_pose_link,
                                      keep_topk_pairs=opt.keep_topk_pairs, min_presence_boost=opt.min_presence_boost,
                                      dup_mask_iou=opt.dup_mask_iou, reacquire=opt.reacquire)
    return tracks_from_slots(d, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--infer', type=Path, required=True, help='inference folder')
    ap.add_argument('--out', type=Path, required=True, help='fit folder')
    ap.add_argument('--sequences', nargs='*', default=None)
    ap.add_argument('--max-frames', type=int, default=0, help='fit only the first N frames (debug)')
    ap.add_argument('--device', default='cuda')
    g = ap.add_argument_group('tracking')
    g.add_argument('--tracker', choices=['slots', 'candidates'], default='candidates',
                   help='candidates: Joint Candidate Tracking (JCT) over the stored mask candidates (default, every dataset); '
                        'slots: legacy keep/swap Viterbi over the selected slots (the older reported numbers)')
    g.add_argument('--people', type=int, default=0, help='candidates: number of people (default: slots stored per frame)')
    g.add_argument('--triangulation', choices=['plain', 'outlier'], default='outlier',
                   help='candidates: all-view DLT, or drop a view only when it is a clear leave-one-out outlier (> 100 px)')
    g.add_argument('--up-axis', choices=['z', 'body'], default='body',
                   help='candidates: silhouette centre -> pelvis shift along world z or the body-estimated up axis')
    g.add_argument('--up-shift', type=float, default=0., help='candidates: that shift in metres (default 0: the mask centre is used as is)')
    g.add_argument('--prefer-consistent', action=argparse.BooleanOptionalAction, default=True,
                   help='candidates (default on): among duplicates keep first a candidate whose network pelvis agrees (< 1 m) with its '
                        'mask triangulation (drops mixed candidates: masks on one person, body on another)')
    g.add_argument('--topk-bonus', type=float, default=0.,
                   help="candidates: link-cost bonus for the network's top-k slots (motion units, 1 = 10 cm; 0 = off)")
    g.add_argument('--w-pose-link', type=float, default=0.,
                   help='candidates: link cost per radian of mean body-pose change (0 = off)')
    g.add_argument('--keep-topk-pairs', action=argparse.BooleanOptionalAction, default=True,
                   help="candidates (default on): never merge two of the network's top-k slots as duplicates (people hugging have mask "
                        'centroids < 0.25 m apart; the network\'s 3D NMS already separated them)')
    g.add_argument('--min-presence-boost', type=float, default=0.1,
                   help='candidates: a mask-supported candidate counts as confident only if its presence >= this (0 = always)')
    g.add_argument('--dup-mask-iou', type=float, default=0.,
                   help='candidates: candidates whose mean mask IoU exceeds this are one person, top-k slots included (0 = off)')
    g.add_argument('--reacquire', action=argparse.BooleanOptionalAction, default=True,
                   help='candidates (default on): a track that missed the previous frame may take a clear candidate at any distance')
    g = ap.add_argument_group('fit (defaults = every reported number)')
    g.add_argument('--mode', choices=['joint', 'indep'], default='joint', help='joint: occlusion-aware; indep: each person alone')
    g.add_argument('--per-person-smooth', action=argparse.BooleanOptionalAction, default=True,
                   help='(default on) smoothness / beta use per-person ambiguity (a missing or touching person only turns off its own)')
    g.add_argument('--chunk', type=int, default=64)
    g.add_argument('--iters-trans', type=int, default=100, help='stage A')
    g.add_argument('--iters-global', type=int, default=50, help='stage B')
    g.add_argument('--iters', type=int, default=100, help='stage C')
    g.add_argument('--w-pose', type=float, default=.1)
    g.add_argument('--w-beta', type=float, default=1.)
    g.add_argument('--w-smooth', type=float, default=.01)
    g.add_argument('--n-cover', type=int, default=256, help='mask pixels sampled per view')
    g.add_argument('--n-vsub', type=int, default=1500, help='vertices used by the cover term')
    opt = ap.parse_args()
    opt.out.mkdir(parents=True, exist_ok=True)
    init_smplx()
    dev = torch.device(opt.device)
    files = sorted(opt.infer.glob('*.npz'))
    if opt.sequences:
        files = [f for f in files if f.stem in opt.sequences]
    timing = {}
    for f in files:
        if (opt.out / f.name).is_file():
            print(f'{f.stem}: exists, skip', flush=True)
            continue
        d = dict(np.load(f))
        n = len(d['pose']) if not opt.max_frames else min(opt.max_frames, len(d['pose']))
        tracks = build_tracks(d, n, opt, dev)
        fitter = SequenceFit(tracks, d['K'], d['E'], opt, dev)
        log, sec = fitter.run()
        res = fitter.result() | {'repair': stats_json(tracks)}
        np.savez(opt.out / f.name, **res)
        swaps = int((np.diff(res['perm'][:, 0]) != 0).sum())
        timing[f.stem] = {'frames': n, 'seconds': sec, 'ambiguous_frac': float(res['ambiguous'].mean()), 'swaps': swaps}
        print(f'{f.stem[:50]:50s} {n:4d} f {sec:6.0f}s ({sec / n * 1e3:.0f} ms/f) ambiguous {res["ambiguous"].mean():.2f} '
              f'swaps {swaps} tracking {res["repair"]} w_in {np.round(res["w_in_mean"], 2)} w_cov {np.round(res["w_cov_mean"], 2)}',
              flush=True)
    (opt.out / f'timing_{int(time.time())}.json').write_text(
        json.dumps({'args': {k: str(v) for k, v in vars(opt).items()}, 'timing': timing}, indent=1))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
