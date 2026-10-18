"""MAMMA dance / singles metrics for network predictions or fits, through the fair evaluator's own helpers:
neutral SMPL-X decode -> smplx2smpl -> SMPL-22 and H36M-14 regressed joints (both rooted on their own pelvis),
per-frame Hungarian on the absolute world H36M pelvis, MPJPE root-relative, PA = Umeyama. Every GT person counts.

  --dataset dance    MAMMA dance (neutral GT); reported for all 18 / 2 DEV / 16 TEST sequences
  --dataset singles  MAMMA singles (gendered GT, any number of people); also writes per-frame MPJPE-22
                     (<out stem>_<seq>_per_frame.npy: frame, gt_person, mpjpe22, pa22) and the WalkGreenChair windows
  --kind infer       multismpl.inference.mamma output (camera-0 gauge)      -> network row
  --kind fit         multismpl.tto.run output (world)

  python -m multismpl.eval.mamma --dataset dance --kind fit --pred-dir <fit_dir> --out eval_fit.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from multismpl.eval.common import prediction_world
from multismpl.eval.protocols import mamma as proto
from multismpl.paths import PATHS

KEYS = ('mpjpe22_mm', 'pa_mpjpe22_mm', 'mpjpe14_mm', 'pa_mpjpe14_mm')
DEV = ['250225_WestCoastSwing_Improv22_03675_03676_1', '250225_WestCoastSwing_Outside_Turn_03675_03676_1']  # dance tuning set
WINDOWS = {'crossed 13-17s': (390, 510), 'sitting 22-38s': (660, 1140)}   # singles WalkGreenChair_50049_1, 30 fps
DATASETS = ('dance', 'singles')


def score_sequence(assets, seq_dir, pose, beta, pelvis, n):
    """Matched (prediction, GT) pairs of one sequence -> list of metric dicts, number of GT people."""
    gt_j, gt_v, n_gt, n_people = proto.load_sequence_gt(seq_dir, assets, n)
    n = min(n, n_gt)
    P = pose.shape[1]
    pairs = []
    for s in range(0, n, 128):
        e = min(n, s + 128)
        Pp, B, A = pose[s:e].reshape(-1, 72), beta[s:e].reshape(-1, 10), pelvis[s:e].reshape(-1, 3)
        with torch.inference_mode():
            pj, pv = assets.body(torch.tensor(Pp, dtype=torch.float32), torch.tensor(B, dtype=torch.float32), torch.zeros(len(Pp), 3))
            pv = pv + (torch.tensor(A, dtype=torch.float32) - pj[:, 0])[:, None]
            pv = torch.einsum('sv,pvc->psc', assets.smplx2smpl, pv.to(assets.smplx2smpl.device)).cpu().numpy()
        pv = pv.reshape(e - s, P, 6890, 3)
        for fi in range(e - s):
            fr = s + fi
            ph = np.stack([assets.j_regressor @ v for v in pv[fi]])
            gh = np.stack([assets.j_regressor @ v for v in gt_v[fr]])
            cost = np.linalg.norm(ph[:, None, 0] - gh[None, :, 0], axis=-1)
            for p, g in zip(*linear_sum_assignment(cost)):
                m = proto.fair_regressed_metrics(pv[fi][p], gt_v[fr][g], assets.smpl_j_regressor, assets.j_regressor)
                m.update(frame=fr, gt_person=int(g))
                pairs.append(m)
    return pairs, n, n_people


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=DATASETS, required=True)
    ap.add_argument('--kind', choices=['infer', 'fit'], required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, default=None)
    ap.add_argument('--max-frames', type=int, default=0)
    opt = ap.parse_args()
    root = opt.dataset_root or (PATHS.mamma_dance if opt.dataset == 'dance' else PATHS.mamma_singles)
    singles = opt.dataset == 'singles'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assets = proto.load_eval_assets(str(PATHS.smplx_root), PATHS.smplx2smpl, PATHS.j_regressor_h36m, PATHS.smpl_neutral, dev)

    out = {'kind': opt.kind, 'pred_dir': str(opt.pred_dir)} | ({} if singles else {'dev_sequences': DEV})
    out['per_sequence'] = {}
    splits = ['all'] if singles else ['all', 'dev', 'test']
    pairs_all, gt_all = {k: [] for k in splits}, {k: 0 for k in splits}
    for f in sorted(opt.pred_dir.glob('*.npz')):
        seq = f.stem
        seq_dir = root / seq
        if not (seq_dir / 'gt/global.npz').is_file():
            continue
        pose, beta, pelvis = prediction_world(dict(np.load(f)), opt.kind)
        n = len(pose) if not opt.max_frames else min(opt.max_frames, len(pose))
        pairs, n, n_people = score_sequence(assets, seq_dir, pose, beta, pelvis, n)
        res = proto.aggregate(pairs, n_people * n, len(pairs))
        row = {'frames': n, **{k: res[k] for k in KEYS}}
        for k in ['all'] + ([] if singles else ['dev' if seq in DEV else 'test']):
            pairs_all[k] += pairs
            gt_all[k] += n_people * n
        print(f'{seq[:52]:52s} n={n:4d} ' + ' '.join(f'{k[:-3]}={res[k]:.2f}' for k in KEYS), flush=True)
        if singles:
            per_frame = np.array([[p['frame'], p['gt_person'], p['mpjpe22_mm'], p['pa_mpjpe22_mm']] for p in pairs])
            np.save(opt.out.with_name(opt.out.stem + f'_{seq}_per_frame.npy'), per_frame)
            row['windows'] = {}
            for name, (lo, hi) in (WINDOWS.items() if 'WalkGreenChair_50049_1' in seq else []):
                m = (per_frame[:, 0] >= lo) & (per_frame[:, 0] <= hi)
                row['windows'][name] = {'mpjpe22_mm': float(per_frame[m, 2].mean()), 'pa_mpjpe22_mm': float(per_frame[m, 3].mean()),
                                        'frames': int(m.sum())}
                print(f'   {name:15s} mpjpe22={row["windows"][name]["mpjpe22_mm"]:.2f}', flush=True)
        out['per_sequence'][seq] = row
    for k in splits:
        if pairs_all[k]:
            r = proto.aggregate(pairs_all[k], gt_all[k], len(pairs_all[k]))
            out[k] = {kk: r[kk] for kk in KEYS} | {'pairs': len(pairs_all[k]), 'gt_people': gt_all[k]}
            if singles:
                out[k]['sequences'] = len(out['per_sequence'])
            print(f'== {k:4s}: ' + ' '.join(f'{kk[:-3]}={r[kk]:.2f}' for kk in KEYS) + f' pairs {len(pairs_all[k])}/{gt_all[k]}', flush=True)
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
