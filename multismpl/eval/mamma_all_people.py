"""Supplementary MAMMA singles / dance evaluation where the method outputs EVERY person it detects (not the official
protocol, which takes a fixed number of people per frame). Per frame the prediction is every slot with presence >
threshold (the top-1 slot if none passes); GT people are matched to predictions by Hungarian on PVE; unmatched
predictions are false positives (on singles they are often real but unannotated bystanders). Reports, per threshold:
MPJPE / PA-MPJPE (SMPL-22, H36M-14) of the matched pairs without and with the 500 mm PVE gate, recall, precision and
false positives per frame. The per-pair metrics are the official ones (protocols.mamma.fair_regressed_metrics).

Needs an inference run that stored every slot (all_presence / all_pose / all_beta / all_trans, i.e. made with the
current multismpl.inference.mamma).

  python -m multismpl.eval.mamma_all_people --dataset singles --pred-dir <infer_dir> --out all_people.json \
      [--thresholds 0.3 0.35 0.4 0.5 0.6 0.7]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from multismpl.body.body_model import gauge_to_world
from multismpl.eval.protocols import mamma as proto
from multismpl.eval.protocols.common import PVE_MATCH_THRESH_MM
from multismpl.paths import PATHS

KEYS = ('mpjpe22_mm', 'pa_mpjpe22_mm', 'mpjpe14_mm', 'pa_mpjpe14_mm')


def world_verts(assets, pose, beta, pelvis):
    """(N,72) gauge-converted world pose, (N,10), (N,3) -> (N,6890,3) SMPL-topology vertices (as the evaluator)."""
    with torch.inference_mode():
        pj, pv = assets.body(torch.tensor(pose, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32),
                             torch.zeros(len(pose), 3))
        pv = pv + (torch.tensor(pelvis, dtype=torch.float32) - pj[:, 0])[:, None]
        return torch.einsum('sv,pvc->psc', assets.smplx2smpl, pv.to(assets.smplx2smpl.device)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['singles', 'dance'], default='singles')
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--thresholds', type=float, nargs='*', default=[0.3, 0.35, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument('--selected', action='store_true',
                    help="also score the run's own stored selection (pose / trans, fixed people per frame) the same way; "
                         'with --thresholds and no values, only that')
    opt = ap.parse_args()
    keys = list(opt.thresholds) + (['selected'] if opt.selected else [])
    root = PATHS.mamma_singles if opt.dataset == 'singles' else PATHS.mamma_dance
    assets = proto.load_eval_assets(str(PATHS.smplx_root), PATHS.smplx2smpl, PATHS.j_regressor_h36m, PATHS.smpl_neutral,
                                    torch.device('cpu'))
    acc = {t: {m: {'pairs': [], 'gt': 0, 'pred': 0, 'frames': 0} for m in ('nogate', 'gated')} for t in keys}
    per_seq = {}
    for f in sorted(opt.pred_dir.glob('*.npz')):
        d = dict(np.load(f))
        if opt.thresholds and 'all_pose' not in d:
            raise SystemExit(f'{f}: no all_pose (re-run multismpl.inference.mamma)')
        seq_dir = root / f.stem
        if not (seq_dir / 'gt/global.npz').is_file():
            continue
        _, gt_v, n_gt, _ = proto.load_sequence_gt(seq_dir, assets, len(d['pose']))
        n = min(len(d['pose']), n_gt)
        if opt.thresholds:
            pose_w, pel_w = gauge_to_world(d['all_pose'][:n], d['all_trans'][:n], d['E'][0])  # every slot, world
        sel_pose_w, sel_pel_w = gauge_to_world(d['pose'][:n], d['trans'][:n], d['E'][0])     # stored selection
        seq_acc = {t: [] for t in keys}
        for fr in range(n):
            gv = gt_v[fr]
            cache = {}
            for t in keys:
                if t == 'selected':
                    key = 'selected'
                    src = (sel_pose_w[fr], d['beta'][fr], sel_pel_w[fr])
                else:
                    prob = d['all_presence'][fr]
                    sel = np.nonzero(prob > t)[0]
                    if len(sel) == 0:
                        sel = np.array([int(np.argmax(prob))])
                    key = tuple(sel.tolist())
                    src = (pose_w[fr][sel], d['all_beta'][fr][sel], pel_w[fr][sel])
                if key not in cache:
                    pv = world_verts(assets, *src)
                    cost = np.array([[np.linalg.norm(p - g, axis=-1).mean() * 1000 for g in gv] for p in pv])
                    r, c = linear_sum_assignment(cost)
                    cache[key] = (pv, cost, r, c)
                pv, cost, r, c = cache[key]
                for mode, gate in (('nogate', np.inf), ('gated', PVE_MATCH_THRESH_MM)):
                    a = acc[t][mode]
                    a['gt'] += len(gv); a['pred'] += len(pv); a['frames'] += 1
                    for p, g in zip(r, c):
                        if cost[p, g] <= gate:
                            m = proto.fair_regressed_metrics(pv[p], gv[g], assets.smpl_j_regressor, assets.j_regressor)
                            a['pairs'].append(m)
                            if mode == 'nogate':
                                seq_acc[t].append(m['mpjpe22_mm'])
        per_seq[f.stem] = {str(t): float(np.mean(v)) for t, v in seq_acc.items()}
        print(f'{f.stem}: {n} frames', flush=True)
    out = {'dataset': opt.dataset, 'pred_dir': str(opt.pred_dir),
           'protocol': 'supplementary: every slot with presence > threshold (top-1 if none), Hungarian on PVE; '
                       'not the official fixed-count protocol', 'per_sequence_mpjpe22_nogate': per_seq, 'thresholds': {}}
    for t in keys:
        out['thresholds'][str(t)] = {}
        for mode in ('nogate', 'gated'):
            a = acc[t][mode]
            matched = len(a['pairs'])
            res = {k: float(np.mean([p[k] for p in a['pairs']])) for k in KEYS} if matched else {}
            res.update(recall=100 * matched / a['gt'], precision=100 * matched / a['pred'],
                       fp_per_frame=(a['pred'] - matched) / a['frames'], people_per_frame=a['pred'] / a['frames'])
            out['thresholds'][str(t)][mode] = res
            print(f"t={t!s:8s} {mode:6s} MPJPE22 {res.get('mpjpe22_mm', float('nan')):6.2f} PA22 {res.get('pa_mpjpe22_mm', float('nan')):6.2f} "
                  f"MPJPE14 {res.get('mpjpe14_mm', float('nan')):6.2f} recall {res['recall']:5.1f}% precision {res['precision']:5.1f}% "
                  f"FP/frame {res['fp_per_frame']:.2f} people/frame {res['people_per_frame']:.2f}", flush=True)
    opt.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
