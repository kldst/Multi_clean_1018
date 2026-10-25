"""Harmony4D test metrics for network predictions or fits: exactly Harmony4D_testing/eval/eval_h4d_nano.py's metric.
Neutral SMPL-X decode -> smplx2smpl; H36M-14 rooted on the H36M regressor's pelvis; camera-0 coordinates; Hungarian on
absolute PVE. 'nogate' scores every GT person, 'gated' drops pairs above 500 mm PVE (unmatched = false positive).
Also SMPL-22 and a per-activity breakdown (nogate). Only the evaluator's scored frames (scored_idx) are used.

  python -m multismpl.eval.harmony4d --kind infer --pred-dir <infer_dir> --out net.json
  python -m multismpl.eval.harmony4d --kind fit --pred-dir <fit_dir> --infer-dir <infer_dir> --out fit.json
  python -m multismpl.eval.harmony4d --kind verts --pred-dir <dir> --infer-dir <infer_dir> --out other.json

--kind verts scores any method from its meshes: <pred_dir>/<seq>.npz with verts (F,P,V,3) in the inference run's
camera-0 coordinates (V = 10475 SMPL-X or 6890 SMPL), row = the inference row, optional valid (F,P) bool and views (4,)
(must equal the inference run's). Matching and metrics are the same code path as for our predictions.
"""
import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from multismpl.body.smplx_body import _get_smplx_model, set_smplx_model_root
from multismpl.eval.common import load_smpl_jreg, load_smplx2smpl, smpl22_metrics
from multismpl.eval.protocols.common import PVE_MATCH_THRESH_MM
from multismpl.eval.protocols.harmony4d import aggregate, pair_metrics
from multismpl.paths import PATHS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['infer', 'fit', 'verts'], required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--infer-dir', type=Path, default=None, help='--kind fit: the inference folder (GT, cameras)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-frame', type=Path, default=None, help='also write every scored frame\'s nogate pairs (json)')
    ap.add_argument('--gt-all', type=Path, default=None,
                    help='score EVERY frame with the GT from multismpl.eval.harmony4d_gt (default: the evaluator\'s '
                         '1-in-10 scored frames); not the eval_h4d_nano.py protocol')
    opt = ap.parse_args()
    infer_dir = opt.pred_dir if opt.kind == 'infer' else opt.infer_dir
    set_smplx_model_root(str(PATHS.smplx_root))
    body = _get_smplx_model(torch.device('cpu'), 'neutral')
    smplx2smpl = torch.tensor(load_smplx2smpl())
    jreg = np.load(PATHS.j_regressor_h36m).astype(np.float32)
    smpl_jreg = load_smpl_jreg()
    rec = {m: {'pairs': [], 'fp': 0} for m in ('gated', 'nogate')}
    per_act = defaultdict(lambda: {'pairs': [], 'gt': 0})
    n_gt = n = 0
    frames = []
    for f in sorted(infer_dir.glob('*.npz')):
        if opt.kind != 'infer' and not (opt.pred_dir / f.name).is_file():
            print(f'WARNING {f.stem}: no prediction, sequence skipped', flush=True)
            continue
        g = np.load(f)
        if opt.kind == 'infer':                   # camera-0 coordinates, trans = pelvis
            P_, B_, A_ = g['pose'], g['beta'], g['trans']
        elif opt.kind == 'fit':                   # world = camera 0 (E[0] = [I|0])
            d = np.load(opt.pred_dir / f.name)
            P_, B_, A_ = d['pose'], d['beta'], d['anchor']
            assert np.allclose(g['E'][0], np.eye(3, 4), atol=1e-6)
        else:                                     # meshes in camera-0 coordinates
            d = np.load(opt.pred_dir / f.name)
            VS = d['verts']
            VALID = d['valid'] if 'valid' in d else np.ones(VS.shape[:2], bool)
            if 'views' in d:
                assert list(d['views']) == list(g['views']), (f.stem, d['views'], g['views'])
        act = '_'.join(f.stem.split('_')[:2])
        if opt.gt_all:
            ga = np.load(opt.gt_all / f.name)
            GV, GN, ROWS = ga['gt_v'], ga['gt_n'], np.arange(len(ga['gt_n']))
        else:
            GV, GN, ROWS = g['gt_v'], g['gt_n'], g['scored_idx']
        for k, fr in enumerate(ROWS):
            gv = GV[k][:int(GN[k])]
            with torch.inference_mode():
                if opt.kind == 'verts':
                    pv = torch.tensor(VS[fr][VALID[fr]], dtype=torch.float32)
                else:
                    pj, pv = body(torch.tensor(P_[fr], dtype=torch.float32), torch.tensor(B_[fr], dtype=torch.float32),
                                  torch.zeros(len(P_[fr]), 3))
                    pv = pv + (torch.tensor(A_[fr], dtype=torch.float32) - pj[:, 0])[:, None]
                if pv.shape[1] != 6890:
                    pv = torch.einsum('sv,pvc->psc', smplx2smpl, pv)
                pv = pv.numpy()
            if not len(pv):                       # nothing predicted: every GT person is missed
                frames.append({'seq': f.stem, 'row': int(fr), 'frame_id': int(g['frame_ids'][fr]), 'n_gt': len(gv),
                               'views': g['views'].tolist(), 'pairs': []})
                per_act[act]['gt'] += len(gv)
                n_gt += len(gv)
                n += 1
                continue
            pp = np.stack([(jreg @ v)[0:1] for v in pv])
            gp = np.stack([(jreg @ v)[0:1] for v in gv])
            cost = np.zeros((len(pv), len(gv)))
            for p, q in product(range(len(pv)), range(len(gv))):
                cost[p, q] = np.linalg.norm(pv[p] - gv[q], axis=-1).mean() * 1000
            rows, cols = linear_sum_assignment(cost)
            pairs_f = []
            for mode, gate in (('gated', PVE_MATCH_THRESH_MM), ('nogate', np.inf)):
                matches = [(p, q) for p, q in zip(rows, cols) if cost[p, q] <= gate]
                rec[mode]['fp'] += len(pv) - len(matches)
                for p, q in matches:
                    m = pair_metrics(pv[p], gv[q], pp[p], gp[q], jreg) | smpl22_metrics(pv[p], gv[q], smpl_jreg)
                    rec[mode]['pairs'].append(m)
                    if mode == 'nogate':
                        per_act[act]['pairs'].append(m)
                        pairs_f.append({'pred': int(p), 'gt': int(q)} | m)
            frames.append({'seq': f.stem, 'row': int(fr), 'frame_id': int(g['frame_ids'][fr]), 'n_gt': len(gv),
                           'views': g['views'].tolist(), 'pairs': pairs_f})
            per_act[act]['gt'] += len(gv)
            n_gt += len(gv)
            n += 1
    out = {'kind': opt.kind, 'pred_dir': str(opt.pred_dir), 'clips_scored': n,
           'frames': 'all' if opt.gt_all else 'eval_h4d_nano.py scored frames (every 10th clip)',
           'protocol': 'eval_h4d_nano.py metric (H36M-14, regressed pelvis, cam0, Hungarian on PVE)'}
    for m in rec:
        out[m] = aggregate(rec[m]['pairs'], n_gt, rec[m]['fp'])
        for k in ('smpl22_mpjpe_mm', 'smpl22_pa_mpjpe_mm'):
            out[m][k] = float(np.mean([q[k] for q in rec[m]['pairs']]))
    out['per_activity_nogate'] = {k: aggregate(v['pairs'], v['gt'], 0) for k, v in sorted(per_act.items())}
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps(out, indent=1))
    if opt.per_frame:
        opt.per_frame.write_text(json.dumps(frames))
    for m in ('gated', 'nogate'):
        s = out[m]
        print(f"== {m:6s} clips {n}: MPJPE {s['mpjpe_mm']:.2f}  PA {s['pa_mpjpe_mm']:.2f}  PVE {s['pve_mm']:.2f}  "
              f"PA-PVE {s['pa_pve_mm']:.2f}  Transl {s['transl_mm']:.1f}  R {s['recall']:.1f}%  "
              f"| SMPL-22 MPJPE {s['smpl22_mpjpe_mm']:.2f}  PA {s['smpl22_pa_mpjpe_mm']:.2f}", flush=True)


if __name__ == '__main__':
    main()
