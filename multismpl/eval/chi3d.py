"""CHI3D s04 metrics for network predictions or fits: eval/CHI3D/eval_chi3d.py's transcription of MV4DHMR test.py.
SMPL-X -> smplx2smpl -> J_regressor_h36m -> H36M-14, both sides rooted on their own SMPL-X kinematic joint 0;
greedy matching on absolute PVE with a 500 mm gate (--gate-mm inf scores every pair); camera-0 coordinates;
sequences aggregated weighted by matched person-frames. Also SMPL-22 (MAMMA dance definition).

  --kind infer   multismpl.inference.chi3d output (camera 0 = world, no conversion)
  --kind fit     multismpl.tto.run output; --infer-dir gives the sequence names / scored frame ids
  python -m multismpl.eval.chi3d --kind fit --pred-dir <fit_dir> --infer-dir <infer_dir> --out eval_fit.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multismpl.body.smplx_body import _get_smplx_model, set_smplx_model_root
from multismpl.eval.common import load_smpl_jreg, load_smplx2smpl, smpl22_metrics
from multismpl.eval.protocols.chi3d import aggregate, pair_metrics, rotmat_to_aa
from multismpl.eval.protocols.common import PVE_MATCH_THRESH_MM, match_greedy_pve
from multismpl.paths import PATHS

KEYS = ['mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'pa_pve_mm', 'transl_mm', 'mpjpe22_mm', 'smpl22_mpjpe_mm', 'smpl22_pa_mpjpe_mm']


def load_gt(seqroot, name):
    """GT SMPL-X of every person / frame + camera-0 extrinsics (R, T) of the sequence."""
    cam = json.loads((seqroot / 'camera_parameters' / '50591643' / (name + '.json')).read_text())['extrinsics']
    R, T = np.asarray(cam['R']), np.asarray(cam['T']).reshape(3)
    gt = {k: np.asarray(v, dtype=np.float32) for k, v in json.loads((seqroot / 'smplx' / (name + '.json')).read_text()).items()}
    n_people, ngt = gt['betas'].shape[:2]
    rot = np.concatenate([gt['global_orient'], gt['body_pose']], axis=2)
    pose72 = torch.zeros(n_people, ngt, 72)
    pose72[:, :, :66] = rotmat_to_aa(rot).reshape(n_people, ngt, 66)
    return gt, pose72, R, T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', choices=['infer', 'fit'], required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--infer-dir', type=Path, default=None, help='--kind fit: the inference folder (sequence names, frames)')
    ap.add_argument('--dataset-root', type=Path, default=None)
    ap.add_argument('--gate-mm', type=float, default=PVE_MATCH_THRESH_MM, help='PVE matching gate (test.py: 500); inf = no gate')
    opt = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seqroot = (opt.dataset_root or PATHS.chi3d) / 'train/s04'
    smplx2smpl = torch.tensor(load_smplx2smpl(), device=dev)
    j_regressor = np.load(PATHS.j_regressor_h36m).astype(np.float32)
    smpl_jreg = load_smpl_jreg()
    set_smplx_model_root(PATHS.smplx_root)
    body = _get_smplx_model(torch.device('cpu'), 'neutral')

    def to_smpl(v):
        return torch.einsum('sv,pvc->psc', smplx2smpl, torch.as_tensor(v, dtype=torch.float32, device=dev)).cpu().numpy()

    rows = []
    for f in sorted(opt.pred_dir.glob('*.npz')):
        d = dict(np.load(f))
        info = d if opt.kind == 'infer' else dict(np.load((opt.infer_dir or opt.pred_dir) / f.name))
        name, frames = str(info['sequence']), info['frames']
        pose, beta = d['pose'], d['beta']
        pelvis = d['trans'] if opt.kind == 'infer' else d['anchor']
        gt, gt_pose72, R, T = load_gt(seqroot, name)
        n_people = gt['betas'].shape[0]
        n = min(len(pose), len(frames))
        pairs, fp = [], 0
        with torch.inference_mode():
            for i in range(n):
                fr = int(frames[i])
                gj, gv = body(gt_pose72[:, fr], torch.tensor(gt['betas'][:, fr]), torch.tensor(gt['transl'][:, fr]))
                gt_j22 = (gj.numpy()[:, :22] - T) @ R.T
                gt_v_smpl = to_smpl((gv.numpy() - T) @ R.T)
                pj, pv = body(torch.tensor(pose[i], dtype=torch.float32), torch.tensor(beta[i], dtype=torch.float32), torch.zeros(2, 3))
                pj, pv = pj.numpy(), pv.numpy()
                off = (pelvis[i] - pj[:, 0])[:, None]
                pj22, pv_smpl = pj[:, :22] + off, to_smpl(pv + off)
                matches, f_p = match_greedy_pve(pv_smpl, gt_v_smpl, opt.gate_mm)
                fp += f_p
                for p, g in matches:
                    pairs.append(pair_metrics(pv_smpl[p], gt_v_smpl[g], pj22[p, 0:1], gt_j22[g, 0:1], pj22[p], gt_j22[g], j_regressor)
                                 | smpl22_metrics(pv_smpl[p], gt_v_smpl[g], smpl_jreg))
        r = aggregate(pairs, n_people * n, fp)
        r['sequence'] = name
        for k in ('smpl22_mpjpe_mm', 'smpl22_pa_mpjpe_mm'):           # extra keys (aggregate() keeps test.py's)
            r[k] = float(np.mean([q[k] for q in pairs])) if pairs else None
        rows.append(r)
        print(f'{len(rows):3d} {name[:40]:40s} n={n:4d} mpjpe={r["mpjpe_mm"]:.2f} pa={r["pa_mpjpe_mm"]:.2f} '
              f'pve={r["pve_mm"]:.2f} transl={r["transl_mm"]:.1f} recall={r["recall"]:.1f}', flush=True)
    w = np.array([r['matched_person_frames'] for r in rows], float)
    summ = {k: float(np.nansum(np.array([r[k] if r[k] is not None else np.nan for r in rows]) * w) / w.sum()) for k in KEYS}
    m = int(w.sum())
    g = int(sum(r['gt_person_frames'] for r in rows))
    fpt = int(sum(r['false_positives'] for r in rows))
    summ.update(sequences=len(rows), gt_person_frames=g, matched_person_frames=m, false_positives=fpt,
                recall=100 * m / g if g else 0, precision=100 * m / (m + fpt) if m + fpt else 0)
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps({'kind': opt.kind, 'pred_dir': str(opt.pred_dir), 'summary': summ, 'per_sequence': rows}, indent=1))
    print('== ' + ' '.join(f'{k[:-3]}={summ[k]:.2f}' for k in KEYS) + f' recall={summ["recall"]:.2f}% ({m}/{g}) seqs={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
