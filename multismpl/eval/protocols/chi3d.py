"""CHI3D s04 protocol = MV4DHMR test.py (from eval/CHI3D/eval_chi3d.py; function bodies unchanged).
4 cameras, every 3rd frame (2, 5, 8, ...), camera-0 coordinates, H36M-14 joints with both sides centred on their own
SMPL-X kinematic joint 0, greedy PVE matching with a 500 mm gate."""
import numpy as np
import torch

from multismpl.eval.protocols.common import H36M_TO_J14, procrustes

CAMERAS = ('50591643', '58860488', '60457274', '65906101')


def pair_metrics(pred_v_smpl, gt_v_smpl, pred_pelvis, gt_pelvis, pred_j22, gt_j22, j_regressor):
    """All of test.py's per-pair numbers, in millimetres."""
    pv = pred_v_smpl - pred_pelvis        # centred on each person's own SMPL-X pelvis
    gv = gt_v_smpl - gt_pelvis
    pj = (j_regressor @ pv)[H36M_TO_J14]
    gj = (j_regressor @ gv)[H36M_TO_J14]
    return dict(
        mpjpe_mm=float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000),
        pa_mpjpe_mm=float(np.linalg.norm(procrustes(pj, gj) - gj, axis=-1).mean() * 1000),
        pve_mm=float(np.linalg.norm(pv - gv, axis=-1).mean() * 1000),
        pa_pve_mm=float(np.linalg.norm(procrustes(pv, gv) - gv, axis=-1).mean() * 1000),
        transl_mm=float(np.linalg.norm(pred_pelvis - gt_pelvis) * 1000),
        mpjpe22_mm=float(np.linalg.norm((pred_j22 - pred_pelvis) - (gt_j22 - gt_pelvis), axis=-1).mean() * 1000),
    )


def rotmat_to_aa(rotmats):
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle."""
    flat = torch.as_tensor(rotmats, dtype=torch.float32).reshape(-1, 3, 3)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    axis = torch.stack([flat[:, 2, 1] - flat[:, 1, 2],
                        flat[:, 0, 2] - flat[:, 2, 0],
                        flat[:, 1, 0] - flat[:, 0, 1]], dim=-1)
    denom = torch.where(angle.abs() < 1e-6, torch.ones_like(angle), 2 * torch.sin(angle))
    return (axis / denom[:, None] * angle[:, None]).reshape(rotmats.shape[:-2] + (3,))


def aggregate(pairs, gt_person_frames, false_positives):
    keys = ['mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'pa_pve_mm', 'transl_mm', 'mpjpe22_mm']
    out = {k: (float(np.mean([p[k] for p in pairs])) if pairs else None) for k in keys}
    matched = len(pairs)
    recall = matched / gt_person_frames if gt_person_frames else 0.0
    precision = matched / (matched + false_positives) if matched + false_positives else 0.0
    out.update(gt_person_frames=gt_person_frames, matched_person_frames=matched,
               missed_person_frames=gt_person_frames - matched, false_positives=false_positives,
               recall=100 * recall, precision=100 * precision,
               f1=100 * 2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return out
