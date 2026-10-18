"""Harmony4D test protocol = Harmony4D_testing/eval/eval_h4d_nano.py's metric (helpers from RICH/eval/eval_rich_nano.py;
function bodies unchanged). H36M-14 rooted on the H36M regressor's pelvis, camera-0 coordinates, Hungarian on PVE."""
import numpy as np

from multismpl.eval.protocols.common import H36M_TO_J14, procrustes


def pair_metrics(pv, gv, p_pelvis, g_pelvis, j_regressor):
    pj = (j_regressor @ pv)[H36M_TO_J14]
    gj = (j_regressor @ gv)[H36M_TO_J14]
    pjc, gjc = pj - p_pelvis, gj - g_pelvis
    pvc, gvc = pv - p_pelvis, gv - g_pelvis
    return dict(
        mpjpe_mm=float(np.linalg.norm(pjc - gjc, axis=-1).mean() * 1000),
        pa_mpjpe_mm=float(np.linalg.norm(procrustes(pjc, gjc) - gjc, axis=-1).mean() * 1000),
        pve_mm=float(np.linalg.norm(pvc - gvc, axis=-1).mean() * 1000),
        pa_pve_mm=float(np.linalg.norm(procrustes(pvc, gvc) - gvc, axis=-1).mean() * 1000),
        transl_mm=float(np.linalg.norm(p_pelvis - g_pelvis) * 1000))


def aggregate(pairs, gt_person_frames, fp):
    keys = ['mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm', 'pa_pve_mm', 'transl_mm']
    out = {k: (float(np.mean([p[k] for p in pairs])) if pairs else None) for k in keys}
    m = len(pairs)
    rec = m / gt_person_frames if gt_person_frames else 0.0
    prec = m / (m + fp) if m + fp else 0.0
    out.update(gt_person_frames=gt_person_frames, matched_person_frames=m,
               missed_person_frames=gt_person_frames - m, false_positives=fp,
               recall=100 * rec, precision=100 * prec,
               f1=100 * 2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out
