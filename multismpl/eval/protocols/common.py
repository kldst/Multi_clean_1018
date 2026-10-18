"""Pieces shared by every benchmark protocol (copied verbatim from the evaluators they come from)."""
from itertools import product

import numpy as np

H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]   # MV4DHMR utils/constants.py
PVE_MATCH_THRESH_MM = 500.0                                        # MV4DHMR test.py matching gate
NMS_DIST_M = 0.25


def top2_nms3d(prob, roots):
    """The evaluators' slot selection: slots by presence, skipping any within 0.25 m (3D) of a kept one, until two
    are kept; if fewer than two survive, fill with the next most present slots."""
    ranked = np.argsort(-prob, kind='stable')
    kept = []
    for i in ranked:
        if all(np.linalg.norm(roots[i] - roots[j]) >= NMS_DIST_M for j in kept):
            kept.append(int(i))
            if len(kept) == 2:
                break
    if len(kept) < 2:
        kept += [int(i) for i in ranked if i not in kept][:2 - len(kept)]
    return kept


def topk_nms3d(prob, roots, k):
    """top2_nms3d generalised to k people (k == 2 -> identical)."""
    if k == 2:
        return top2_nms3d(prob, roots)
    ranked = np.argsort(-prob, kind='stable')
    kept = []
    for i in ranked:
        if all(np.linalg.norm(roots[i] - roots[j]) >= NMS_DIST_M for j in kept):
            kept.append(int(i))
            if len(kept) == k:
                break
    if len(kept) < k:
        kept += [int(i) for i in ranked if i not in kept][:k - len(kept)]
    return kept


def procrustes(pred, gt):
    """Umeyama similarity fit of `pred` onto `gt` (what roma.rigid_points_registration(compute_scaling=True) does)."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    x, y = pred - mu_p, gt - mu_g
    u, s, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1., 1., d]) @ u.T
    scale = (s[:2].sum() + d * s[2]) / (x ** 2).sum()
    return scale * (r @ pred.T).T + (mu_g - scale * (r @ mu_p))


def match_greedy_pve(pred_v, gt_v, thresh_mm):
    """MV4DHMR test.py's matcher: greedy on mean vertex distance, rejecting pairs over thresh (a rejected pair leaves
    its GT person unmatched). Returns (matches, n_false_positive)."""
    combs = list(product(range(len(pred_v)), range(len(gt_v))))
    dist = {c: float(np.linalg.norm(pred_v[c[0]] - gt_v[c[1]], axis=-1).mean() * 1000) for c in combs}
    pred_taken, gt_taken, matches, fp = set(), set(), [], 0
    for p, g in sorted(combs, key=lambda c: dist[c]):
        if p in pred_taken or g in gt_taken:
            continue
        if dist[(p, g)] > thresh_mm:
            pred_taken.add(p)
            fp += 1
            continue
        pred_taken.add(p)
        gt_taken.add(g)
        matches.append((p, g))
    return matches, fp
