"""Who is who over time: turn per-frame network slots into P consistent person tracks before the fit.

Two trackers:
  slots       the inference already picked P slots per frame (in arbitrary order). A Viterbi over the P! orderings
              (P = 2: keep / swap) links them into tracks. Used for MAMMA dance, CHI3D and singles (P = 1: nothing to do).
  candidates  "v9": the inference stored C mask-ranked candidates per frame. Each candidate is placed in 3D by
              triangulating its mask centroids; all P tracks are then assigned jointly per frame (Hungarian) with
              motion + mask-IoU + presence costs; a track may stay unassigned (person missing). A start position is
              replaced by the triangulated one when the network's pelvis is > 0.3 m off. Used for Harmony4D and the
              markerless MAMMA scenes (works for any P; needs P given).

Both return a `Tracks` record with world-frame arrays the fit starts from.
"""
import itertools
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from multismpl.body.body_model import body_up, gauge_to_world
from multismpl.tto.triangulation import triangulate_masks


@dataclass
class Tracks:
    pose: np.ndarray            # (F,P,72) world global orient + body pose (hands / face zeroed)
    beta: np.ndarray            # (F,P,10)
    pelvis: np.ndarray          # (F,P,3) world pelvis = the fit's translation ("anchor")
    masks: np.ndarray           # (F,P,V,G,G) uint8 masks of each track (0 where the person is missing)
    perm: np.ndarray            # (F,P) slot ordering chosen by the slot tracker (identity for candidates)
    ambiguous: np.ndarray       # (F,) frames with uncertain identity: no smoothness across them, not used for beta
    missing: np.ndarray         # (F,P) person not found (no mask evidence, filled from neighbouring frames)
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- slot tracker
def track_slots(pelvis, masks, margin_frac=0.25):
    """Viterbi over slot orderings. pelvis (F,P,3) world, masks (F,P,V,G,G).
    Cost per frame = constant-velocity pelvis error (10 cm units) + 2 * (1 - soft mask IoU), summed over people.
    A frame is ambiguous when the chosen ordering's motion cost is within margin_frac of the best alternative, or
    two people are < 15 cm apart. Returns perm (F,P) (track k at frame f = slot perm[f,k]) and ambiguous (F,)."""
    n, P = pelvis.shape[:2]
    if P == 1:
        return np.zeros((n, 1), np.int64), np.zeros(n, bool)
    m = masks.astype(np.float32).reshape(n, P, -1) / 255.

    def iou(a, b):
        return float((np.minimum(a, b)).sum() / max(np.maximum(a, b).sum(), 1e-6))
    perms = np.array(list(itertools.permutations(range(P))))
    S = len(perms)
    cost = np.zeros((n, S))
    back = np.zeros((n, S), np.int64)
    for f in range(1, n):
        for s in range(S):                      # ordering at f
            best, arg = np.inf, 0
            for r in range(S):                  # ordering at f-1
                prev = pelvis[f - 1, perms[r]]
                if f >= 2:                      # constant-velocity prediction (f-2 taken with its own best ordering)
                    pred = prev + (prev - pelvis[f - 2, perms[back[f - 1, r]]])
                else:
                    pred = prev
                cur = pelvis[f, perms[s]]
                c_pos = np.linalg.norm(cur - pred, axis=-1).sum() / 0.10
                c_iou = sum(1 - iou(m[f - 1, perms[r][k]], m[f, perms[s][k]]) for k in range(P)) * 2
                c = cost[f - 1, r] + c_pos + c_iou
                if c < best:
                    best, arg = c, r
            cost[f, s], back[f, s] = best, arg
    state = np.zeros(n, np.int64)
    state[-1] = int(np.argmin(cost[-1]))
    for f in range(n - 1, 0, -1):
        state[f - 1] = back[f, state[f]]
    ambiguous = np.zeros(n, bool)
    for f in range(1, n):
        a0 = pelvis[f - 1, perms[state[f - 1]]]
        costs = np.array([np.linalg.norm(pelvis[f, perms[s]] - a0, axis=-1).sum() for s in range(S)])
        keep = costs[state[f]]
        alt = np.min(np.delete(costs, state[f]))
        close = min(np.linalg.norm(pelvis[f, i] - pelvis[f, j]) for i in range(P) for j in range(i + 1, P))
        ambiguous[f] = abs(alt - keep) < margin_frac * max(keep, 0.02) or close < 0.15
    return perms[state], ambiguous


def tracks_from_slots(d, n):
    """d = inference npz (camera-0 gauge slots)."""
    pose_w, pelvis_w = gauge_to_world(d['pose'][:n], d['trans'][:n], d['E'][0])
    masks = d['masks'][:n]
    perm, amb = track_slots(pelvis_w, masks)
    P = perm.shape[1]
    return _apply_perm(pose_w, d['beta'][:n], pelvis_w, masks, perm, amb, np.zeros((n, P), bool), {})


# ------------------------------------------------------------------------------------------ candidate tracker
def track_candidates_joint(pos, presence, P, support, cmasks, w_motion=0.10, miss_cost=8.0, dup_m=0.25,
                           w_iou=4.0, jump_m=0.35, gap_speed=0.05, max_gap=60):
    """Joint online track-by-detection over the stored candidates.

    pos (F,C,3) world position of every candidate, presence (F,C), support (F,C) = fraction of views where its mask
    has > 300 px, cmasks (F,C,V,G,G). A candidate with support < 0.5 is a phantom (unusable); a usable one gets
    presence' = max(presence, support) (a visible but under-confident person still counts).
    Per frame, duplicates of one person (< dup_m apart, presence order) are removed, then ALL P tracks are assigned
    at once (Hungarian): cost = distance to the track's last position / w_motion + w_iou * (1 - mask IoU with the
    track's last mask) - log presence'; staying unassigned costs miss_cost. A link longer than
    jump_m + gap_speed * (frames since last seen - 1) is forbidden (free again after max_gap missing frames).
    Why jointly: when the confident slot slides from person A onto neighbour B, a one-track-at-a-time tracker for A
    follows it onto B; jointly the presence terms cancel and the smaller total motion wins.
    Start: the frame whose P-th best (usable, mutually > dup_m apart, present within 0.3 m in all frames of +-5)
    candidate is most confident; then track forward and backward from it.
    Returns choice (F,P): candidate index or -1 (missing)."""
    n, C = presence.shape
    usable = support >= 0.5
    pres = np.where(usable, np.maximum(presence, support), presence)
    unary = -np.log(np.clip(pres, 1e-4, 1))
    fg = (cmasks >= 128).reshape(n, C, cmasks.shape[2], -1)
    choice = np.full((n, P), -1, np.int64)

    best, f0 = -1., None
    for f in range(n):
        kept = []
        for c in np.argsort(-presence[f]):
            if usable[f, c] and all(np.linalg.norm(pos[f, c] - pos[f, j]) > dup_m for j in kept) and \
                    all((usable[h] & (np.linalg.norm(pos[h] - pos[f, c], axis=-1) < 0.3)).any()
                        for h in range(max(0, f - 5), min(n, f + 6))):
                kept.append(c)
        if len(kept) >= P and presence[f, kept[P - 1]] > best:
            best, f0, init = float(presence[f, kept[P - 1]]), f, kept[:P]
    if f0 is None:
        return choice
    choice[f0] = init

    for frames in (range(f0 + 1, n), range(f0 - 1, -1, -1)):                  # forward, then backward
        last_f = np.full(P, f0)
        last_c = choice[f0].copy()
        for f in frames:
            cand = []
            for c in np.argsort(-presence[f]):                                  # duplicate suppression
                if usable[f, c] and all(np.linalg.norm(pos[f, c] - pos[f, j]) >= dup_m for j in cand):
                    cand.append(c)
            cand = np.asarray(cand, np.int64)
            if len(cand) == 0:
                continue
            cost = np.full((P, len(cand)), np.inf)
            for k in range(P):
                g = abs(f - last_f[k])
                dist = np.linalg.norm(pos[f, cand] - pos[last_f[k], last_c[k]], axis=-1)
                ok = (dist <= jump_m + gap_speed * (g - 1)) | (g > max_gap)
                c_k = np.where(g > max_gap, 0., dist / w_motion) + unary[f, cand]
                a = fg[last_f[k], last_c[k]][None]
                b = fg[f, cand]
                inter = (a & b).sum(-1)
                union = (a | b).sum(-1)
                c_k = c_k + w_iou * (1 - np.where(union > 0, inter / np.maximum(union, 1), 0).mean(-1))
                cost[k, ok] = c_k[ok]
            big = np.concatenate([np.where(np.isfinite(cost), cost, 1e6),
                                  np.where(np.eye(P, dtype=bool), miss_cost, 1e6)], 1)   # + one "missing" column per track
            r, c = linear_sum_assignment(big)
            for k, j in zip(r, c):
                if j < len(cand) and big[k, j] < 1e5:
                    choice[f, k] = cand[j]
                    last_f[k] = f
                    last_c[k] = cand[j]
    return choice


def tracks_from_candidates(d, n, P, drop_outlier_views=False, up_axis='z', dev='cpu'):
    """d = inference npz with cand_* arrays (camera-0 gauge). P = number of people.
    up_axis: the triangulated silhouette centre is shifted 0.15 m down to ~pelvis height along world z ('z', the
    MAMMA markerless worlds are z-up) or along the up-axis estimated from the bodies ('body', any world frame)."""
    cmasks = d['cand_masks'][:n]
    cpose_w, cpos_w = gauge_to_world(d['cand_pose'][:n], d['cand_trans'][:n], d['E'][0])
    support = ((cmasks >= 128).sum((-1, -2)) > 300).mean(-1)
    # image-supported positions (an under-confident slot can have a good mask but a depth metres off)
    Xt, tri_ok = triangulate_masks(cmasks, d['K'], d['E'], drop_outliers=drop_outlier_views)
    if up_axis == 'body':
        Xt -= 0.15 * tri_ok[..., None] * body_up(cpose_w, d['cand_beta'][:n], cpos_w, d['cand_presence'][:n], dev)
    else:
        Xt[..., 2] -= 0.15 * tri_ok
    pos = np.where(tri_ok[..., None], Xt, cpos_w)
    choice = track_candidates_joint(pos, d['cand_presence'][:n], P, support, cmasks)

    found = choice >= 0
    ci = np.where(found, choice, 0)
    ii = np.arange(n)[:, None]
    beta = d['cand_beta'][:n][ii, ci]
    masks = np.where(found[:, :, None, None, None], cmasks[ii, ci], 0)
    pose_w, pelvis_w = gauge_to_world(d['cand_pose'][:n][ii, ci], d['cand_trans'][:n][ii, ci], d['E'][0])
    # start from the triangulated position when the network pelvis disagrees by > 0.3 m
    Xc, okc = Xt[ii, ci], tri_ok[ii, ci] & found
    far = okc & (np.linalg.norm(pelvis_w - Xc, axis=-1) > 0.3)
    pelvis_w[far] = Xc[far]
    # missing frames: pose / beta from the nearest tracked frame, pelvis linearly interpolated
    for p in range(P):
        good = np.nonzero(found[:, p])[0]
        if len(good) == 0:
            continue
        for f in np.nonzero(~found[:, p])[0]:
            g = good[np.argmin(np.abs(good - f))]
            pose_w[f, p] = pose_w[g, p]
            beta[f, p] = beta[g, p]
            lo, hi = good[good < f], good[good > f]
            pelvis_w[f, p] = pelvis_w[g, p] if not (len(lo) and len(hi)) else \
                pelvis_w[lo[-1], p] + (pelvis_w[hi[0], p] - pelvis_w[lo[-1], p]) * (f - lo[-1]) / (hi[0] - lo[-1])
    close = np.array([min(np.linalg.norm(pelvis_w[f, i] - pelvis_w[f, j]) for i in range(P) for j in range(i + 1, P))
                      for f in range(n)]) if P > 1 else np.full(n, 9.)
    missing = ~found
    amb = (close < 0.15) | missing.any(1)
    stats = {'tracked_by_candidates': True, 'missing_person_frames': int(missing.sum()), 'triangulated_init': int(far.sum())}
    return _apply_perm(pose_w, beta, pelvis_w, masks, np.tile(np.arange(P), (n, 1)), amb, missing, stats)


def _apply_perm(pose_w, beta, pelvis_w, masks, perm, amb, missing, stats):
    idx = np.arange(len(perm))[:, None]
    pose = pose_w[idx, perm]
    pose[..., 66:] = 0                                   # hands / jaw / eyes are not fitted
    return Tracks(pose=pose, beta=beta[idx, perm], pelvis=pelvis_w[idx, perm], masks=masks[idx, perm], perm=perm,
                  ambiguous=amb | missing.any(1), missing=missing, stats=stats)


def stats_json(tracks):
    return json.dumps(tracks.stats)
