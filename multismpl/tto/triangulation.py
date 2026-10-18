"""3D position of each mask candidate from its silhouette centroids in the calibrated views (DLT).

The centroid is a silhouette centre (~torso), not the pelvis; it is used to associate candidates over time and as
a fallback start position when the network's depth is far off.
"""
import numpy as np

from multismpl.tto.silhouette import G, IMG


def dlt(uvs, Pms):
    """Linear triangulation of one point. uvs [(u,v)] in px, Pms [(3,4)] projection matrices. None if degenerate."""
    rows = []
    for (u, w), Pv in zip(uvs, Pms):
        rows += [u * Pv[2] - Pv[0], w * Pv[2] - Pv[1]]
    h = np.linalg.svd(np.stack(rows))[2][-1]
    return h[:3] / h[3] if abs(h[3]) > 1e-9 else None


def reprojection_errors(X, uvs, Pms):
    out = []
    for (u, w), Pv in zip(uvs, Pms):
        q = Pv @ np.r_[X, 1.]
        out.append(np.hypot(q[0] / q[2] - u, q[1] / q[2] - w) if q[2] > 1e-6 else np.inf)
    return np.array(out)


def dlt_drop_clear_outliers(uvs, Pms, abs_px=100., ratio=0.5):
    """All-view DLT that drops a view only when it is a CLEAR outlier (leave-one-out).

    If the all-view solution misses some view by > abs_px, re-solve without each view in turn; drop view i when the
    remaining views then agree within ratio x the all-view max error AND view i is still > abs_px off. Repeat while
    >= 3 views remain. (One bad centroid spreads its error over every view of the all-view solution, so a per-view
    test on that solution cannot find it; normal 30-50 px centroid noise is never removed -- unlike RANSAC, which
    did remove it and made the hugging sequence worse.)"""
    uvs, Pms = list(uvs), list(Pms)
    X = dlt(uvs, Pms)
    while X is not None and len(uvs) >= 3:
        r = reprojection_errors(X, uvs, Pms)
        if r.max() <= abs_px:
            break
        best = None
        for i in range(len(uvs)):
            ou, op = uvs[:i] + uvs[i + 1:], Pms[:i] + Pms[i + 1:]
            Xi = dlt(ou, op)
            if Xi is None:
                continue
            rest_err = reprojection_errors(Xi, ou, op).max()
            own_err = reprojection_errors(Xi, [uvs[i]], [Pms[i]])[0]
            if best is None or rest_err < best[0]:
                best = (rest_err, own_err, i, Xi)
        if best is None or not (best[0] < ratio * r.max() and best[1] > abs_px):
            break
        del uvs[best[2]]
        del Pms[best[2]]
        X = best[3]
    return X


def triangulate_masks(cmasks, K, E, min_px=300, drop_outliers=False):
    """cmasks (F,C,V,G,G) uint8 candidate masks, K (V,3,3) 518 px, E (V,3,4) world->cam.
    A view supports a candidate when its mask has > min_px pixels (G grid). Returns X (F,C,3) world and ok (F,C)
    (>= 2 supporting views). drop_outliers: see dlt_drop_clear_outliers (else plain all-view DLT)."""
    n, C, V = cmasks.shape[:3]
    sc = IMG / G
    Pm = np.stack([K[v] @ E[v] for v in range(V)])
    X = np.zeros((n, C, 3))
    ok = np.zeros((n, C), bool)
    for f in range(n):
        for c in range(C):
            uvs, vs = [], []
            for v in range(V):
                ys, xs = np.nonzero(cmasks[f, c, v] >= 128)
                if len(ys) > min_px:
                    uvs.append(((xs.mean() + .5) * sc - .5, (ys.mean() + .5) * sc - .5))
                    vs.append(v)
            if len(vs) < 2:
                continue
            Xf = dlt_drop_clear_outliers(uvs, list(Pm[vs])) if drop_outliers else dlt(uvs, Pm[vs])
            if Xf is not None:
                X[f, c] = Xf
                ok[f, c] = True
    return X, ok
