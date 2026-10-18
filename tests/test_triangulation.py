"""Mask-centroid triangulation, including the clear-outlier rule that fixed the hugging sequence."""
import numpy as np

from conftest import project_np
from multismpl.tto.silhouette import IMG, G
from multismpl.tto.triangulation import dlt, dlt_drop_clear_outliers, triangulate_masks

X_TRUE = np.array([0.3, -0.2, 4.1])


def views(rig):
    K, E = rig
    Pm = [K[v] @ E[v] for v in range(len(K))]
    return Pm, [tuple(u) for u in project_np(X_TRUE[None], K, E)[:, 0]]


def test_dlt_recovers_the_point(rig):
    Pm, uvs = views(rig)
    np.testing.assert_allclose(dlt(uvs, Pm), X_TRUE, atol=1e-6)


def test_one_clearly_wrong_view_is_dropped(rig):
    Pm, uvs = views(rig)
    uvs[2] = (uvs[2][0] + 200., uvs[2][1])                 # e.g. a mask blob on the wrong person in view 2
    assert np.linalg.norm(dlt(uvs, Pm) - X_TRUE) > 0.05     # plain DLT is pulled away
    np.testing.assert_allclose(dlt_drop_clear_outliers(uvs, Pm), X_TRUE, atol=1e-6)


def test_normal_centroid_noise_is_kept(rig):
    Pm, uvs = views(rig)
    uvs[1] = (uvs[1][0] + 30., uvs[1][1] - 20.)             # ordinary silhouette-centroid noise
    np.testing.assert_allclose(dlt_drop_clear_outliers(uvs, Pm), dlt(uvs, Pm), atol=1e-12)


def test_triangulate_masks_from_discs(rig):
    K, E = rig
    uv = project_np(X_TRUE[None], K, E)[:, 0]
    yy, xx = np.mgrid[:G, :G]
    masks = np.zeros((1, 2, len(K), G, G), np.uint8)        # 1 frame, 2 candidates
    for v, (u, w) in enumerate(uv):
        cx, cy = (u + .5) * G / IMG - .5, (w + .5) * G / IMG - .5
        masks[0, 0, v][(xx - cx) ** 2 + (yy - cy) ** 2 < 25 ** 2] = 255
    masks[0, 1, 0, :5, :5] = 255                            # candidate 1: too small everywhere -> not triangulated
    X, ok = triangulate_masks(masks, K, E)
    assert ok.tolist() == [[True, False]]
    assert np.linalg.norm(X[0, 0] - X_TRUE) < 0.02          # disc centroid quantised on the 259 grid
