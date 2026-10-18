"""Shared helpers for the tests. Everything here is synthetic: no dataset, checkpoint or GPU is needed.
Tests that need the SMPL-X body model are skipped when configs/paths.yaml (or its smplx_root) is missing."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def look_at(center, target, down=(0., 1., 0.)):
    """OpenCV camera (x right, y down, z forward) at `center` looking at `target` -> (3,4) world->camera."""
    center, target, down = (np.asarray(v, np.float64) for v in (center, target, down))
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return np.concatenate([R, (-R @ center)[:, None]], 1)


def ring_rig(target=(0., 0., 4.), n_views=4, f=500.):
    """n_views cameras on a horizontal circle around `target`, camera 0 at the origin looking along +z, so the world
    frame is camera 0 (as for CHI3D / Harmony4D). Returns K (V,3,3) in 518-px crop coordinates and E (V,3,4)."""
    target = np.asarray(target, np.float64)
    r = np.linalg.norm(target)
    E = []
    for v in range(n_views):
        a = 2 * np.pi * v / n_views
        center = target + r * np.array([np.sin(a), 0., -np.cos(a)])
        E.append(look_at(center, target))
    K = np.array([[f, 0., 258.5], [0., f, 258.5], [0., 0., 1.]])
    return np.repeat(K[None], n_views, 0), np.stack(E)


def project_np(X, K, E):
    """(N,3) world points -> (V,N,2) pixels."""
    xc = np.einsum('vij,nj->vni', E[:, :, :3], X) + E[:, None, :, 3]
    uv = xc[..., :2] / xc[..., 2:3]
    return uv * np.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None] + K[:, None, :2, 2]


@pytest.fixture
def rig():
    return ring_rig()


def smplx_available():
    try:
        from multismpl.paths import PATHS
        return (Path(PATHS.smplx_root) / 'neutral' / 'model.pkl').is_file()
    except (FileNotFoundError, KeyError):
        return False


needs_smplx = pytest.mark.skipif(not smplx_available(), reason='SMPL-X model not configured (configs/paths.yaml smplx_root)')
