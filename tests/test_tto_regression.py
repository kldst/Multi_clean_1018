"""End-to-end TTO on a synthetic scene: one SMPL-X person seen by 4 cameras, masks rendered from the true body, the
"network" start placed 34 cm off (mostly in depth, the network's typical error).

  * behaviour: the fit must pull the pelvis back to within 5 cm and not move a correct body pose
  * regression: the fitted parameters must match tests/fixtures/tto_synthetic_expected.npz (written by
    `python tests/test_tto_regression.py --write-golden` after an intended change)

Needs the SMPL-X model (configs/paths.yaml smplx_root); skipped otherwise. ~1 min on CPU.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from conftest import needs_smplx, ring_rig

GOLDEN = Path(__file__).resolve().parent / 'fixtures/tto_synthetic_expected.npz'
N_FRAMES = 4
PELVIS = np.array([0., 0., 4.])
START_OFFSET = np.array([0.15, 0.0, 0.30])


def synthetic_infer():
    """An inference record (camera 0 = world, as for CHI3D / Harmony4D) with masks rendered from the true body."""
    from multismpl.body.body_model import decode_anchor, init_smplx
    from multismpl.tto.silhouette import IMG, G, project
    init_smplx()
    K, E = ring_rig(target=PELVIS)
    pose = np.zeros((N_FRAMES, 1, 72), np.float32)
    pose[..., 0] = np.pi                                       # upright in camera coordinates (y down), facing camera 0
    pose[:, 0, 3 + 3 * 15 + 2] = np.linspace(-0.3, 0.3, N_FRAMES)  # left shoulder moves a little over time
    beta = np.zeros((N_FRAMES, 1, 10), np.float32)
    pelvis = np.tile(PELVIS, (N_FRAMES, 1, 1)).astype(np.float32)
    with torch.no_grad():
        _, v = decode_anchor(torch.tensor(pose[:, 0]), torch.tensor(beta[:, 0]), torch.tensor(pelvis[:, 0]))
        uv, _ = project(v, torch.tensor(K, dtype=torch.float32), torch.tensor(E, dtype=torch.float32))
    masks = np.zeros((N_FRAMES, 1, len(K), G, G), np.uint8)
    for f in range(N_FRAMES):
        for c in range(len(K)):
            g = ((uv[f, c].numpy() + .5) * G / IMG).astype(int)
            ok = (g >= 0).all(1) & (g < G).all(1)
            masks[f, 0, c][g[ok, 1], g[ok, 0]] = 255
            masks[f, 0, c] = cv2.morphologyEx(masks[f, 0, c], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    d = dict(pose=pose, beta=beta, trans=pelvis + START_OFFSET.astype(np.float32), masks=masks, K=K, E=E)
    return d, pose, pelvis


def run_fit(d):
    from multismpl.tto.fit import SequenceFit
    from multismpl.tto.tracking import tracks_from_slots
    opt = SimpleNamespace(mode='joint', chunk=64, iters_trans=60, iters_global=20, iters=30, w_pose=.1, w_beta=1.,
                          w_smooth=.01, n_cover=256, n_vsub=1500)
    tracks = tracks_from_slots(d, N_FRAMES)
    fitter = SequenceFit(tracks, d['K'], d['E'], opt, torch.device('cpu'))
    fitter.run()
    return fitter.result()


@pytest.fixture(scope='module')
def fitted():
    torch.manual_seed(0)
    d, pose, pelvis = synthetic_infer()
    return run_fit(d), pose, pelvis


@needs_smplx
def test_fit_recovers_the_pelvis(fitted):
    res, _, pelvis = fitted
    start = np.linalg.norm(res['init_anchor'] - pelvis, axis=-1).max()
    end = np.linalg.norm(res['anchor'] - pelvis, axis=-1).max()
    assert start > 0.3 and end < 0.05, (start, end)


@needs_smplx
def test_fit_keeps_a_correct_pose(fitted):
    res, pose, _ = fitted
    assert np.abs(res['pose'][..., 3:66] - pose[..., 3:66]).max() < 0.15     # radians


@needs_smplx
def test_fit_matches_the_stored_result(fitted):
    if not GOLDEN.is_file():
        pytest.skip('no golden file yet: python tests/test_tto_regression.py --write-golden')
    res, _, _ = fitted
    ref = np.load(GOLDEN)
    for k in ('anchor', 'pose', 'beta'):
        np.testing.assert_allclose(res[k], ref[k], atol=1e-3, err_msg=k)


if __name__ == '__main__' and '--write-golden' in sys.argv:
    torch.manual_seed(0)
    r = run_fit(synthetic_infer()[0])
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez(GOLDEN, anchor=r['anchor'], pose=r['pose'], beta=r['beta'])
    print('wrote', GOLDEN, 'pelvis', r['anchor'][:, 0])
