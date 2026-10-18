"""TTO geometry: projection, the robust penalty, mask lookups, depth splatting."""
import numpy as np
import torch

from conftest import project_np
from multismpl.tto.silhouette import IMG, G, depth_map, gm, project, sample_dt


def test_project_matches_pinhole(rig):
    K, E = rig
    X = np.array([[0., 0., 4.], [0.3, -0.5, 4.2], [-0.4, 0.2, 3.7]])
    uv, z = project(torch.tensor(X, dtype=torch.float32)[None], torch.tensor(K, dtype=torch.float32),
                    torch.tensor(E, dtype=torch.float32))
    np.testing.assert_allclose(uv[0].numpy(), project_np(X, K, E), atol=1e-3)
    # every camera looks at the target, which projects to the principal point at distance 4 m
    np.testing.assert_allclose(uv[0, :, 0].numpy(), np.full((4, 2), 258.5), atol=1e-3)
    np.testing.assert_allclose(z[0, :, 0].numpy(), np.full(4, 4.), atol=1e-5)


def test_geman_mcclure_saturates():
    x = torch.tensor([0., 10., 100., 1e4])
    for sigma in (25., 100.):
        v = gm(x, sigma)
        assert v[0] == 0
        assert torch.all(v[1:] > v[:-1])                                   # monotone
        assert v[-1] <= (sigma / 10) ** 2 and v[-1] > 0.99 * (sigma / 10) ** 2   # capped at (sigma / 10 px)^2
    assert abs(float(gm(torch.tensor(10.), 1e9)) - 1.0) < 1e-6            # 10 px = 1 unit when sigma is huge


def test_sample_dt_adds_distance_to_the_image_border():
    dt = torch.zeros(1, 1, G, G)                                           # everywhere "inside the mask"
    uv = torch.tensor([[[[100., 200.], [-100.5, 200.], [IMG - .5 + 30, 200.]]]])
    d = sample_dt(dt, uv)[0, 0]
    np.testing.assert_allclose(d.numpy(), [0., 100., 30.], atol=1e-4)


def test_depth_map_keeps_the_nearest_point():
    uv = torch.tensor([[[[100., 100.], [100., 100.], [300., 300.]]]])
    z = torch.tensor([[[5., 3., 4.]]])
    d = depth_map(uv, z, dil=0)[0, 0]
    gx = int((100 + .5) * G / IMG)
    assert float(d[gx, gx]) == 3.
    assert float(d.max()) == 1e3                                           # empty cells
