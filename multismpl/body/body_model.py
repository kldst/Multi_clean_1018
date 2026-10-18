"""SMPL-X helpers shared by the fit and the evaluators.

Coordinate contract: the network predicts in the gauge of the FIRST input view (camera 0), metric scale;
`mesh_translate` is the pelvis (joint 0) position in that gauge. World = R0^T (X_cam0 - t0) with [R0|t0] the first
view's world->camera extrinsics. (CHI3D and Harmony4D store camera 0 as the world frame, so there E[0] = [I|0].)
"""
import numpy as np
import torch

from multismpl.body.smplx_body import (_decode_smpl_batch, axis_angle_to_rotmat, rotmat_to_axis_angle, set_smpl_model_root,
                                         set_smplx_model_root)
from multismpl.paths import PATHS


def init_smplx():
    """Point the body-model code at configs/paths.yaml (SMPL-X for predictions / MAMMA GT, SMPL for Harmony4D GT)."""
    set_smplx_model_root(str(PATHS.smplx_root))
    set_smpl_model_root(str(PATHS.smpl_root))


def decode_anchor(pose, beta, anchor, with_vertices=True):
    """Neutral SMPL-X with the pelvis (joint 0) placed at `anchor`. pose (N,72), beta (N,10), anchor (N,3) torch.
    Differentiable. Returns joints (N,J,3) and vertices (N,10475,3)."""
    n = pose.shape[0]
    zero = torch.zeros(n, 3, device=pose.device, dtype=pose.dtype)
    j, v = _decode_smpl_batch(pose, beta, zero, ['neutral'] * n, use_mamma=True, with_vertices=with_vertices)
    offset = (anchor - j[:, 0])[:, None]
    return j + offset, (v + offset if v is not None else None)


def gauge_to_world(pose, trans, E0):
    """Camera-0-gauge prediction -> world. pose (...,72) numpy, trans (...,3) pelvis, E0 (3,4) world->cam0.
    Only the global orientation and the pelvis change. Returns (pose_world, pelvis_world)."""
    R0, t0 = E0[:, :3], E0[:, 3]
    p = pose.copy()
    Rg = axis_angle_to_rotmat(torch.tensor(p.reshape(-1, 72)[:, :3], dtype=torch.float64))[:, 0]
    p.reshape(-1, 72)[:, :3] = rotmat_to_axis_angle(torch.tensor(R0.T)[None] @ Rg).numpy()
    return p, (trans - t0) @ R0


def body_up(pose_w, beta, pelvis_w, presence, dev, k=256):
    """Scene up-axis estimated from the network's confident bodies: median unit vector pelvis (joint 0) -> head
    (joint 15) over up to k candidate-frames with presence > 0.5. Arrays are (F,C,...) in the world frame."""
    f, c = np.nonzero(presence > 0.5)
    if len(f) == 0:
        f, c = np.nonzero(presence >= presence.max())
    pick = np.linspace(0, len(f) - 1, min(k, len(f))).astype(int)
    f, c = f[pick], c[pick]
    with torch.no_grad():
        j, _ = decode_anchor(torch.tensor(pose_w[f, c], dtype=torch.float32, device=dev),
                             torch.tensor(beta[f, c], dtype=torch.float32, device=dev),
                             torch.tensor(pelvis_w[f, c], dtype=torch.float32, device=dev))
    u = (j[:, 15] - j[:, 0]).cpu().numpy()
    u /= np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9
    m = np.median(u, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)
