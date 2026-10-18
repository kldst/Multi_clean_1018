"""Silhouette geometry for the fit: projection, mask lookups, the robust penalty and the per-person mask targets.

Pixel convention: every image coordinate is in the 518x518 network-input crop (pixel centres at integers). The
person masks are stored at G = 259 (half resolution), so mask-grid quantities are rescaled by 518 / G.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG = 518          # network input size (px)
G = 259            # mask grid size


def gm(x, sigma, ref=10.0):
    """Geman-McClure penalty in (x / 10 px)^2 units, saturating at (sigma / 10)^2: ~x^2 below sigma, flat beyond, so a
    wrong mask far away (chair, other person) cannot drag the body."""
    x2 = x.square()
    return x2 * sigma ** 2 / (x2 + sigma ** 2) / ref ** 2


def project(points, K, E):
    """points (B,N,3) world; K (V,3,3); E (V,3,4) world->cam -> uv (B,V,N,2) in 518 px, depth z (B,V,N)."""
    xc = torch.einsum('vij,bnj->bvni', E[:, :, :3], points) + E[None, :, None, :, 3]
    z = xc[..., 2]
    uv = xc[..., :2] / z.clamp_min(1e-4)[..., None]
    uv = uv * torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[None, :, None] + K[None, :, None, :2, 2]
    return uv, z


def sample_map(m, uv, mode='bilinear', pad='border'):
    """Look up a per-view G x G map at 518-px positions. m (B,V,G,G), uv (B,V,N,2) -> (B,V,N)."""
    B, V = m.shape[:2]
    grid = (uv + .5) / IMG * 2 - 1
    out = F.grid_sample(m.reshape(B * V, 1, G, G), grid.reshape(B * V, 1, -1, 2), mode=mode, padding_mode=pad,
                        align_corners=False)
    return out.reshape(B, V, -1)


def sample_dt(dt, uv):
    """Distance (px) from each projected point to the mask: distance transform inside the image, plus the distance
    to the image border for points that fall outside the crop."""
    return sample_map(dt, uv) + (uv - uv.clamp(-.5, IMG - .5)).norm(dim=-1)


def depth_map(uv, z, big=1e3, dil=2):
    """Nearest depth of projected points splatted onto the G grid, dilated by `dil` cells (fills gaps between
    vertices). uv (B,V,N,2), z (B,V,N) -> (B,V,G,G); empty cells = `big`. Detached (only decides who is in front)."""
    B, V, N = z.shape
    gx = ((uv[..., 0] + .5) * G / IMG).long().clamp(0, G - 1)
    gy = ((uv[..., 1] + .5) * G / IMG).long().clamp(0, G - 1)
    flat = (torch.arange(B * V, device=z.device).view(B, V, 1) * G * G + gy * G + gx).reshape(-1)
    d = torch.full((B * V * G * G,), big, device=z.device)
    d.scatter_reduce_(0, flat, z.reshape(-1).detach(), reduce='amin')
    d = d.view(B * V, 1, G, G)
    d = -F.max_pool2d(-d, 2 * dil + 1, 1, dil)
    return d.view(B, V, G, G)


def silhouette_targets(masks_own, masks_others, exclusive_cover, n_cover, rng):
    """Targets of ONE person for every frame / view.

    masks_own (F,V,G,G) uint8 probability*255 of this person; masks_others (F,V,G,G) per-pixel max over everyone else.
    Returns
      dt     (F,V,G,G) distance (518 px, capped at 1036) to this person's confident mask (p >= 0.5) -> inside term
      cover  (F,V,n_cover,2) random mask pixels (518 px) the body must reach -> cover term
      valid  (F,V) the view has >= 20 cover pixels
    exclusive_cover (joint mode): cover pixels must also beat every other person by 0.1, so two people never compete
    for the same pixels."""
    n, V = masks_own.shape[:2]
    scale = IMG / G
    dt = np.empty((n, V, G, G), np.float16)
    cover = np.zeros((n, V, n_cover, 2), np.float32)
    valid = np.zeros((n, V), bool)
    for i in range(n):
        for j in range(V):
            p_own = masks_own[i, j].astype(np.float32) / 255.
            fg = p_own >= 0.5
            if not fg.any():            # empty mask (person missing in this view): no inside term
                dt[i, j] = 0
                continue
            dt[i, j] = np.minimum(cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5) * scale, 1036)
            if exclusive_cover:
                fg = fg & (p_own - masks_others[i, j].astype(np.float32) / 255. > 0.1)
            ys, xs = np.nonzero(fg)
            if len(ys) < 20:
                continue
            pick = rng.choice(len(ys), n_cover, replace=len(ys) < n_cover)
            cover[i, j, :, 0] = (xs[pick] + .5) * scale - .5
            cover[i, j, :, 1] = (ys[pick] + .5) * scale - .5
            valid[i, j] = True
    return dt, cover, valid
