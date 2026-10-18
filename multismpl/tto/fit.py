"""Test-time optimisation (TTO) of SMPL-X for every track of one sequence, from the network's own person masks and
the calibrated cameras. No GT, no 2D keypoints, no differentiable renderer.

Parameters (world frame), all optimised together for the whole sequence:
  pelvis (F,P,3), global orient (F,P,3), body pose (F,P,63) per frame; ONE beta (P,10) per person for the sequence.

Loss (per iteration, summed over all frames; the sequence is processed in chunks of 64 frames only to bound memory,
gradients of all chunks are accumulated before a single Adam step):
  inside  every projected vertex should fall inside the person's mask: distance transform outside the mask (px)
  cover   every sampled mask pixel should have a projected vertex nearby: nearest-vertex distance (px)
          both through the Geman-McClure penalty, weighted by a per-(frame, person, view) reliability
  pose    0.1 * |body pose - network pose|^2        orient 1 * |global orient - network orient|^2
  beta    1 * |beta - median network beta|^2         smooth 0.01 * |joint acceleration|^2 (cm^2, not across
                                                      ambiguous frames)
Stages (which parameters move = learning rate > 0; sigma = penalty saturation):
  A 100 it: pelvis only (lr 2e-2), sigma 100 px   -> move the person to the right place first
  B  50 it: + orient + beta (5e-3), sigma 100 px
  C 100 it: + body pose (5e-3), sigma 25 px      -> details only; far-off mask errors are ignored
Several people (mode 'joint'):
  occlusion  a vertex of p is excused from `inside` when it lands on another person's mask (p > 0.5) and that person
             is >= 2 cm nearer there (min-depth splat of the others, refreshed every 10 iterations)
  cover      samples only p's exclusive pixels (see silhouette.silhouette_targets)
Mode 'indep' fits every person alone (ablation).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from multismpl.body.body_model import decode_anchor
from multismpl.tto.silhouette import depth_map, gm, project, sample_dt, sample_map, silhouette_targets

N_VERTS = 10475
# (name, iterations option, sigma px, learning rates for [pelvis, orient, body pose, beta])
STAGES = (('A', 'iters_trans', 100., (2e-2, 0, 0, 0)),
          ('B', 'iters_global', 100., (5e-3, 5e-3, 0, 5e-3)),
          ('C', 'iters', 25., (5e-3, 5e-3, 5e-3, 5e-3)))


class SequenceFit:
    def __init__(self, tracks, K, E, opt, dev):
        """tracks: tracking.Tracks; K (V,3,3) / E (V,3,4) numpy cameras (518 px, world->cam); opt: multismpl.tto.run options."""
        self.opt, self.dev = opt, dev
        self.n, self.P = tracks.pose.shape[:2]
        n, P = self.n, self.P
        self.K = torch.tensor(K, device=dev, dtype=torch.float32)
        self.E = torch.tensor(E, device=dev, dtype=torch.float32)
        self.V = V = self.E.shape[0]
        self.tracks = tracks
        self.joint = opt.mode == 'joint'

        # --- silhouette targets (rng order: person 0 then person 1 ...)
        masks = tracks.masks
        rng = np.random.default_rng(0)
        others = (lambda p: np.max(np.delete(masks, p, axis=1), axis=1)) if P > 1 else (lambda p: np.zeros_like(masks[:, p]))
        tg = [silhouette_targets(masks[:, p], others(p), self.joint, opt.n_cover, rng) for p in range(P)]
        self.dt = torch.tensor(np.stack([x[0] for x in tg], 1), device=dev, dtype=torch.float32)       # (n,P,V,G,G)
        self.cover = torch.tensor(np.stack([x[1] for x in tg], 1), device=dev)                          # (n,P,V,S,2)
        view_ok = ~tracks.missing[:, :, None]                                                          # missing person: no views
        self.valid = torch.tensor(np.stack([x[2] for x in tg], 1) & view_ok, device=dev, dtype=torch.float32)
        self.mprob = torch.tensor(masks, device=dev, dtype=torch.float16)                               # (n,P,V,G,G) 0..255
        self.v_cover = torch.randperm(N_VERTS, generator=torch.Generator().manual_seed(0))[:opt.n_vsub].to(dev)   # cover term
        self.v_inside = torch.randperm(N_VERTS, generator=torch.Generator().manual_seed(1))[:3000].to(dev)       # inside term

        # --- parameters, initialised from the network (world frame)
        keep = ~tracks.ambiguous
        beta0 = np.stack([np.median(tracks.beta[keep if keep.sum() > 5 else slice(None), p], axis=0) for p in range(P)])
        self.pose0 = torch.tensor(tracks.pose, device=dev, dtype=torch.float32)                         # (n,P,72)
        self.orient = self.pose0[..., :3].clone().requires_grad_(True)
        self.body = self.pose0[..., 3:66].clone().requires_grad_(True)
        self.rest = torch.zeros(n, P, 6, device=dev)                                                    # jaw / eyes fixed
        self.pelvis = torch.tensor(tracks.pelvis, device=dev, dtype=torch.float32).requires_grad_(True)
        self.beta_mean = torch.tensor(beta0, device=dev, dtype=torch.float32)
        self.beta = self.beta_mean.clone().requires_grad_(True)                                         # (P,10)
        self.smooth_ok = torch.tensor(~tracks.ambiguous, device=dev, dtype=torch.float32)
        self.chunks = [(s, min(n, s + opt.chunk)) for s in range(0, n, opt.chunk)]
        self.w_in = torch.ones(n, P, V, device=dev) * self.valid
        self.w_cov = self.w_in.clone()

    # ------------------------------------------------------------------------------------------------ helpers
    def decode(self, s, e):
        """Joints / vertices of frames [s,e) for every person: (b,P,J,3), (b,P,10475,3)."""
        b, P = e - s, self.P
        pose = torch.cat([self.orient[s:e], self.body[s:e], self.rest[s:e]], -1).reshape(b * P, 72)
        bt = self.beta[None].expand(b, -1, -1).reshape(b * P, 10)
        j, v = decode_anchor(pose, bt, self.pelvis[s:e].reshape(b * P, 3))
        return j.view(b, P, -1, 3), v.view(b, P, -1, 3)

    def project_inside_vertices(self, v):
        """v (b,P,10475,3) -> uv (b,P,V,N,2), z (b,P,V,N) of the inside-term vertex subset."""
        b = v.shape[0]
        uv, z = project(v.reshape(-1, N_VERTS, 3)[:, self.v_inside], self.K, self.E)
        return uv.view(b, self.P, self.V, -1, 2), z.view(b, self.P, self.V, -1)

    def cover_distances(self, s, e, v_p, p):
        """Distance (px) from each cover sample of person p to the nearest projected vertex: (b*V, n_cover)."""
        uvs, _ = project(v_p[:, self.v_cover], self.K, self.E)
        return torch.cdist(self.cover[s:e, p].reshape(-1, self.opt.n_cover, 2),
                           uvs.reshape(-1, self.opt.n_vsub, 2)).min(-1).values

    def occlusion_excuse(self, s, e, uv, z):
        """(b,P,V,N) True where a vertex is hidden behind another person (excused from the inside term)."""
        exc = torch.zeros_like(z, dtype=torch.bool)
        if not self.joint or self.P == 1:
            return exc
        P = self.P
        for p in range(P):
            depth_others = torch.stack([depth_map(uv[:, q], z[:, q]) for q in range(P) if q != p]).min(0).values
            z_front = sample_map(depth_others, uv[:, p], mode='nearest', pad='border')
            mask_others = torch.stack([sample_map(self.mprob[s:e, q].float() / 255., uv[:, p])
                                       for q in range(P) if q != p]).max(0).values
            exc[:, p] = (mask_others > 0.5) & (z_front < z[:, p] - 0.02)
        return exc

    @torch.no_grad()
    def reliability(self):
        """Per (frame, person, view) weights from how well the current body agrees with that view's mask:
        inside: fraction of vertices inside the mask (< 6 px) or excused; cover: fraction of cover samples with a
        vertex < 6 px. Ramp 50% -> 0.1 ... 85% -> 1, then a 15-frame moving average."""
        n, P, V = self.n, self.P, self.V
        f_in = torch.zeros(n, P, V, device=self.dev)
        f_cov = torch.zeros(n, P, V, device=self.dev)
        for s, e in self.chunks:
            _, v = self.decode(s, e)
            uv, z = self.project_inside_vertices(v)
            exc = self.occlusion_excuse(s, e, uv, z)
            din = torch.stack([sample_dt(self.dt[s:e, p], uv[:, p]) for p in range(P)], 1)
            f_in[s:e] = ((din < 6) | exc).float().mean(-1)
            for p in range(P):
                dd = self.cover_distances(s, e, v[:, p], p)
                f_cov[s:e, p] = (dd < 6).float().reshape(e - s, V, -1).mean(-1)

        def ramp(x):
            return ((x - .5) / .35).clamp(0.1, 1)

        def box(x, k=15):
            return F.avg_pool1d(x.reshape(n, -1).T[:, None], k, 1, k // 2, count_include_pad=False)[:, 0, :n].T.reshape(n, P, V)
        return box(ramp(f_in)) * self.valid, box(ramp(f_cov)) * self.valid

    # --------------------------------------------------------------------------------------------- the loss
    def chunk_loss(self, s, e, sigma, occ):
        """Loss of frames [s,e), already normalised by the WHOLE sequence length (so chunk losses add up to the
        sequence mean). Joints of the 2 frames before s are decoded too, for the acceleration term."""
        n, P, V, opt = self.n, self.P, self.V, self.opt
        lo = max(0, s - 2)
        j, v = self.decode(lo, e)
        vv = v[s - lo:]                                                              # own frames (b,P,10475,3)
        b = e - s
        uv, z = self.project_inside_vertices(vv)
        exc = occ(s, e, uv, z)
        l_in = 0.
        l_cov = 0.
        for p in range(P):
            r = gm(sample_dt(self.dt[s:e, p], uv[:, p]), sigma) * (~exc[:, p]).float()
            l_in = l_in + (r.mean(-1) * self.w_in[s:e, p]).sum()
            dd = self.cover_distances(s, e, vv[:, p], p)
            l_cov = l_cov + (gm(dd, sigma).reshape(b, V, -1).mean(-1) * self.w_cov[s:e, p]).sum()
        norm = n * V * P
        l_in = l_in / norm
        l_cov = l_cov / norm
        l_pose = (self.body[s:e] - self.pose0[s:e, :, 3:66]).square().sum(-1).sum() / (P * n)
        l_orient = (self.orient[s:e] - self.pose0[s:e, :, :3]).square().sum(-1).sum() / (P * n)
        jj = j[..., :24, :]
        if jj.shape[0] >= 3:
            acc = (jj[2:] - 2 * jj[1:-1] + jj[:-2]).norm(dim=-1).square().mean(-1)                     # (b'-2,P)
            ok = (self.smooth_ok[lo + 2:e] * self.smooth_ok[lo + 1:e - 1] * self.smooth_ok[lo:e - 2])[:, None]
            l_smooth = (acc * ok).sum() * 1e4 / max(1, P * (n - 2))
        else:
            l_smooth = torch.zeros((), device=self.dev)
        loss = l_in + l_cov + opt.w_pose * l_pose + l_orient + opt.w_smooth * l_smooth
        return loss, float(l_in), float(l_cov)

    # ---------------------------------------------------------------------------------------------- the run
    def run(self):
        opt = self.opt
        groups = [{'params': [self.pelvis]}, {'params': [self.orient]}, {'params': [self.body]}, {'params': [self.beta]}]
        adam = torch.optim.Adam(groups, lr=0.)
        occ_cache = {}
        t0 = time.time()
        log = []
        for name, iters_opt, sigma, lrs in STAGES:
            for g, lr in zip(groups, lrs):
                g['lr'] = lr
            if name in 'BC':
                self.w_in, self.w_cov = self.reliability()
            for it in range(getattr(opt, iters_opt)):
                def occ(s, e, uv, z):                    # occlusion map refreshed every 10 iterations
                    if it % 10 == 0 or (s, e) not in occ_cache:
                        occ_cache[(s, e)] = self.occlusion_excuse(s, e, uv.detach(), z.detach())
                    return occ_cache[(s, e)]
                adam.zero_grad(set_to_none=True)
                tot = {'inside': 0., 'cover': 0.}
                for s, e in self.chunks:
                    loss, l_in, l_cov = self.chunk_loss(s, e, sigma, occ)
                    loss.backward()                      # accumulate; one Adam step after all chunks
                    tot['inside'] += l_in
                    tot['cover'] += l_cov
                (opt.w_beta * (self.beta - self.beta_mean).square().sum()).backward()
                adam.step()
                if it == 0 or it == getattr(opt, iters_opt) - 1:
                    log.append({'stage': name, 'iter': it, **tot, 't': time.time() - t0})
        return log, time.time() - t0

    def result(self):
        tr = self.tracks
        with torch.no_grad():
            pose = torch.cat([self.orient, self.body, self.rest], -1)
        return {'pose': pose.cpu().numpy(),
                'beta': self.beta[None].expand(self.n, -1, -1).detach().cpu().numpy(),
                'anchor': self.pelvis.detach().cpu().numpy(),
                'init_pose': tr.pose, 'init_beta': tr.beta, 'init_anchor': tr.pelvis,
                'perm': tr.perm, 'ambiguous': tr.ambiguous, 'dropped': tr.missing,
                'w_in_mean': self.w_in.mean((0, 1)).cpu().numpy(), 'w_cov_mean': self.w_cov.mean((0, 1)).cpu().numpy()}
