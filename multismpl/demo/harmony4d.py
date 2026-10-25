"""Gradio demo on the Harmony4D test split: pick a sequence, a frame and 4 cameras -> the network's presence
probability for every person slot, the chosen slots decoded to SMPL-X meshes (overlaid on every view and in 3D), their
person masks, and the GT with this frame's H36M-14 metrics.

  python -m multismpl.demo.harmony4d [--device cuda] [--port 7860] [--share] [--run-dir outputs/<benchmark run>]

Per frame the contract is multismpl.inference.harmony4d's: Harmony4DDataset loader (fisheye undistortion, 518 crop),
causal T=3 window of the same consecutive run (aggregator tokens of the two previous frames are cached), camera 0 =
the first chosen view = the coordinate frame. The default views are the evaluator's seeded draw, so 'top-2' at a
scored frame reproduces the network numbers of multismpl.eval.harmony4d.

--run-dir: a finished benchmark run (infer/, fit/, per_frame_{network,fit}.json from
`multismpl.eval.harmony4d --per-frame`) adds a worst-frames browser and the TTO meshes of that run. For every GT person
the demo also reports the closest of all 20 slots and its presence rank, which tells a wrong slot choice (a good slot
existed but was not in the top 2) from a bad prediction (no slot was close).
"""
import argparse
import json
import random
from pathlib import Path
import tempfile
from collections import OrderedDict

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch
import trimesh
from scipy.optimize import linear_sum_assignment

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from multismpl.body.body_model import init_smplx  # noqa: E402
from multismpl.body.smplx_body import _get_smpl_model, _get_smplx_model  # noqa: E402
from multismpl.eval.common import load_smplx2smpl  # noqa: E402
from multismpl.eval.protocols.harmony4d import pair_metrics  # noqa: E402
from multismpl.inference.harmony4d import V, cameras_in_cam0, gt_vertices, load_dataset  # noqa: E402
from multismpl.inference.network import CachedTemporalHead, head_meta, load_model, person_masks, slot_outputs  # noqa: E402
from multismpl.paths import MODEL_CONFIG, PATHS  # noqa: E402

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 220, 220],
                    [240, 50, 230], [160, 200, 40], [255, 100, 140], [0, 128, 128], [120, 90, 255], [170, 110, 40],
                    [255, 200, 0], [128, 0, 0], [40, 230, 130], [128, 128, 0], [255, 140, 90], [0, 60, 200],
                    [200, 0, 120], [90, 160, 255]], np.uint8)   # saturated: distinct from the grey GT
GT_COLOR = np.array([200, 200, 200], np.uint8)
TTO_COLORS = np.array([[255, 140, 0], [0, 170, 255], [120, 220, 60], [230, 60, 200]], np.uint8)
SELECT_TOP2, SELECT_THRESH = 'Top-2（評估協定）', 'presence ≥ 閾值'
SORT_NET, SORT_TTO, SORT_GAP = 'Network MPJPE', 'TTO MPJPE', 'Network − 候選 oracle（選錯人嫌疑）'
WRONG_PICK_MM = 30      # a non-selected slot this much closer (PVE) to a GT person than the matched one = wrong pick


class Demo:
    def __init__(self, device, checkpoint=None, run_dir=None):
        self.device = torch.device(device)
        print('loading Harmony4D test ...', flush=True)
        self.ds, _ = load_dataset(PATHS.harmony4d_test, stride=10)
        self.names = [r.name.replace('/', '_') for r in self.ds.sequences]
        self.sample_of = {(s, f): i for i, (s, f) in enumerate(self.ds.samples)}
        self.frames = [sorted(r.frames, key=self.ds._numeric_frame_id) for r in self.ds.sequences]
        self._avail = {}
        init_smplx()
        self.body = _get_smplx_model(self.device, 'neutral')
        self.faces_x = np.asarray(self._smplx_faces(), np.int64)
        self.faces_s = np.asarray(_get_smpl_model(torch.device('cpu'), 'neutral').faces, np.int64)
        self.smplx2smpl = torch.tensor(load_smplx2smpl(), device=self.device)
        self.jreg = np.load(PATHS.j_regressor_h36m).astype(np.float32)
        print('loading model ...', flush=True)
        self.model = load_model(MODEL_CONFIG, checkpoint or PATHS.checkpoint, self.device)
        self.meta = head_meta(V, self.device)
        self.tokens = OrderedDict()      # (seq, views, frame) -> last aggregator layer (temporal head history)
        self.results = OrderedDict()     # (seq, views, frame) -> network outputs + GT of that frame
        self.run_dir = Path(run_dir) if run_dir else None
        self.worst = self._load_run() if self.run_dir else []

    @torch.inference_mode()
    def decode(self, pose, beta, pelvis):
        """Neutral SMPL-X with the pelvis at `pelvis`, as multismpl.eval.harmony4d decodes. numpy -> torch (N,10475,3)."""
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        j, v = self.body(t(pose), t(beta), torch.zeros(len(pose), 3, device=self.device))
        return v + (t(pelvis) - j[:, 0])[:, None]

    def pair(self, pv, gv):
        """pair_metrics of one SMPL-topology prediction / GT pair (eval's pelvis = H36M regressor joint 0)."""
        return pair_metrics(pv, gv, (self.jreg @ pv)[0:1], (self.jreg @ gv)[0:1], self.jreg)

    # ---------------------------------------------------------------- benchmark run
    def _load_run(self):
        """Per scored frame: network / TTO MPJPE (eval's nogate pairs) and the best the 4 stored candidates allow."""
        net = json.loads((self.run_dir / 'per_frame_network.json').read_text())
        fit_path = self.run_dir / 'per_frame_fit.json'
        fit = {(r['seq'], r['row']): r for r in json.loads(fit_path.read_text())} if fit_path.is_file() else {}
        cache = self.run_dir / 'per_frame_oracle.json'
        oracle = {tuple(k.split('|')): v for k, v in json.loads(cache.read_text()).items()} if cache.is_file() else {}
        if not oracle:
            print('candidate oracle per scored frame (cached in per_frame_oracle.json) ...', flush=True)
            for f in sorted((self.run_dir / 'infer').glob('*.npz')):
                g = np.load(f)
                for k, row in enumerate(g['scored_idx']):
                    gv = g['gt_v'][k][:int(g['gt_n'][k])]
                    cv = torch.einsum('sv,pvc->psc', self.smplx2smpl, self.decode(
                        g['cand_pose'][row], g['cand_beta'][row], g['cand_trans'][row])).cpu().numpy()
                    cost = np.array([[np.linalg.norm(c - x, axis=-1).mean() for x in gv] for c in cv])
                    rows, cols = linear_sum_assignment(cost)
                    oracle[(f.stem, str(int(row)))] = float(np.mean([self.pair(cv[p], gv[q])['mpjpe_mm']
                                                                     for p, q in zip(rows, cols)]))
            cache.write_text(json.dumps({'|'.join(k): v for k, v in oracle.items()}))
        out = []
        for r in net:
            key = (r['seq'], r['row'])
            mean = lambda rec: float(np.mean([p['mpjpe_mm'] for p in rec['pairs']])) if rec and rec['pairs'] else None
            out.append({'seq': r['seq'], 'row': r['row'], 'frame_id': r['frame_id'], 'views': r['views'],
                        'net': mean(r), 'tto': mean(fit.get(key)), 'oracle': oracle.get((r['seq'], str(r['row']))),
                        'net_pp': [round(p['mpjpe_mm']) for p in sorted(r['pairs'], key=lambda p: p['gt'])]})
        return out

    def tto_vertices(self, seq_i, row, views):
        """The run's TTO bodies at this frame (world = camera 0) when the demo shows the run's views, else None."""
        if not self.run_dir:
            return None
        name = self.names[seq_i]
        fi, ff = self.run_dir / 'infer' / f'{name}.npz', self.run_dir / 'fit' / f'{name}.npz'
        if not ff.is_file() or list(np.load(fi)['views']) != list(views):
            return None
        d = np.load(ff)
        if row >= len(d['pose']):
            return None
        return self.decode(d['pose'][row], d['beta'][row], d['anchor'][row]).cpu().numpy()

    @staticmethod
    def _smplx_faces():
        import pickle
        with open(PATHS.smplx_root / 'neutral' / 'model.pkl', 'rb') as h:
            return pickle.load(h, encoding='latin1')['f']

    # ---------------------------------------------------------------- data
    def available_cameras(self, seq_i):
        """Cameras present in every frame (the evaluator draws its 4 views among these)."""
        if seq_i not in self._avail:
            rec = self.ds.sequences[seq_i]
            self._avail[seq_i] = [c for c, cam in enumerate(rec.cameras)
                                  if all(self.ds._view_available(rec, cam, f) for f in self.frames[seq_i])]
        return self._avail[seq_i]

    def default_views(self, seq_i):
        return [int(i) for i in np.random.default_rng(seq_i).permutation(self.available_cameras(seq_i))[:V]]

    def window(self, seq_i, row):
        """Rows of the causal T=3 window ending at `row`, cut at the start of its consecutive run."""
        fid = self.ds._numeric_frame_id
        fr = self.frames[seq_i]
        rows = [row]
        while len(rows) < 3 and rows[0] > 0 and fid(fr[rows[0]]) - fid(fr[rows[0] - 1]) == 1:
            rows.insert(0, rows[0] - 1)
        return rows

    def load(self, seq_i, row, views):
        frame = self.frames[seq_i][row]
        np.random.seed(row)                      # as multismpl.inference.harmony4d
        random.seed(row)
        b = self.ds._get_single_frame_data(seq_index=self.sample_of[(seq_i, frame)], ids=list(views), img_per_seq=V)
        imgs = torch.as_tensor(np.asarray(b['images'])).to(device=self.device, dtype=torch.float32)
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        if imgs.max() > 1.5:
            imgs = imgs / 255.
        return b, imgs

    @staticmethod
    def _put(cache, key, value, size):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > size:
            cache.popitem(last=False)

    # ---------------------------------------------------------------- network
    @torch.inference_mode()
    def run(self, seq_i, row, views):
        views = tuple(views)
        key = (seq_i, views, row)
        if key in self.results:
            self.results.move_to_end(key)
            return self.results[key]
        rows = self.window(seq_i, row)
        head = CachedTemporalHead(self.model, self.meta)
        for r in rows[:-1]:                      # history: last-layer tokens of the previous frames
            k = (seq_i, views, r)
            if k not in self.tokens:
                _, imgs = self.load(seq_i, r, views)
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    tok, _, _ = self.model.aggregator(imgs.unsqueeze(0))
                self._put(self.tokens, k, tok[-1].detach(), 64)
                del tok
            head.history.append(self.tokens[k])
        b, imgs = self.load(seq_i, row, views)
        pred, tokens, patch = head(imgs)
        self._put(self.tokens, key, tokens[-1].detach(), 64)
        prob, trans, pose, beta = slot_outputs(pred)
        masks = person_masks(self.model, pred, tokens, imgs[None], patch, list(range(len(prob))))
        del tokens, pred
        verts = self.decode(pose, beta, trans)                                                # camera-0 coordinates
        verts_smpl = torch.einsum('sv,pvc->psc', self.smplx2smpl, verts).cpu().numpy()
        K, E, R0, t0 = cameras_in_cam0(b)
        gv, gn = gt_vertices(b, R0, t0, self.smplx2smpl, self.device)
        gv = gv[:gn]
        slot_pve = np.array([[np.linalg.norm(p - g, axis=-1).mean() * 1000 for g in gv] for p in verts_smpl])
        img_u8 = (imgs.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().cpu().numpy()
        res = dict(prob=prob, trans=trans, verts=verts.cpu().numpy(), verts_smpl=verts_smpl, masks=masks,
                   K=K, E=E, gt=gv, slot_pve=slot_pve.reshape(len(prob), gn), images=img_u8, window=rows,
                   frame=self.frames[seq_i][row], cams=[self.ds.sequences[seq_i].cameras[c].name for c in views],
                   tto=self.tto_vertices(seq_i, row, views))
        self._put(self.results, key, res, 16)
        return res

    # ---------------------------------------------------------------- metrics
    def match(self, res, sel):
        """This frame's Hungarian matching on PVE (multismpl.eval.harmony4d) -> {slot: (gt index, metrics)}."""
        pv, gv = res['verts_smpl'][sel], res['gt']
        if not len(pv) or not len(gv):
            return {}
        rows, cols = linear_sum_assignment(res['slot_pve'][sel])
        return {sel[p]: (q, self.pair(pv[p], gv[q])) for p, q in zip(rows, cols)}

    def diagnose(self, res, sel, matched):
        """Per GT person: the slot it got, and the closest of all slots (with its presence rank) -> verdict."""
        rank = {int(s): r + 1 for r, s in enumerate(np.argsort(-res['prob']))}
        got = {q: s for s, (q, _) in matched.items()}
        pve = res['slot_pve']
        lines = []
        for q in range(len(res['gt'])):
            best = int(np.argmin(pve[:, q]))
            bp = pve[best, q]
            s = got.get(q)
            if s is not None and pve[s, q] - bp <= WRONG_PICK_MM:
                verdict = '✅ 選到最接近的 slot' + ('（但預測本身誤差大）' if bp > 150 else '')
            elif best not in sel:
                verdict = f'⚠️ **選錯人**：slot {best}（第 {rank[best]} 名）更接近，但沒被選進來'
            else:
                verdict = f'⚠️ 最接近的 slot {best} 被配給了另一個 GT（被選的 slot 擠在同一人附近）'
            mine = '沒有' if s is None else f'slot {s}（presence 第 {rank[s]} 名）PVE {pve[s, q]:.0f} mm'
            lines.append(f'- **GT {q}**：配到 {mine}；20 個 slot 中最接近 = slot {best}（presence {res["prob"][best]:.2f}，'
                         f'第 {rank[best]} 名）PVE {bp:.0f} mm → {verdict}')
        for s in sel:
            if pve[s].min() > 500:
                lines.append(f'- ❗ 被選的 slot {s}（presence {res["prob"][s]:.2f}）離所有 GT 都超過 0.5 m '
                             f'（最近 {pve[s].min() / 1000:.1f} m）→ **旁觀者 / 非 GT 的人、假人，或位置偏太遠**')
        return '\n'.join(lines)


# -------------------------------------------------------------------- drawing
def project(X, K, E):
    """Camera-0 points (N,3) -> pixels (N,2) and depth (N,) in a view with cam0 -> cam extrinsics E (3,4)."""
    Xc = X @ E[:, :3].T + E[:, 3]
    uv = Xc @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), Xc[:, 2]


def render(image, meshes, K, E, alpha=0.75):
    """Painter's-algorithm mesh overlay: meshes = [(verts (N,3) cam0, faces (F,3), rgb)], flat Lambert shading."""
    out = image.copy()
    layer = image.copy()
    polys, depth, colors = [], [], []
    for verts, faces, rgb in meshes:
        Xc = verts @ E[:, :3].T + E[:, 3]
        if (Xc[:, 2] <= 0.05).any():
            continue
        uv, _ = project(verts, K, E)
        tri = Xc[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        c = tri.mean(1)
        facing = -(n * c).sum(-1) / np.linalg.norm(c, axis=-1)            # > 0: front face
        keep = facing > 0
        shade = 0.35 + 0.65 * facing[keep]
        polys.append(uv[faces[keep]])
        depth.append(c[keep, 2])
        colors.append(shade[:, None] * rgb[None].astype(np.float32))
    if not polys:
        return out
    polys, depth, colors = np.concatenate(polys), np.concatenate(depth), np.concatenate(colors)
    covered = np.zeros(image.shape[:2], np.uint8)
    for i in np.argsort(-depth):                                         # far -> near
        p = np.round(polys[i] * 4).astype(np.int32)                      # 2 bits of sub-pixel precision
        col = tuple(int(x) for x in colors[i])
        cv2.fillConvexPoly(layer, p, col, lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(covered, p, 255, shift=2)
    m = (covered > 0)[..., None]
    out = np.where(m, (alpha * layer + (1 - alpha) * image).astype(np.uint8), image)
    return out


def mask_overlay(image, masks, colors):
    """masks (k,G,G) uint8 for one view -> image tinted by every mask >= 0.5, colored per slot."""
    out = image.astype(np.float32) * 0.45
    H, W = image.shape[:2]
    for m, rgb in zip(masks, colors):
        a = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.
        a = np.where(a >= 0.5, a, 0) * 0.75
        out = out * (1 - a) + rgb[None, None].astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def presence_plot(prob, sel, thresh):
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=110)
    idx = np.arange(len(prob))
    colors = [PALETTE[i % len(PALETTE)] / 255. if i in sel else (0.75, 0.75, 0.75) for i in idx]
    ax.bar(idx, prob, color=colors, edgecolor='black', linewidth=0.4)
    ax.axhline(thresh, color='black', linestyle='--', linewidth=0.8)
    for i in sel:
        ax.text(i, prob[i] + 0.02, f'{prob[i]:.2f}', ha='center', fontsize=7)
    ax.set_xticks(idx)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('slot')
    ax.set_ylabel('presence')
    ax.set_title('presence probability per person slot (colored = shown)', fontsize=9)
    fig.tight_layout()
    return fig


def scene_glb(meshes):
    """Meshes in camera-0 coordinates (OpenCV: y down, z forward) -> .glb in a y-up viewer frame."""
    flip = np.diag([1., -1., -1.])
    scene = trimesh.Scene()
    for i, (verts, faces, rgb) in enumerate(meshes):
        m = trimesh.Trimesh(verts @ flip, faces, process=False)
        m.visual.face_colors = np.concatenate([rgb, [255]]).astype(np.uint8)
        scene.add_geometry(m, node_name=f'mesh{i}')
    path = tempfile.NamedTemporaryFile(suffix='.glb', delete=False).name
    scene.export(path)
    return path


# -------------------------------------------------------------------- UI
def build_ui(demo):
    def seq_index(name):
        return demo.names.index(name)

    def camera_updates(s, views):
        cams = demo.ds.sequences[s].cameras
        choices = [cams[c].name for c in demo.available_cameras(s)]
        return [gr.update(choices=choices, value=cams[c].name) for c in views]

    def on_sequence(name):
        s = seq_index(name)
        n = len(demo.frames[s])
        return [gr.update(maximum=n - 1, value=min(2, n - 1))] + camera_updates(s, demo.default_views(s))

    def compute(name, row, c0, c1, c2, c3, mode, thresh, show_gt):
        s = seq_index(name)
        cams = [c.name for c in demo.ds.sequences[s].cameras]
        names = [c0, c1, c2, c3]
        if len(set(names)) != V or not all(n in cams for n in names):
            raise gr.Error('請選 4 個不同的相機')
        views = [cams.index(n) for n in names]
        row = min(max(int(row), 0), len(demo.frames[s]) - 1)
        res = demo.run(s, row, views)
        prob = res['prob']
        if mode == SELECT_TOP2:
            sel = [int(i) for i in np.argsort(-prob)[:2]]
        else:
            sel = [int(i) for i in np.argsort(-prob) if prob[i] >= thresh]
        matched = demo.match(res, sel)
        color = {i: PALETTE[i % len(PALETTE)] for i in sel}

        pred_meshes = [(res['verts'][i], demo.faces_x, color[i]) for i in sel]
        gt_meshes = [(g, demo.faces_s, GT_COLOR) for g in res['gt']]
        tto_meshes = [] if res['tto'] is None else \
            [(v, demo.faces_x, TTO_COLORS[k % len(TTO_COLORS)]) for k, v in enumerate(res['tto'])]
        overlays, gts, masks, ttos = [], [], [], []
        for v in range(V):
            img = res['images'][v]
            label = f'view {v} · {res["cams"][v]}' + (' (camera 0 = 座標系)' if v == 0 else '')
            overlays.append((render(img, pred_meshes, res['K'][v], res['E'][v]), label))
            masks.append((mask_overlay(img, res['masks'][sel, v], [color[i] for i in sel]), label))
            if show_gt:
                gts.append((render(img, gt_meshes, res['K'][v], res['E'][v]), label + ' · GT'))
            if tto_meshes:
                ttos.append((render(img, tto_meshes, res['K'][v], res['E'][v]), label + ' · TTO'))
        glb = scene_glb(pred_meshes + (gt_meshes if show_gt else []))

        table = []
        for i in np.argsort(-prob):
            i = int(i)
            vis = int(((res['masks'][i] >= 128).sum((-1, -2)) > 300).sum())
            q, m = matched.get(i, (None, None))
            near = '' if not len(res['gt']) else \
                f"GT {int(np.argmin(res['slot_pve'][i]))} · {res['slot_pve'][i].min():.0f}"
            table.append([i, round(float(prob[i]), 4), '✓' if i in sel else '',
                          *[round(float(x), 3) for x in res['trans'][i]], f'{vis}/{V}', near,
                          '' if q is None else f'GT {q}',
                          *(['', '', '', ''] if m is None else
                            [round(m['mpjpe_mm'], 1), round(m['pa_mpjpe_mm'], 1), round(m['pve_mm'], 1),
                             round(m['transl_mm'], 1)])])
        n_gt = len(res['gt'])
        if matched:
            ms = [m for _, m in matched.values()]
            avg = {k: np.mean([m[k] for m in ms]) for k in ('mpjpe_mm', 'pa_mpjpe_mm', 'pve_mm')}
            score = (f"本幀 network (H36M-14，Hungarian on PVE，無門檻)：MPJPE **{avg['mpjpe_mm']:.1f}** mm · "
                     f"PA-MPJPE **{avg['pa_mpjpe_mm']:.1f}** mm · PVE **{avg['pve_mm']:.1f}** mm · "
                     f"配對 {len(matched)}/{n_gt} 個 GT")
        else:
            score = f'本幀沒有可配對的人（選中 {len(sel)} 個 slot，GT {n_gt} 人）'
        if res['tto'] is not None and n_gt:
            tv = torch.einsum('sv,pvc->psc', demo.smplx2smpl, torch.tensor(res['tto'], device=demo.device)).cpu().numpy()
            cost = np.array([[np.linalg.norm(p - g, axis=-1).mean() for g in res['gt']] for p in tv])
            r_, c_ = linear_sum_assignment(cost)
            tm = [demo.pair(tv[p], res['gt'][q]) for p, q in zip(r_, c_)]
            score += (f"  \n本幀 TTO：MPJPE **{np.mean([m['mpjpe_mm'] for m in tm]):.1f}** mm · PA-MPJPE "
                      f"**{np.mean([m['pa_mpjpe_mm'] for m in tm]):.1f}** mm（track → GT："
                      + ', '.join(f'{p}→{q}' for p, q in zip(r_, c_)) + '）')
        elif demo.run_dir:
            score += '  \n（TTO mesh 只在視角 = 該次 run 的視角時顯示）'
        fid = demo.ds._numeric_frame_id(res['frame'])
        window = ', '.join(demo.frames[s][r] for r in res['window'])
        info = (f"**{name}** · frame `{res['frame']}`（第 {row} 幀，共 {len(demo.frames[s])} 幀，id {fid}）· "
                f"T=3 視窗：{window}  \n選中 slot：{sel or '無'}  \n{score}\n\n**選人診斷**\n"
                f"{demo.diagnose(res, sel, matched) if n_gt else '（本幀沒有 GT）'}")
        return (info, presence_plot(prob, sel, thresh), table, overlays, masks, glb,
                gr.update(value=gts, visible=bool(show_gt)), gr.update(value=ttos, visible=bool(ttos)))

    # ---------------------------------------------------------------- worst frames
    activities = sorted({'_'.join(w['seq'].split('_')[:2]) for w in demo.worst})

    def worst_list(sort_by, act, n):
        rows = [w for w in demo.worst if act == '全部' or w['seq'].startswith(act + '_')]
        key = {SORT_NET: lambda w: w['net'] or 0, SORT_TTO: lambda w: w['tto'] or 0,
               SORT_GAP: lambda w: (w['net'] or 0) - (w['oracle'] or 0)}[sort_by]
        rows = sorted(rows, key=key, reverse=True)[:int(n)]
        fmt = lambda x: '' if x is None else round(x, 1)
        table = [[k + 1, w['seq'], w['row'], w['frame_id'], fmt(w['net']), ' / '.join(map(str, w['net_pp'])),
                  fmt(w['tto']), fmt(w['oracle']), fmt((w['net'] or 0) - (w['oracle'] or 0))]
                 for k, w in enumerate(rows)]
        return table, rows

    def jump(rows, evt: gr.SelectData):
        w = rows[evt.index[0]]
        s = seq_index(w['seq'])
        n = len(demo.frames[s])
        return ([gr.update(value=w['seq']), gr.update(maximum=n - 1, value=w['row'])] +
                camera_updates(s, w['views']) + [gr.update(value=SELECT_TOP2)])

    headers = ['slot', 'presence', '顯示', 'pelvis x', 'pelvis y', 'pelvis z', 'mask 可見視角', '最近 GT · PVE',
               '配對', 'MPJPE', 'PA-MPJPE', 'PVE', 'Transl']
    with gr.Blocks(title='MultiSMPL · Harmony4D demo') as ui:
        gr.Markdown('## MultiSMPL · Harmony4D test demo\n4 個校正過的視角 → 每個 person slot 的 presence 機率、'
                    'decode 成 SMPL-X mesh（網路輸出）。單位：公尺 / mm，座標系 = view 0 的相機。')
        worst_rows = gr.State([])
        with gr.Accordion(f'最差幀（{demo.run_dir}）' if demo.run_dir else '最差幀（啟動時加 --run-dir 才有）',
                          open=bool(demo.run_dir), visible=bool(demo.run_dir)):
            gr.Markdown('點一列跳到該幀（自動設成該次 run 的視角與 Top-2）。**候選 oracle** = 推論存下的 4 個候選裡'
                        '最佳配對的 MPJPE；Network 遠大於 oracle → 好的候選存在但沒被選進 top-2（選錯人）。')
            with gr.Row():
                sort_by = gr.Radio([SORT_NET, SORT_TTO, SORT_GAP], value=SORT_NET, label='排序')
                act = gr.Dropdown(['全部'] + activities, value='全部', label='活動')
                topn = gr.Slider(5, 200, step=5, value=30, label='顯示幾幀')
            worst = gr.Dataframe(headers=['#', '序列', '第幾幀', 'frame id', 'Network MPJPE', '每人 (GT0 / GT1)',
                                          'TTO MPJPE', '候選 oracle', 'Network − oracle'], interactive=False)
        with gr.Row():
            with gr.Column(scale=1):
                seq = gr.Dropdown(demo.names, value=demo.names[0], label='影片（序列）')
                frame = gr.Slider(0, max(map(len, demo.frames)) - 1, step=1, value=2, label='幀')
                with gr.Row():
                    prev_btn = gr.Button('◀ 上一幀')
                    next_btn = gr.Button('下一幀 ▶')
                cam_dd = [gr.Dropdown([], label=f'view {v}' + ('（camera 0）' if v == 0 else ''), allow_custom_value=True)
                          for v in range(V)]      # choices are filled per sequence; compute() validates the names
                mode = gr.Radio([SELECT_TOP2, SELECT_THRESH], value=SELECT_THRESH, label='顯示哪些 slot')
                thresh = gr.Slider(0.0, 1.0, step=0.01, value=0.5, label='presence 閾值')
                show_gt = gr.Checkbox(True, label='顯示 GT（灰色）')
                run_btn = gr.Button('執行', variant='primary')
            with gr.Column(scale=2):
                info = gr.Markdown()
                plot = gr.Plot(label='slot presence')
                model3d = gr.Model3D(label='3D（彩色 = 預測，灰色 = GT）', height=420)
        overlay = gr.Gallery(label='預測 SMPL-X mesh 疊圖', columns=4, height='auto')
        gt_gallery = gr.Gallery(label='GT mesh 疊圖', columns=4, height='auto')
        tto_gallery = gr.Gallery(label='TTO 結果（橘 = track 0，藍 = track 1）', columns=4, height='auto', visible=False)
        mask_gallery = gr.Gallery(label='預測 person mask', columns=4, height='auto')
        table = gr.Dataframe(headers=headers, label='所有 slot（依 presence 排序；最近 GT = PVE 最小的 GT 人；'
                                                     '配對 / 誤差 = 本幀 Hungarian 配對）', interactive=False)

        inputs = [seq, frame, *cam_dd, mode, thresh, show_gt]
        outputs = [info, plot, table, overlay, mask_gallery, model3d, gt_gallery, tto_gallery]
        seq.input(on_sequence, seq, [frame, *cam_dd]).then(compute, inputs, outputs)
        run_btn.click(compute, inputs, outputs)
        frame.release(compute, inputs, outputs)
        for ctrl in (mode, show_gt):
            ctrl.input(compute, inputs, outputs)
        thresh.release(compute, inputs, outputs)
        prev_btn.click(lambda f: max(int(f) - 1, 0), frame, frame).then(compute, inputs, outputs)
        next_btn.click(lambda n, f: min(int(f) + 1, len(demo.frames[seq_index(n)]) - 1), [seq, frame], frame) \
            .then(compute, inputs, outputs)
        for ctrl in (sort_by, act, topn):
            ctrl.change(worst_list, [sort_by, act, topn], [worst, worst_rows])
        worst.select(jump, worst_rows, [seq, frame, *cam_dd, mode]).then(compute, inputs, outputs)
        ui.load(on_sequence, seq, [frame, *cam_dd]).then(compute, inputs, outputs)
        if demo.run_dir:
            ui.load(worst_list, [sort_by, act, topn], [worst, worst_rows])
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--checkpoint', default=None, help='default: configs/paths.yaml checkpoint')
    ap.add_argument('--run-dir', type=Path, default=None,
                    help='finished benchmark run (infer/, fit/, per_frame_*.json): worst frames + TTO meshes')
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to serve other machines on the network')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true', help='public gradio.live link')
    opt = ap.parse_args()
    demo = Demo(opt.device, opt.checkpoint, opt.run_dir)
    build_ui(demo).queue(default_concurrency_limit=1).launch(server_name=opt.host, server_port=opt.port,
                                                             share=opt.share)


if __name__ == '__main__':
    main()
