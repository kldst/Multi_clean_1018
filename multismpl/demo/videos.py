"""Gradio demo on any synchronised multi-camera video clips (no GT): pick a clip, a frame and 4 cameras -> the
network's presence probability for every person slot, the shown slots decoded to SMPL-X meshes (overlaid on every view
and in 3D) and their person masks.

  python -m multismpl.demo.videos --video-root <folder of session folders> --calib-dir <camera folder> \
      [--device cuda] [--host 0.0.0.0] [--port 7860]

Layout: <video-root>/<session>/<clip>_<cam>.mp4 (e.g. 241217_1/1_00_01_3.mp4 = clip 1_00_01, camera 3), or pass a
single session folder as --video-root. Calibration (only for the crop intrinsics and the overlays; the network itself
does not take cameras): <calib-dir>/Cam_<cam>_intrinsic.npy ([fx, fy, cx, cy, ...] or 3x3) and Cam_<cam>_extrinsic.npy
(3x4 world -> camera, OpenCV). Without --calib-dir the overlays use a guessed focal (max(W, H)) and identity poses, so
only view 0 is meaningful.

Per frame the contract is the inference scripts': process_one_image 518 crop, causal T=3 window (the two previous
frames of the clip feed the temporal head), camera 0 = the first chosen view = the coordinate frame.
"""
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from multismpl.body.body_model import init_smplx
from multismpl.body.smplx_body import _get_smplx_model
from multismpl.demo.harmony4d import PALETTE, mask_overlay, presence_plot, render, scene_glb
from multismpl.inference.network import CachedTemporalHead, head_meta, image_preprocessor, load_model, person_masks, slot_outputs
from multismpl.paths import MODEL_CONFIG, PATHS

V = 4
SELECT_TOPK, SELECT_THRESH = 'Top-k', 'presence ≥ 閾值'
CLIP_RE = re.compile(r'^(.+)_(\d+)\.mp4$')


def load_calib(calib_dir, cam, wh):
    """(K 3x3, E 3x4 world -> camera) of one camera; a guess when there is no calibration."""
    if calib_dir is None:
        f = float(max(wh))
        return np.array([[f, 0, wh[0] / 2], [0, f, wh[1] / 2], [0, 0, 1]]), np.eye(3, 4)
    k = np.load(calib_dir / f'Cam_{cam}_intrinsic.npy').astype(np.float64)
    if k.size != 9 or k.reshape(3, 3)[2, 2] != 1:          # flat [fx, fy, cx, cy, (distortion...)]
        k = k.reshape(-1)
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    else:
        K = k.reshape(3, 3)
    return K, np.load(calib_dir / f'Cam_{cam}_extrinsic.npy').astype(np.float64).reshape(3, 4)


class Demo:
    def __init__(self, video_root, calib_dir, device, checkpoint=None):
        self.device = torch.device(device)
        self.calib_dir = calib_dir
        sessions = [video_root] if any(video_root.glob('*.mp4')) else sorted(p for p in video_root.iterdir() if p.is_dir())
        self.clips = OrderedDict()                           # '<session>/<clip>' -> {cam: path}
        for s in sessions:
            for p in sorted(s.glob('*.mp4')):
                m = CLIP_RE.match(p.name)
                if m:
                    self.clips.setdefault(f'{s.name}/{m.group(1)}', {})[int(m.group(2))] = p
        self.clips = OrderedDict((k, dict(sorted(v.items()))) for k, v in self.clips.items() if len(v) >= V)
        if not self.clips:
            raise SystemExit(f'{video_root}: no <clip>_<cam>.mp4 groups with >= {V} cameras')
        self.pre = image_preprocessor()
        init_smplx()
        self.body = _get_smplx_model(self.device, 'neutral')
        import pickle
        with open(PATHS.smplx_root / 'neutral' / 'model.pkl', 'rb') as h:
            self.faces = np.asarray(pickle.load(h, encoding='latin1')['f'], np.int64)
        print('loading model ...', flush=True)
        self.model = load_model(MODEL_CONFIG, checkpoint or PATHS.checkpoint, self.device)
        self.meta = head_meta(V, self.device)
        self.info = {}
        self.crops = OrderedDict()       # (clip, cam, frame) -> (518x518x3 uint8, crop K)
        self.tokens = OrderedDict()      # (clip, cams, frame) -> last aggregator layer
        self.results = OrderedDict()

    def clip_info(self, clip):
        """(frame count, (W, H)) of a clip = the shortest of its cameras."""
        if clip not in self.info:
            n, wh = [], None
            for p in self.clips[clip].values():
                cap = cv2.VideoCapture(str(p))
                n.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cap.release()
            self.info[clip] = (min(n), wh)
        return self.info[clip]

    @staticmethod
    def _put(cache, key, value, size):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > size:
            cache.popitem(last=False)

    def crop(self, clip, cam, frame):
        """Decoded frame of one camera -> the 518 crop (uint8) and its intrinsics."""
        key = (clip, cam, frame)
        if key not in self.crops:
            cap = cv2.VideoCapture(str(self.clips[clip][cam]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, im = cap.read()
            cap.release()
            if not ok:
                raise gr.Error(f'{clip} cam {cam}: cannot decode frame {frame}')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            K, E = load_calib(self.calib_dir, cam, (im.shape[1], im.shape[0]))
            out = self.pre.process_one_image(im, None, E, K, np.array(im.shape[:2]), np.array([518, 518]))
            self._put(self.crops, key, (np.ascontiguousarray(out[0]).astype(np.uint8), np.asarray(out[3], np.float64)), 64)
        return self.crops[key]

    def images(self, clip, cams, frame):
        crops = [self.crop(clip, c, frame) for c in cams]
        imgs = torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im, _ in crops])
        return imgs.to(self.device, torch.float32) / 255., crops

    @torch.inference_mode()
    def run(self, clip, cams, frame):
        cams = tuple(cams)
        key = (clip, cams, frame)
        if key in self.results:
            return self.results[key]
        head = CachedTemporalHead(self.model, self.meta)
        for f in range(max(frame - 2, 0), frame):            # history: last-layer tokens of the 2 previous frames
            k = (clip, cams, f)
            if k not in self.tokens:
                imgs, _ = self.images(clip, cams, f)
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    tok, _, _ = self.model.aggregator(imgs.unsqueeze(0))
                self._put(self.tokens, k, tok[-1].detach(), 32)
                del tok
            head.history.append(self.tokens[k])
        imgs, crops = self.images(clip, cams, frame)
        pred, tokens, patch = head(imgs)
        self._put(self.tokens, key, tokens[-1].detach(), 32)
        prob, trans, pose, beta = slot_outputs(pred)
        masks = person_masks(self.model, pred, tokens, imgs[None], patch, list(range(len(prob))))
        del tokens, pred
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        j, v = self.body(t(pose), t(beta), torch.zeros(len(pose), 3, device=self.device))
        verts = (v + (t(trans) - j[:, 0])[:, None]).cpu().numpy()          # camera-0 coordinates
        wh = self.clip_info(clip)[1]
        ext = [load_calib(self.calib_dir, c, wh)[1] for c in cams]
        R0, t0 = ext[0][:, :3], ext[0][:, 3]
        E = np.stack([np.concatenate([e[:, :3] @ R0.T, (e[:, 3] - e[:, :3] @ R0.T @ t0)[:, None]], 1) for e in ext])
        res = dict(prob=prob, trans=trans, verts=verts, masks=masks, K=np.stack([k for _, k in crops]), E=E,
                   images=[im for im, _ in crops])
        self._put(self.results, key, res, 16)
        return res


def build_ui(demo):
    names = list(demo.clips)

    def cams_of(clip):
        return list(demo.clips[clip])

    def on_clip(clip):
        n, _ = demo.clip_info(clip)
        cams = cams_of(clip)
        default = cams[:V] if len(cams) < 8 else [cams[i] for i in np.linspace(0, len(cams) - 1, V).round().astype(int)]
        return [gr.update(maximum=n - 1, value=min(2, n - 1))] + \
               [gr.update(choices=[str(c) for c in cams], value=str(c)) for c in default]

    def compute(clip, frame, c0, c1, c2, c3, mode, k, thresh):
        cams = [int(c) for c in (c0, c1, c2, c3)]
        if len(set(cams)) != V or not all(c in demo.clips[clip] for c in cams):
            raise gr.Error('請選 4 個不同的相機')
        n, _ = demo.clip_info(clip)
        frame = min(max(int(frame), 0), n - 1)
        res = demo.run(clip, cams, frame)
        prob = res['prob']
        order = [int(i) for i in np.argsort(-prob)]
        sel = order[:int(k)] if mode == SELECT_TOPK else [i for i in order if prob[i] >= thresh]
        color = {i: PALETTE[i % len(PALETTE)] for i in sel}
        meshes = [(res['verts'][i], demo.faces, color[i]) for i in sel]
        overlays, masks, raw = [], [], []
        for v in range(V):
            img = res['images'][v]
            label = f'view {v} · cam {cams[v]}' + ('（camera 0 = 座標系）' if v == 0 else '')
            overlays.append((render(img, meshes, res['K'][v], res['E'][v]), label))
            masks.append((mask_overlay(img, res['masks'][sel, v], [color[i] for i in sel]), label))
            raw.append((img, label))
        table = []
        for i in order:
            vis = int(((res['masks'][i] >= 128).sum((-1, -2)) > 300).sum())
            table.append([i, round(float(prob[i]), 4), '✓' if i in sel else '',
                          *[round(float(x), 3) for x in res['trans'][i]], f'{vis}/{V}'])
        info = (f'**{clip}** · 第 {frame} 幀（共 {n} 幀）· 相機 {cams} · T=3 視窗：幀 {max(frame - 2, 0)}–{frame}  \n'
                f'顯示 slot：{sel or "無"}（presence > 0.5 的有 {int((prob > 0.5).sum())} 個）')
        return info, presence_plot(prob, sel, thresh), table, overlays, masks, raw, scene_glb(meshes) if meshes else None

    with gr.Blocks(title='MultiSMPL · multi-view video demo') as ui:
        gr.Markdown('## MultiSMPL · 多視角影片 demo\n選 4 個相機 → 每個 person slot 的 presence 機率、decode 成 SMPL-X mesh'
                    '（網路輸出，無 TTO）。座標系 = view 0 的相機；疊圖用的是相機校正檔。')
        with gr.Row():
            with gr.Column(scale=1):
                clip = gr.Dropdown(names, value=names[0], label='片段（session/clip）')
                frame = gr.Slider(0, 10000, step=1, value=2, label='幀')
                with gr.Row():
                    prev_btn = gr.Button('◀ 上一幀')
                    next_btn = gr.Button('下一幀 ▶')
                cam_dd = [gr.Dropdown([], label=f'view {v}' + ('（camera 0）' if v == 0 else ''), allow_custom_value=True)
                          for v in range(V)]
                mode = gr.Radio([SELECT_TOPK, SELECT_THRESH], value=SELECT_THRESH, label='顯示哪些 slot')
                k = gr.Slider(1, 20, step=1, value=2, label='k（Top-k）')
                thresh = gr.Slider(0.0, 1.0, step=0.01, value=0.5, label='presence 閾值')
                run_btn = gr.Button('執行', variant='primary')
            with gr.Column(scale=2):
                info = gr.Markdown()
                plot = gr.Plot(label='slot presence')
                model3d = gr.Model3D(label='3D（camera-0 座標）', height=420)
        overlay = gr.Gallery(label='預測 SMPL-X mesh 疊圖（518 crop）', columns=4, height='auto')
        mask_gallery = gr.Gallery(label='預測 person mask', columns=4, height='auto')
        raw_gallery = gr.Gallery(label='網路輸入（518 crop）', columns=4, height='auto')
        table = gr.Dataframe(headers=['slot', 'presence', '顯示', 'pelvis x', 'pelvis y', 'pelvis z', 'mask 可見視角'],
                             label='所有 slot（依 presence 排序；pelvis 為 camera-0 座標，公尺）', interactive=False)
        inputs = [clip, frame, *cam_dd, mode, k, thresh]
        outputs = [info, plot, table, overlay, mask_gallery, raw_gallery, model3d]
        clip.input(on_clip, clip, [frame, *cam_dd]).then(compute, inputs, outputs)
        run_btn.click(compute, inputs, outputs)
        frame.release(compute, inputs, outputs)
        mode.input(compute, inputs, outputs)
        for ctrl in (k, thresh):
            ctrl.release(compute, inputs, outputs)
        prev_btn.click(lambda f: max(int(f) - 1, 0), frame, frame).then(compute, inputs, outputs)
        next_btn.click(lambda c, f: min(int(f) + 1, demo.clip_info(c)[0] - 1), [clip, frame], frame) \
            .then(compute, inputs, outputs)
        ui.load(on_clip, clip, [frame, *cam_dd]).then(compute, inputs, outputs)
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video-root', type=Path, required=True)
    ap.add_argument('--calib-dir', type=Path, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--checkpoint', default=None, help='default: configs/paths.yaml checkpoint')
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to serve other machines on the network')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true', help='public gradio.live link')
    opt = ap.parse_args()
    demo = Demo(opt.video_root, opt.calib_dir, opt.device, opt.checkpoint)
    build_ui(demo).queue(default_concurrency_limit=1).launch(server_name=opt.host, server_port=opt.port,
                                                             share=opt.share)


if __name__ == '__main__':
    main()
