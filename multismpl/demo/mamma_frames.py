"""Gradio browser for MAMMA singles / dance frames where two slot-selection runs disagree, from STORED inference outputs
(no network, CPU only): pick a frame from the worst-frames list, see the 4 views with GT (grey), run A's selected slot
(blue), run B's selected slot (orange), the stored mask-ranked candidates with their masks, every slot's presence, and
each candidate's distance to the GT person.

  python -m multismpl.demo.mamma_frames --run-a <infer_dir A> --label-a track --run-b <infer_dir B> --label-b top-1 \
      [--dataset singles] [--port 7861] [--host 0.0.0.0]

Run B must have been inferred with --candidates (cand_* arrays). Per-frame errors are the singles / dance evaluator's
own (multismpl.eval.mamma.score_sequence: SMPL-22 MPJPE after Hungarian on the H36M pelvis).
"""
import argparse
import pickle
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from multismpl.body.body_model import gauge_to_world, init_smplx  # noqa: E402
from multismpl.body.smplx_body import _get_smpl_model, _get_smplx_model  # noqa: E402
from multismpl.demo.harmony4d import mask_overlay, render  # noqa: E402
from multismpl.eval.common import prediction_world  # noqa: E402
from multismpl.eval.mamma import score_sequence  # noqa: E402
from multismpl.eval.protocols import mamma as proto  # noqa: E402
from multismpl.inference.network import image_preprocessor  # noqa: E402
from multismpl.paths import PATHS  # noqa: E402

GREY, BLUE, ORANGE = (np.array(c, np.uint8) for c in ([200, 200, 200], [40, 140, 235], [240, 130, 30]))
CAND = np.array([[230, 60, 60], [60, 180, 80], [170, 80, 210], [230, 200, 40], [40, 200, 200], [240, 100, 180]], np.uint8)


class Browser:
    def __init__(self, opt):
        self.opt = opt
        self.root = PATHS.mamma_singles if opt.dataset == 'singles' else PATHS.mamma_dance
        names = sorted(set(f.stem for f in opt.run_a.glob('*.npz')) & set(f.stem for f in opt.run_b.glob('*.npz')))
        self.names = [n for n in names if (self.root / n / 'gt/global.npz').is_file()]
        init_smplx()
        self.body = _get_smplx_model(torch.device('cpu'), 'neutral')
        with open(PATHS.smplx_root / 'neutral' / 'model.pkl', 'rb') as h:
            self.fx = np.asarray(pickle.load(h, encoding='latin1')['f'], np.int64)
        self.fs = np.asarray(_get_smpl_model(torch.device('cpu'), 'neutral').faces, np.int64)
        self.assets = proto.load_eval_assets(str(PATHS.smplx_root), PATHS.smplx2smpl, PATHS.j_regressor_h36m,
                                             PATHS.smpl_neutral, torch.device('cpu'))
        self.pre = image_preprocessor()
        self.a, self.b, self.err, self.gt = {}, {}, {}, {}
        rows = []
        for n in self.names:
            a, b = dict(np.load(opt.run_a / f'{n}.npz')), dict(np.load(opt.run_b / f'{n}.npz'))
            self.a[n], self.b[n] = a, b
            ea, eb = self._per_frame(n, a), self._per_frame(n, b)
            self.err[n] = (ea, eb)
            for f in range(min(len(ea), len(eb))):
                rows.append((n, f, ea[f], eb[f]))
            print(f'{n}: {len(ea)} frames scored', flush=True)
        self.rows = rows

    def _per_frame(self, n, d):
        pose, beta, pelvis = prediction_world(d, 'infer')
        pairs, nn, _ = score_sequence(self.assets, self.root / n, pose, beta, pelvis, len(pose))
        e = np.full(nn, np.nan)
        for p in pairs:                                       # singles: one GT person -> one pair per frame
            e[p['frame']] = np.nanmax([e[p['frame']], p['mpjpe22_mm']])
        return e

    def worst(self, k, sort):
        rows = [r for r in self.rows if not (np.isnan(r[2]) or np.isnan(r[3]))]
        key = (lambda r: r[3] - r[2]) if sort == 'diff' else (lambda r: r[3])
        rows = sorted(rows, key=key, reverse=True)[:int(k)]
        return [[i + 1, r[0].replace('230929_WhiteRabbit_', ''), r[1], round(r[2], 1), round(r[3], 1), round(r[3] - r[2], 1)]
                for i, r in enumerate(rows)], rows

    # ------------------------------------------------------------------ one frame
    def images(self, n, f, d):
        views = [str(v) for v in d['views']]
        meta = 'gt'
        out = []
        for v in views:
            cam = dict(np.load(self.root / n / meta / f'{v}.npz'))
            vd = self.root / n / 'videos'
            vd = vd if vd.is_dir() else self.root / n / 'videos_crf16'
            cap = cv2.VideoCapture(str(vd / f'{v}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            cap.release()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            o = self.pre.process_one_image(im, None, cam['cam_ext'][:3], cam['cam_int'], np.array(im.shape[:2]), np.array([518, 518]))
            out.append(o[0])
        return out, views

    def verts(self, pose, beta, pelvis):
        with torch.inference_mode():
            j, v = self.body(torch.tensor(pose, dtype=torch.float32).reshape(-1, 72),
                             torch.tensor(beta, dtype=torch.float32).reshape(-1, 10), torch.zeros(len(pose), 3))
            v = v + (torch.tensor(pelvis, dtype=torch.float32).reshape(-1, 3) - j[:, 0])[:, None]
        return v.numpy()

    def gt_verts(self, n, f):
        if n not in self.gt:
            _, gv, _, _ = proto.load_sequence_gt(self.root / n, self.assets, 0)
            self.gt[n] = gv
        return self.gt[n][f]

    def frame(self, n, f):
        a, b = self.a[n], self.b[n]
        imgs, views = self.images(n, f, b)
        K, E = b['K'], b['E']
        pa_, ba_, la_ = prediction_world(a, 'infer')
        pb_, bb_, lb_ = prediction_world(b, 'infer')
        va = self.verts(pa_[f], ba_[f], la_[f])
        vb = self.verts(pb_[f], bb_[f], lb_[f])
        cp, cl = gauge_to_world(b['cand_pose'][f], b['cand_trans'][f], b['E'][0])
        vc = self.verts(cp, b['cand_beta'][f], cl)
        gv = self.gt_verts(n, f)
        gt_pel = np.stack([self.assets.j_regressor @ x for x in gv])[:, 0]
        cand_smpl = np.einsum('sv,pvc->psc', self.assets.smplx2smpl.cpu().numpy(), vc)
        cand_pel = np.stack([self.assets.j_regressor @ x for x in cand_smpl])[:, 0]
        dist = np.linalg.norm(cand_pel[:, None] - gt_pel[None], axis=-1).min(1)

        def row(meshes):
            return [render(imgs[v], meshes, K[v], E[v], alpha=0.65) for v in range(len(views))]
        g_gt = row([(x, self.fs, GREY) for x in gv])
        g_sel = row([(va[0], self.fx, BLUE), (vb[0], self.fx, ORANGE)])
        g_cand = row([(vc[i], self.fx, CAND[i % len(CAND)]) for i in range(len(vc))])
        masks = b['cand_masks'][f]                                            # (C,V,G,G)
        g_mask = [mask_overlay(imgs[v], masks[:, v], [CAND[i % len(CAND)] for i in range(len(vc))]) for v in range(len(views))]
        sup = ((masks >= 128).sum((-1, -2)) > 300).mean(-1)
        slot_a, slot_b = int(a['slots'][f][0]), int(b['slots'][f][0])
        table = [[i, int(s), round(float(b['cand_presence'][f][i]), 3), f'{sup[i]:.2f}', round(float(dist[i]), 2),
                  ('A ' if int(s) == slot_a else '') + ('B' if int(s) == slot_b else '')]
                 for i, s in enumerate(b['cand_slots'][f])]
        ea, eb = self.err[n][0][f], self.err[n][1][f]
        info = (f"**{n}** · 第 {f} 幀  \n{self.opt.label_a}（藍）選 slot {slot_a}：MPJPE-22 **{ea:.1f}** mm · "
                f"{self.opt.label_b}（橘）選 slot {slot_b}：MPJPE-22 **{eb:.1f}** mm  \n"
                f"灰色 = GT。下表：B 推論存下的候選（第 0 個 = B 的選擇），support = 剪影 > 300 px 的視角比例，"
                f"距離 = 候選骨盆到最近 GT 的距離（m）。")
        cap = [f'{v}' for v in views]
        return (info, self.presence_plot(b['all_presence'][f] if 'all_presence' in b else None, slot_a, slot_b,
                                         [int(s) for s in b['cand_slots'][f]]),
                list(zip(g_gt, cap)), list(zip(g_sel, cap)), list(zip(g_cand, cap)), list(zip(g_mask, cap)), table)

    @staticmethod
    def presence_plot(prob, slot_a, slot_b, cands):
        fig, ax = plt.subplots(figsize=(7, 2.4), dpi=110)
        if prob is None:
            ax.text(0.5, 0.5, 'no all_presence stored', ha='center')
            return fig
        idx = np.arange(len(prob))
        col = ['#28a0eb' if i == slot_a else '#f08c1e' if i == slot_b else '#999999' if i not in cands else '#c03c3c' for i in idx]
        ax.bar(idx, prob, color=col, edgecolor='black', linewidth=0.4)
        ax.set_xticks(idx); ax.set_ylim(0, 1.05); ax.set_xlabel('slot'); ax.set_ylabel('presence')
        ax.set_title('presence of every slot (blue = A, orange = B, red = other stored candidates)', fontsize=9)
        fig.tight_layout()
        return fig


def build(br, opt):
    with gr.Blocks(title='MAMMA 幀瀏覽') as ui:
        gr.Markdown(f'## MAMMA {opt.dataset}：{opt.label_a}（A，藍） vs {opt.label_b}（B，橘）\n'
                    '左邊選一幀（預設依 B − A 的誤差差距排序）。全部來自已存的推論結果，不重跑網路。')
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    sort = gr.Radio([('B − A 差距', 'diff'), ('B 的誤差', 'b')], value='diff', label='排序')
                    k = gr.Slider(10, 200, value=40, step=10, label='顯示幾幀')
                lst = gr.Dataframe(headers=['#', '序列', '幀', f'A {opt.label_a}', f'B {opt.label_b}', 'B − A'], interactive=False)
            with gr.Column(scale=3):
                info = gr.Markdown()
                plot = gr.Plot()
                cands = gr.Dataframe(headers=['候選', 'slot', 'presence', 'support', '離 GT (m)', '被誰選'], interactive=False)
        g_sel = gr.Gallery(label='A（藍） vs B（橘）', columns=4, height='auto')
        g_gt = gr.Gallery(label='GT（灰）', columns=4, height='auto')
        g_cand = gr.Gallery(label='B 存下的全部候選', columns=4, height='auto')
        g_mask = gr.Gallery(label='候選的剪影（顏色同上）', columns=4, height='auto')

        def refresh(s, kk):
            t, rows = br.worst(kk, s)
            return t, rows

        def pick(rows, evt: gr.SelectData):
            r = rows[evt.index[0]]
            return br.frame(r[0], r[1])

        outs = [info, plot, g_gt, g_sel, g_cand, g_mask, cands]
        sort.change(refresh, [sort, k], [lst, state]); k.release(refresh, [sort, k], [lst, state])
        lst.select(pick, state, outs)
        ui.load(refresh, [sort, k], [lst, state])
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-a', type=Path, required=True)
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--run-b', type=Path, required=True)
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--dataset', choices=['singles', 'dance'], default='singles')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7861)
    opt = ap.parse_args()
    torch.set_num_threads(8)
    br = Browser(opt)
    build(br, opt).queue(default_concurrency_limit=1).launch(server_name=opt.host, server_port=opt.port)


if __name__ == '__main__':
    main()
