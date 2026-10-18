"""Step 1 for MAMMA videos (dance, singles, markerless scenes): network inference with the fair evaluator's contract
(eval/mamma_eval_dance/eval_mamma_dance_comparison.py: video decode + process_one_image crop, causal T=3 window,
top-k presence + 3D NMS slot selection incl. its fallback, camera-0 gauge) plus the person masks.

  python -m multismpl.inference.mamma --out <dir>                                     # dance, 2 people
  python -m multismpl.inference.mamma --dataset singles --people 1 --select track --out <dir>     # singles
  python -m multismpl.inference.mamma --seq-dirs <seq> --people 3 --meta-dir meta --video-dir videos_crf24 --candidates 7 --out <dir>
                                                                                      # markerless, for the v9 tracker
Per sequence <out>/<seq>.npz:
  pose (F,P,72) beta (F,P,10) trans (F,P,3) presence (F,P) slots (F,P)   the P selected slots (camera-0 gauge)
  all_presence (F,Q) all_trans (F,Q,3)                                   every slot
  masks (F,P,V,259,259) uint8                                            masks of the selected slots
  cand_pose / cand_beta / cand_trans / cand_presence / cand_slots / cand_masks (F,C,...)   with --candidates C
  K (V,3,3) crop intrinsics (518 px)  E (V,3,4) world->cam  views (V,)
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from multismpl.eval.protocols.common import topk_nms3d
from multismpl.inference.network import CachedTemporalHead, head_meta, load_model, person_masks, rank_candidates_by_mask, slot_outputs
from multismpl.inference.video import video_worker
from multismpl.paths import MODEL_CONFIG, PATHS


class CausalSingleTracker:
    """--select track (one person): bystanders can outrank a bent / occluded subject on presence. Lock on the first
    frame whose best presence >= 0.9, then take argmax presence - 20 * max(0, |pelvis - previous| - 0.35 m)."""

    def __init__(self):
        self.prev = None

    def __call__(self, prob, trans, sel):
        if self.prev is None:
            if prob.max() >= 0.9:
                self.prev = trans[sel[0]]
            return sel
        d = np.linalg.norm(trans - self.prev, axis=-1)
        sel = [int(np.argmax(prob - 20.0 * np.maximum(0.0, d - 0.35)))]
        self.prev = trans[sel[0]]
        return sel


def run_sequence(model, seq_dir, opt, device):
    views, people, C = opt.views, opt.people, opt.candidates
    cams = [dict(np.load(seq_dir / opt.meta_dir / f'{v}.npz')) for v in views]
    n = int(np.load(seq_dir / opt.meta_dir / 'global.npz')['frames_len'])
    if opt.max_frames:
        n = min(n, opt.max_frames)
    ctx = mp.get_context('spawn')
    queues = [ctx.Queue(maxsize=2) for _ in views]
    workers = [ctx.Process(target=video_worker, args=(seq_dir / opt.video_dir / f'{v}.mp4', c, n, q), daemon=True)
               for v, c, q in zip(views, cams, queues)]
    for w in workers:
        w.start()
    V = len(views)
    head = CachedTemporalHead(model, head_meta(V, device))
    tracker = CausalSingleTracker() if opt.select == 'track' else None
    rec = {k: [] for k in ('pose', 'beta', 'trans', 'presence', 'slots', 'masks', 'all_presence', 'all_trans')}
    if C:
        rec.update({f'cand_{k}': [] for k in ('pose', 'beta', 'trans', 'presence', 'slots', 'masks')})
    Ks = [None] * V
    t0 = time.time()
    with torch.inference_mode():
        for f in range(n):
            imgs = []
            for vi, q in enumerate(queues):
                idx, im, K = q.get(timeout=1200)
                if idx != f:
                    raise RuntimeError(f'{seq_dir.name}: expected frame {f}, got {idx}: {im}')
                if K is not None:
                    Ks[vi] = K
                imgs.append(torch.from_numpy(im).permute(2, 0, 1))
            images = torch.stack(imgs).to(device=device, dtype=torch.float32) / 255.
            pred, tokens, patch = head(images)
            prob, tr, pose, beta = slot_outputs(pred)
            sel = topk_nms3d(prob, tr, people)
            if tracker is not None:
                sel = tracker(prob, tr, sel)
            run = list(range(len(prob))) if C else sel                    # candidates need every slot's mask
            m = person_masks(model, pred, tokens, images.unsqueeze(0), patch, run)
            if C:
                csel = rank_candidates_by_mask(m, prob, sel, max(C, people))
                rec['masks'].append(m[sel])
                for k, arr in (('pose', pose), ('beta', beta), ('trans', tr), ('presence', prob), ('masks', m)):
                    rec[f'cand_{k}'].append(arr[csel])
                rec['cand_slots'].append(np.asarray(csel))
            else:
                rec['masks'].append(m[:people])
            rec['pose'].append(pose[sel]); rec['beta'].append(beta[sel]); rec['trans'].append(tr[sel])
            rec['presence'].append(prob[sel]); rec['slots'].append(np.asarray(sel))
            rec['all_presence'].append(prob); rec['all_trans'].append(tr)
            del tokens, pred
    for w in workers:
        w.join(timeout=30)
    E = np.stack([c['cam_ext'][:3].astype(np.float64) for c in cams])
    np.savez_compressed(opt.out / f'{seq_dir.name}.npz', **{k: np.stack(v) for k, v in rec.items()},
                        K=np.stack(Ks), E=E, views=np.array(views))
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset', choices=['dance', 'singles'], default='dance', help='which MAMMA set (paths from configs/paths.yaml)')
    ap.add_argument('--dataset-root', type=Path, default=None, help='folder of sequence folders (overrides --dataset)')
    ap.add_argument('--seq-dirs', type=Path, nargs='*', default=None, help='explicit sequence folders (instead of --dataset-root)')
    ap.add_argument('--sequences', nargs='*', default=None)
    ap.add_argument('--checkpoint', type=Path, default=None, help='default: configs/paths.yaml checkpoint')
    ap.add_argument('--config', type=Path, default=MODEL_CONFIG, help='model config')
    ap.add_argument('--views', nargs='+', default=['IOI_01', 'IOI_05', 'IOI_09', 'IOI_13'])
    ap.add_argument('--people', type=int, default=2, help='slots kept per frame (top-k presence + 3D NMS)')
    ap.add_argument('--select', choices=['presence', 'track'], default='presence',
                    help='track: causal slot tracking for one person (singles), see CausalSingleTracker')
    ap.add_argument('--candidates', type=int, default=0, help='also store C mask-ranked candidates (for run_tto.py --tracker candidates)')
    ap.add_argument('--meta-dir', default='gt', help="camera / global npz folder ('meta' for markerless releases)")
    ap.add_argument('--video-dir', default='videos_crf16')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    opt = ap.parse_args()
    if opt.select == 'track' and (opt.people != 1 or opt.candidates):
        raise SystemExit('--select track is for one person without candidates')
    opt.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(opt.device)
    opt.checkpoint = opt.checkpoint or PATHS.checkpoint
    root = opt.dataset_root or (PATHS.mamma_dance if opt.dataset == 'dance' else PATHS.mamma_singles)
    model = load_model(opt.config, opt.checkpoint, device)
    seqs = opt.seq_dirs or sorted(p for p in root.iterdir() if (p / opt.meta_dir / 'global.npz').is_file())
    if opt.sequences:
        seqs = [s for s in seqs if s.name in opt.sequences]
    log = {}
    for s in seqs:
        if (opt.out / f'{s.name}.npz').is_file():
            print(f'{s.name}: exists, skip', flush=True)
            continue
        n, sec = run_sequence(model, s, opt, device)
        log[s.name] = {'frames': n, 'seconds': sec}
        print(f'{s.name}: {n} frames {sec:.0f}s ({sec / n * 1e3:.0f} ms/frame)', flush=True)
    (opt.out / f'run_{int(time.time())}.json').write_text(
        json.dumps({'views': opt.views, 'checkpoint': str(opt.checkpoint), 'log': log}, indent=1))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
