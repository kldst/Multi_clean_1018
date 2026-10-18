"""Step 1 for CHI3D s04: network inference with the exact contract of eval/CHI3D/eval_chi3d.py (4 cameras, every
3rd frame scored, T = 3 consecutive frames ending at the scored one, process_one_image with identity extrinsics and
intrinsics_wo_distortion, top2_nms3d, camera-0 coordinates) plus the person masks on the scored frame.

  python -m multismpl.inference.chi3d --out <dir> [--candidates 4]

Per sequence <out>/<safe_name>.npz (camera 0 = world frame):
  pose (F,2,72) beta (F,2,10) trans (F,2,3) presence slots masks (F,2,4,259,259)   F = scored frames
  cand_* (F,C,...) with --candidates C
  K (4,3,3) crop intrinsics (518 px)  E (4,3,4) cam0 -> cam c ([I|0] for camera 0)  frames (F,) scored frame ids  sequence
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from multismpl.eval.protocols.chi3d import CAMERAS
from multismpl.eval.protocols.common import top2_nms3d
from multismpl.inference.network import head_meta, image_preprocessor, load_model, person_masks, rank_candidates_by_mask, slot_outputs
from multismpl.paths import MODEL_CONFIG, PATHS


def safe(name):
    return ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_')


def cameras(seqroot, name):
    """Raw intrinsics (undistorted model) and every camera's extrinsics in camera-0 coordinates."""
    Ks, Rs, Ts = [], [], []
    for cam in CAMERAS:
        p = json.loads((seqroot / 'camera_parameters' / cam / (name + '.json')).read_text())
        K = np.eye(3, dtype=np.float64)
        K[0, 0], K[1, 1] = np.asarray(p['intrinsics_wo_distortion']['f']).reshape(2)
        K[:2, 2] = np.asarray(p['intrinsics_wo_distortion']['c']).reshape(2)
        Ks.append(K)
        Rs.append(np.asarray(p['extrinsics']['R'], np.float64))
        Ts.append(np.asarray(p['extrinsics']['T'], np.float64).reshape(3))
    # CHI3D convention x_c = R_c (X - T_c); camera-0 coords x0 -> X = R0^T x0 + T0
    R0, T0 = Rs[0], Ts[0]
    E = np.stack([np.concatenate([Rc @ R0.T, (Rc @ (T0 - Tc))[:, None]], 1) for Rc, Tc in zip(Rs, Ts)])
    return Ks, E


def run_sequence(model, pre, meta, seqroot, name, opt, device):
    gt = json.loads((seqroot / 'smplx' / (name + '.json')).read_text())
    ngt = int(np.asarray(gt['betas']).shape[1])
    caps = [cv2.VideoCapture(str(seqroot / 'videos' / cam / (name + '.mp4'))) for cam in CAMERAS]
    expected = min(ngt, int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))) // 3
    if opt.max_samples_per_sequence:
        expected = min(expected, opt.max_samples_per_sequence)
    Kraw, E = cameras(seqroot, name)
    Kcrop = [None] * 4
    keys = ['pose', 'beta', 'trans', 'presence', 'slots', 'masks', 'frames']
    if opt.candidates:
        keys += [f'cand_{k}' for k in ('pose', 'beta', 'trans', 'presence', 'slots', 'masks')]
    rec = {k: [] for k in keys}
    history = deque(maxlen=3)
    with torch.inference_mode():
        for frame in range(expected * 3):
            imgs = []
            for ci, (cap, K) in enumerate(zip(caps, Kraw)):
                ok, bgr = cap.read()
                assert ok, f'{name}: decode failed at frame {frame}'
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                o = pre.process_one_image(rgb, None, np.eye(4, dtype=np.float32)[:3], K.astype(np.float32),
                                          np.asarray(rgb.shape[:2]), np.asarray([518, 518]))
                if Kcrop[ci] is None:
                    Kcrop[ci] = np.asarray(o[3], np.float64)
                imgs.append(torch.from_numpy(np.ascontiguousarray(o[0])).permute(2, 0, 1))
            history.append(torch.stack(imgs))
            if frame % 3 != 2:                       # scored frames: 2, 5, 8, ...
                continue
            images = torch.stack(tuple(history)).to(device=device, dtype=torch.float32) / 255.   # (T,V,3,H,W)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                tokens, patch, _ = model.aggregator(images)          # the evaluator runs the 3 frames together
                with torch.autocast(device.type, enabled=False):
                    pred = model.smpl_multi_query_trans_rot_head([tokens[-1]], patch_start_idx=patch, smpl_inputs=meta)
            prob, tr, pose, beta = slot_outputs(pred)
            sel = top2_nms3d(prob, tr)
            cur = [t[-1:] for t in tokens]                           # masks on the scored frame only
            m = person_masks(model, pred, cur, images[-1:], patch, list(range(len(prob))) if opt.candidates else sel)
            if opt.candidates:
                csel = rank_candidates_by_mask(m, prob, sel, opt.candidates)
                for k, arr in (('pose', pose), ('beta', beta), ('trans', tr), ('presence', prob), ('masks', m)):
                    rec[f'cand_{k}'].append(arr[csel])
                rec['cand_slots'].append(np.asarray(csel))
                m = m[sel]
            rec['pose'].append(pose[sel]); rec['beta'].append(beta[sel]); rec['trans'].append(tr[sel])
            rec['presence'].append(prob[sel]); rec['slots'].append(np.asarray(sel)); rec['masks'].append(m)
            rec['frames'].append(frame)
            del tokens, pred
    for cap in caps:
        cap.release()
    np.savez_compressed(opt.out / (safe(name) + '.npz'), **{k: np.stack(v) for k, v in rec.items()}, K=np.stack(Kcrop), E=E,
                        views=np.array(CAMERAS), sequence=np.array(name))
    return expected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, default=None, help='default: configs/paths.yaml chi3d')
    ap.add_argument('--checkpoint', type=Path, default=None, help='default: configs/paths.yaml checkpoint')
    ap.add_argument('--config', type=Path, default=MODEL_CONFIG, help='model config')
    ap.add_argument('--sequences', nargs='*', default=None)
    ap.add_argument('--candidates', type=int, default=0, help='also store C mask-ranked candidates (for run_tto.py --tracker candidates)')
    ap.add_argument('--max-sequences', type=int, default=0)
    ap.add_argument('--max-samples-per-sequence', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    opt = ap.parse_args()
    opt.out.mkdir(parents=True, exist_ok=True)
    opt.checkpoint = opt.checkpoint or PATHS.checkpoint
    seqroot = (opt.dataset_root or PATHS.chi3d).resolve() / 'train/s04'
    names = sorted(p.stem for p in (seqroot / 'smplx').glob('*.json'))
    if opt.sequences:
        names = [n for n in names if n in opt.sequences or safe(n) in opt.sequences]
    if opt.max_sequences:
        names = names[:opt.max_sequences]
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device(opt.device)
    pre = image_preprocessor()
    model = load_model(opt.config, opt.checkpoint, device)
    meta = head_meta(4, device)
    log = {}
    for seqno, name in enumerate(names, 1):
        if (opt.out / (safe(name) + '.npz')).is_file():
            print(f'{name}: exists, skip', flush=True)
            continue
        t0 = time.time()
        expected = run_sequence(model, pre, meta, seqroot, name, opt, device)
        sec = time.time() - t0
        log[name] = {'scored_frames': expected, 'seconds': sec}
        print(f'{seqno}/{len(names)} {name}: {expected} scored frames {sec:.0f}s', flush=True)
    (opt.out / f'run_{int(time.time())}.json').write_text(json.dumps({'checkpoint': str(opt.checkpoint), 'log': log}, indent=1))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
