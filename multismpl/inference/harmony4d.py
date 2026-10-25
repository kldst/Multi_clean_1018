"""Step 1 for the Harmony4D test split: network inference on EVERY frame (the TTO needs a normal video), storing GT
only for the frames Harmony4D_testing/eval/eval_h4d_nano.py scores (last frame of every 10th temporal clip).

Contract per frame = eval_h4d_nano.py's: Harmony4DDataset single-frame loader (fisheye undistortion), causal T=3
window of the same consecutive run, top-2 presence slots WITHOUT NMS, camera-0 coordinates. Views: one seeded random
draw of 4 cameras per sequence among those present in every frame (identical for every checkpoint).

  python -m multismpl.inference.harmony4d --out <dir>

Per sequence <out>/<seq>.npz (camera 0 = world frame):
  pose (F,2,72) beta (F,2,10) trans (F,2,3) presence slots masks (F,2,4,259,259)
  cand_* (F,C,...) mask-ranked candidates (for multismpl.tto.run --tracker candidates)
  K (4,3,3)  E (4,3,4) cam0 -> cam c  frame_ids (F,)  views (4,)
  scored_idx (S,) rows scored by the evaluator, scored_clips (S,), gt_v (S,2,6890,3) SMPL-topology GT (camera 0,
  zero-padded), gt_n (S,)
"""
import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from multismpl.body.body_model import init_smplx
from multismpl.body.smplx_body import _decode_smpl_batch
from multismpl.data.harmony4d import Harmony4DDataset
from multismpl.eval.protocols.common import topk_nms3d
from multismpl.inference.network import CachedTemporalHead, head_meta, load_model, person_masks, rank_candidates_by_mask, slot_outputs
from multismpl.paths import MODEL_CONFIG, PATHS

GENDER = {0: 'female', 1: 'male', 2: 'neutral'}   # harmony4d.py emits an integer code
V, T = 4, 3


def load_dataset(root, stride):
    common = SimpleNamespace(img_size=518, patch_size=14, augs=SimpleNamespace(scales=None), rescale=True,
                             rescale_aug=False, landscape_check=False, training=False, emit_dense_geometry=False,
                             use_temporal_training=True, temporal_clip_length=T)
    ds = Harmony4DDataset(common_conf=common, split='train', Harmony4D_DIR=str(root.resolve()), min_num_images=V,
                          max_num_people=2, val_sequence_fraction=0.0, body_model_type='smpl')
    ds.debug = False
    scored = defaultdict(dict)                      # seq index -> {last frame of a scored clip: clip index}
    for clip in range(0, len(ds.temporal_samples), stride):
        seq_i, frames, _ = ds.temporal_samples[clip]
        scored[seq_i][frames[-1]] = clip
    return ds, scored


def cameras_in_cam0(b):
    """Per-frame cameras from the loader -> K (V,3,3), E (V,3,4) relative to camera 0, and camera 0's [R0|t0]."""
    ext = np.asarray(b['extrinsics'], np.float64).reshape(V, 3, 4)
    K = np.asarray(b['intrinsics'], np.float64).reshape(V, 3, 3)
    R0, t0 = ext[0, :, :3], ext[0, :, 3]
    E = np.stack([np.concatenate([e[:, :3] @ R0.T, (e[:, 3] - e[:, :3] @ R0.T @ t0)[:, None]], 1) for e in ext])
    return K, E, R0, t0


def gt_vertices(b, R0, t0, smplx2smpl, device):
    """GT people of one frame -> (2,6890,3) SMPL-topology vertices in camera-0 coordinates (zero-padded), count."""
    npeople = int(np.asarray(b['num_people']).reshape(-1)[0])
    gp = np.asarray(b['smpl_pose']).reshape(-1, 72)[:npeople]
    gb = np.asarray(b['smpl_beta']).reshape(-1, 10)[:npeople]
    gt_ = np.asarray(b['smpl_trans']).reshape(-1, 3)[:npeople]
    gg = [GENDER[int(x)] for x in np.asarray(b['smpl_gender']).reshape(-1)[:npeople]]
    _, gv = _decode_smpl_batch(torch.tensor(gp), torch.tensor(gb), torch.tensor(gt_), gg)
    gv = gv.numpy() @ R0.T + t0
    if gv.shape[1] != 6890:
        gv = torch.einsum('sv,pvc->psc', smplx2smpl, torch.tensor(gv, dtype=torch.float32, device=device)).cpu().numpy()
    gpad = np.zeros((2, 6890, 3), np.float32)
    gpad[:min(npeople, 2)] = gv[:2]
    return gpad, min(npeople, 2)


def run_sequence(model, ds, scored, seq_i, record, opt, device, smplx2smpl):
    frames = sorted(record.frames, key=ds._numeric_frame_id)
    if opt.max_frames:
        frames = frames[:opt.max_frames]
    avail = [c for c, cam in enumerate(record.cameras) if all(ds._view_available(record, cam, f) for f in frames)]
    if len(avail) < V:
        return None
    fixed = opt.views_per_seq.get(record.name.replace('/', '_')) if opt.views_per_seq else opt.views
    if fixed:                                                           # fixed camera names, first = camera 0
        names = [c.name for c in record.cameras]
        if not all(n in names and names.index(n) in avail for n in fixed):
            return None
        views = [names.index(n) for n in fixed]
    else:
        views = [int(i) for i in np.random.default_rng(seq_i).permutation(avail)[:V]]
    sample_of = {(s, f): i for i, (s, f) in enumerate(ds.samples)}
    head = CachedTemporalHead(model, head_meta(V, device, T))
    rec, gts = defaultdict(list), defaultdict(list)
    K0 = cam0 = E0 = None
    prev_id = None
    with torch.inference_mode():
        for row, frame in enumerate(frames):
            fid = ds._numeric_frame_id(frame)
            if prev_id is not None and fid - prev_id != 1:
                head.reset()                                           # new consecutive run
            prev_id = fid
            np.random.seed(row)
            random.seed(row)
            b = ds._get_single_frame_data(seq_index=sample_of[(seq_i, frame)], ids=views, img_per_seq=V)
            assert [int(i) for i in b['ids']] == views, (record.name, frame, b['ids'], views)
            imgs = torch.as_tensor(np.asarray(b['images'])).to(device=device, dtype=torch.float32)
            if imgs.shape[-1] == 3:
                imgs = imgs.permute(0, 3, 1, 2)
            if imgs.max() > 1.5:
                imgs = imgs / 255.
            pred, tokens, patch = head(imgs)
            prob, tr, pose, beta = slot_outputs(pred)
            if opt.select == 'nms':                                      # same rule as the MAMMA evaluators
                sel = topk_nms3d(prob, tr, 2)
            else:
                sel = [int(i) for i in np.argsort(-prob)[:2]]           # eval_h4d_nano: top-2, no NMS
            m = person_masks(model, pred, tokens, imgs[None], patch, list(range(len(prob))))
            csel = rank_candidates_by_mask(m, prob, sel, opt.candidates)
            for k, arr in (('pose', pose), ('beta', beta), ('trans', tr), ('presence', prob)):
                rec[k].append(arr[sel])
                rec[f'cand_{k}'].append(arr[csel])
            rec['slots'].append(np.asarray(sel)); rec['masks'].append(m[sel])
            rec['cand_slots'].append(np.asarray(csel)); rec['cand_masks'].append(m[csel])
            rec['all_presence'].append(prob); rec['all_trans'].append(tr)
            rec['frame_ids'].append(fid)

            K, E, R0, t0 = cameras_in_cam0(b)
            if K0 is None:
                K0, E0, cam0 = K, E, np.concatenate([R0, t0[:, None]], 1)
            assert np.allclose(K, K0, atol=1e-3) and np.allclose(np.concatenate([R0, t0[:, None]], 1), cam0, atol=1e-5), \
                f'{record.name}/{frame}: cameras are not static'
            if frame in scored[seq_i]:
                gv, gn = gt_vertices(b, R0, t0, smplx2smpl, device)
                gts['scored_idx'].append(row); gts['scored_clips'].append(scored[seq_i][frame])
                gts['gt_v'].append(gv); gts['gt_n'].append(gn)
            del tokens, pred
    keys = ['pose', 'beta', 'trans', 'presence', 'slots', 'masks'] + \
        [f'cand_{k}' for k in ('pose', 'beta', 'trans', 'presence', 'slots', 'masks')] + ['all_presence', 'all_trans', 'frame_ids']
    np.savez_compressed(opt.out / f'{record.name.replace("/", "_")}.npz', **{k: np.stack(rec[k]) for k in keys},
                        **{k: np.stack(v) for k, v in gts.items()}, K=K0, E=E0, views=np.array(views))
    return {'frames': len(frames), 'scored': len(gts['scored_idx']), 'views': views}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, default=None, help='default: configs/paths.yaml harmony4d_test')
    ap.add_argument('--checkpoint', type=Path, default=None, help='default: configs/paths.yaml checkpoint')
    ap.add_argument('--config', type=Path, default=MODEL_CONFIG, help='model config')
    ap.add_argument('--stride', type=int, default=10, help='evaluator: every stride-th temporal clip is scored')
    ap.add_argument('--candidates', type=int, default=4)
    ap.add_argument('--select', choices=['top2', 'nms'], default='top2',
                    help="top2: eval_h4d_nano's top-2 presence, no NMS (the reported protocol); "
                         "nms: top-2 presence with 3D NMS at 0.25 m, as the MAMMA evaluators")
    ap.add_argument('--sequences', nargs='*', default=None, help='sequence names (record.name with / -> _)')
    ap.add_argument('--views', nargs=V, default=None, metavar='CAM',
                    help='fixed camera names, e.g. cam01 cam02 cam03 cam04 (first = camera 0); '
                         'default: the evaluator\'s seeded random draw')
    ap.add_argument('--views-json', type=Path, default=None,
                    help='per-sequence camera names {sequence: [4 names]} (multismpl.inference.harmony4d_views)')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    opt = ap.parse_args()
    opt.views_per_seq = json.loads(opt.views_json.read_text()) if opt.views_json else None
    opt.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(opt.device)
    torch.set_num_threads(4)
    opt.checkpoint = opt.checkpoint or PATHS.checkpoint
    ds, scored = load_dataset(opt.dataset_root or PATHS.harmony4d_test, opt.stride)
    print(f'Harmony4D test: {len(ds.sequences)} sequences, {sum(map(len, scored.values()))} scored frames', flush=True)
    init_smplx()
    with open(PATHS.smplx2smpl, 'rb') as h:
        smplx2smpl = torch.tensor(np.asarray(pickle.load(h)['matrix'], np.float32), device=device)
    model = load_model(opt.config, opt.checkpoint, device)
    log = {}
    for seq_i, record in enumerate(ds.sequences):
        name = record.name.replace('/', '_')
        if opt.sequences and name not in opt.sequences:
            continue
        if (opt.out / f'{name}.npz').is_file():
            print(f'{name}: exists, skip', flush=True)
            continue
        t0 = time.time()
        info = run_sequence(model, ds, scored, seq_i, record, opt, device, smplx2smpl)
        if info is None:
            print(f'SKIP {name}: ' + ('the fixed views are not all present in every frame'
                                       if opt.views or opt.views_per_seq else
                                       f'fewer than {V} cameras present in every frame'), flush=True)
            continue
        log[name] = info | {'seconds': time.time() - t0}
        print(f'{name}: {info["frames"]} frames ({info["scored"]} scored) {time.time() - t0:.0f}s', flush=True)
    (opt.out / f'run_{int(time.time())}.json').write_text(json.dumps({'checkpoint': str(opt.checkpoint), 'log': log}, indent=1))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
