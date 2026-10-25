"""GT of EVERY Harmony4D test frame (not only the evaluator's 1-in-10 scored frames), for
`python -m multismpl.eval.harmony4d --gt-all <dir>`.

Per sequence of an inference run: <out>/<seq>.npz with gt_v (F,2,6890,3) SMPL-topology GT in that run's camera-0
coordinates (zero-padded) and gt_n (F,), row = the inference row. Decoded straight from processed_data/smpl/<frame>.npy
the way Harmony4DDataset + multismpl.inference.harmony4d.gt_vertices do (neutral SMPL, camera 0 = views[0]), without
reading any image; --check compares the result with the GT stored for the scored frames.

  python -m multismpl.eval.harmony4d_gt --infer-dir <infer_dir> --out <dir> [--check]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from multismpl.body.body_model import init_smplx
from multismpl.body.smplx_body import _decode_smpl_batch
from multismpl.inference.harmony4d import load_dataset
from multismpl.paths import PATHS


def sequence_gt(ds, rec, cam0, device):
    """(F,2,6890,3) camera-0 GT and (F,) person counts of every frame of one sequence."""
    frames = sorted(rec.frames, key=ds._numeric_frame_id)
    E = cam0.extrinsics.astype(np.float64)
    R0, t0 = E[:, :3], E[:, 3]
    gv_all = np.zeros((len(frames), 2, 6890, 3), np.float32)
    gn_all = np.zeros(len(frames), np.int64)
    for row, frame in enumerate(frames):
        smpl = np.load(ds._annotation_path(rec.path, frame), allow_pickle=True).item()
        keys = [k for k in ds.PEOPLE if k in smpl]
        pose = np.stack([np.concatenate([np.asarray(smpl[k]['global_orient'], np.float32).reshape(3),
                                         np.asarray(smpl[k]['body_pose'], np.float32).reshape(-1)]) for k in keys])
        beta = np.stack([np.asarray(smpl[k]['betas'], np.float32).reshape(-1)[:10] for k in keys])
        trans = np.stack([np.asarray(smpl[k]['transl'], np.float32).reshape(3) for k in keys])
        with torch.inference_mode():
            _, gv = _decode_smpl_batch(torch.tensor(pose), torch.tensor(beta), torch.tensor(trans),
                                       ['neutral'] * len(keys))
        gv = gv.cpu().numpy() @ R0.T + t0
        n = min(len(keys), 2)
        gv_all[row, :n] = gv[:n]
        gn_all[row] = n
    return gv_all, gn_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--infer-dir', type=Path, required=True, help='inference run (its views define camera 0)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, default=None, help='default: configs/paths.yaml harmony4d_test')
    ap.add_argument('--check', action='store_true', help='compare with the GT stored for the scored frames')
    opt = ap.parse_args()
    opt.out.mkdir(parents=True, exist_ok=True)
    init_smplx()
    ds, _ = load_dataset(opt.dataset_root or PATHS.harmony4d_test, stride=10)
    by_name = {r.name.replace('/', '_'): r for r in ds.sequences}
    for f in sorted(opt.infer_dir.glob('*.npz')):
        g = np.load(f)
        rec = by_name[f.stem]
        out = opt.out / f.name
        if not out.is_file():
            gv, gn = sequence_gt(ds, rec, rec.cameras[int(g['views'][0])], torch.device('cpu'))
            np.savez(out, gt_v=gv, gt_n=gn)
        msg = ''
        if opt.check:
            d = np.load(out)
            rows = g['scored_idx']
            err = np.abs(d['gt_v'][rows] - g['gt_v']).max()
            same_n = np.array_equal(d['gt_n'][rows], g['gt_n'])
            msg = f'  check vs stored scored GT: max |dv| {err:.2e} m, counts equal {same_n}'
        print(f'{f.stem}: {len(np.load(out)["gt_n"])} frames{msg}', flush=True)


if __name__ == '__main__':
    main()
