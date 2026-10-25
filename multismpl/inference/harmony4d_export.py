"""Export Harmony4D test sequences for another multi-view method (e.g. the MAMMA pipeline): per sequence, the chosen
4 cameras as undistorted pinhole image folders + an OpenCV-JSON calibration. The images are exactly the
Harmony4DDataset undistortion our network starts from (fisheye -> pinhole, longest side 1280, before the 518 crop),
so both methods see the same cameras and the same pixels (the other method gets the full 16:9 frame).

World frame of the export: Harmony4D's metric world rotated to z-up (up = mean camera "up" direction, no GT used) and
centred on the camera rig, because MAMMA's own captures are z-up. world_harmony4d = R_up^T @ X + c; both are stored
in <out>/<seq>/export.json so predictions can be brought back to camera-0 coordinates for scoring.

  python -m multismpl.inference.harmony4d_export --views-json views.json --out <dir> [--sequences ...] [--workers 2]

Layout: <out>/<seq>/images/<cam>/<row:05d>.jpg (row = index in the sorted frame list = the scored row index of
multismpl.inference.harmony4d), <out>/<seq>/calib.json, <out>/<seq>/export.json.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from multismpl.inference.harmony4d import load_dataset
from multismpl.paths import PATHS

_DS = None


def _dataset(root):
    global _DS
    if _DS is None:
        cv2.setNumThreads(10)
        _DS, _ = load_dataset(root, stride=10)
    return _DS


def up_frame(E):
    """Rotation R_up (world -> z-up world) and rig centre c from world -> camera extrinsics E (C,3,4)."""
    R = E[:, :, :3]
    up = -np.mean([r.T @ np.array([0., 1., 0.]) for r in R], 0)                   # image y points down
    z = up / np.linalg.norm(up)
    fwd = np.mean([r.T @ np.array([0., 0., 1.]) for r in R], 0)
    x = np.cross(fwd, z) if np.linalg.norm(np.cross(fwd, z)) > 1e-3 else np.cross([1., 0., 0.], z)
    x /= np.linalg.norm(x)
    R_up = np.stack([x, np.cross(z, x), z])                                         # rows = new axes in world
    c = np.mean([-r.T @ e[:, 3] for r, e in zip(R, E)], 0)
    return R_up, c


def export_sequence(root, name, cams, out):
    ds = _dataset(root)
    seq_i = [r.name.replace('/', '_') for r in ds.sequences].index(name)
    rec = ds.sequences[seq_i]
    by_name = {c.name: c for c in rec.cameras}
    frames = sorted(rec.frames, key=ds._numeric_frame_id)
    d = out / name
    E = np.stack([by_name[c].extrinsics for c in cams]).astype(np.float64)
    R_up, c = up_frame(E)
    calib, size = {}, None
    for cam in cams:
        camera = by_name[cam]
        (d / 'images' / cam).mkdir(parents=True, exist_ok=True)
        K = None
        for row, frame in enumerate(frames):
            path = d / 'images' / cam / f'{row:05d}.jpg'
            if path.is_file() and K is not None:
                continue
            img, K = ds._undistort(rec.name, camera, cv2.imread(str(camera.image_dir / f'{frame}.jpg')))
            size = [int(img.shape[1]), int(img.shape[0])]
            if not path.is_file():
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        e = camera.extrinsics.astype(np.float64)                                    # world -> cam
        e_new = np.concatenate([e[:, :3] @ R_up.T, (e[:, 3] + e[:, :3] @ c)[:, None]], 1)   # z-up world -> cam
        calib[cam] = {'intrinsic_matrix': np.asarray(K).tolist(), 'distortions': [0.] * 5,
                      'extrinsics_matrix': e_new.tolist(), 'image_size': size}
    (d / 'calib.json').write_text(json.dumps(calib, indent=1))
    (d / 'export.json').write_text(json.dumps({
        'sequence': rec.name, 'cams': list(cams), 'frames': frames, 'R_up': R_up.tolist(), 'c': c.tolist(),
        'note': 'world_harmony4d = R_up^T @ X_export + c; cam0 = cams[0]'}, indent=1))
    return name, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--views-json', type=Path, required=True, help='{sequence: [4 camera names]}')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, default=None, help='default: configs/paths.yaml harmony4d_test')
    ap.add_argument('--sequences', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=2)
    opt = ap.parse_args()
    views = json.loads(opt.views_json.read_text())
    names = [n for n in views if not opt.sequences or n in opt.sequences]
    root = (opt.dataset_root or PATHS.harmony4d_test).resolve()
    with ProcessPoolExecutor(opt.workers) as ex:
        for name, n in ex.map(export_sequence, [root] * len(names), names, [views[n] for n in names],
                              [opt.out] * len(names)):
            print(f'{name}: {n} frames', flush=True)


if __name__ == '__main__':
    main()
