"""Pick 4 cameras per Harmony4D test sequence that keep BOTH people inside the network's 518 x 518 crop.

USES THE GROUND TRUTH (the GT SMPL joints) to choose the views, so scores obtained with these views are an analysis
setting, not comparable with the evaluator's seeded random draw (multismpl.inference.harmony4d's default).

Per camera: over all frames, the mean fraction of the 24 GT joints that project inside the crop (with a margin, the
head / hands extend past their joints) for the less visible of the two people (1 = both people always entirely in
view; a soft score, so sequences where people often step out of every crop still get the best partial views).
The 4-camera set maximises the sum
of these fractions subject to every pair of cameras being >= --min-sep degrees apart in azimuth around the rig (a wide
baseline for the triangulation); the separation is relaxed in 10 degree steps when no set satisfies it. The camera
seen best comes first (= camera 0, the coordinate frame).

  python -m multismpl.inference.harmony4d_views --out views.json
  python -m multismpl.inference.harmony4d --views-json views.json --out <dir>
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from multismpl.inference.harmony4d import V, load_dataset
from multismpl.paths import PATHS

SIZE = 518


def camera_geometry(ds, seq_i, cams):
    """K (C,3,3) in the 518 crop and world -> camera E (C,3,4) of the given cameras (static rig: one frame)."""
    frame = sorted(ds.sequences[seq_i].frames, key=ds._numeric_frame_id)[0]
    sample = [i for i, (s, f) in enumerate(ds.samples) if s == seq_i and f == frame][0]
    b = ds._get_single_frame_data(seq_index=sample, ids=list(cams), img_per_seq=len(cams))
    return (np.asarray(b['intrinsics'], np.float64).reshape(-1, 3, 3),
            np.asarray(b['extrinsics'], np.float64).reshape(-1, 3, 4))


def gt_joints(ds, seq_i):
    """(F,P,24,3) world GT joints of every frame."""
    rec = ds.sequences[seq_i]
    out = []
    for f in sorted(rec.frames, key=ds._numeric_frame_id):
        smpl = np.load(ds._annotation_path(rec.path, f), allow_pickle=True).item()
        out.append([np.asarray(smpl[k]['joints'], np.float64)[:24] for k in ds.PEOPLE])
    return np.asarray(out)


def visibility(J, K, E, margin):
    """(C,F) fraction of joints inside the crop (margin px, in front of the camera) of the LESS visible person;
    1 = both people entirely in view."""
    Xc = np.einsum('cij,fpkj->cfpki', E[:, :, :3], J) + E[:, None, None, None, :, 3]
    uv = np.einsum('cij,cfpkj->cfpki', K, Xc)
    uv = uv[..., :2] / np.maximum(uv[..., 2:3], 1e-6)
    inside = (Xc[..., 2] > 0.1) & (uv >= margin).all(-1) & (uv < SIZE - margin).all(-1)
    return inside.mean(-1).min(-1)


def azimuths(E):
    """Camera azimuth (deg) around the rig centre in the plane of best fit of the camera centres."""
    C = np.stack([-e[:, :3].T @ e[:, 3] for e in E])
    P = C - C.mean(0)
    _, _, Vt = np.linalg.svd(P)
    return np.degrees(np.arctan2(P @ Vt[1], P @ Vt[0])) % 360


def choose(vis_frac, az, min_sep):
    """Best 4-set (indices into the camera list) under the separation rule, relaxing it when infeasible."""
    order = np.argsort(-vis_frac)
    for sep in range(int(min_sep), -1, -10):
        best, best_score = None, -1.
        for combo in combinations(range(len(az)), V):
            d = [abs((az[i] - az[j] + 180) % 360 - 180) for i, j in combinations(combo, 2)]
            if min(d) < sep:
                continue
            score = float(vis_frac[list(combo)].sum())
            if score > best_score:
                best, best_score = combo, score
        if best is not None:
            return sorted(best, key=lambda i: list(order).index(i)), sep
    raise RuntimeError('no camera set')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='json {sequence: [4 camera names]} (+ a .report.json)')
    ap.add_argument('--dataset-root', type=Path, default=None, help='default: configs/paths.yaml harmony4d_test')
    ap.add_argument('--margin', type=float, default=15., help='px inside the 518 crop border')
    ap.add_argument('--min-sep', type=float, default=40., help='min azimuth between any two chosen cameras (deg)')
    opt = ap.parse_args()
    ds, _ = load_dataset(opt.dataset_root or PATHS.harmony4d_test, stride=10)
    views, report = {}, {}
    for seq_i, rec in enumerate(ds.sequences):
        name = rec.name.replace('/', '_')
        frames = rec.frames
        avail = [c for c, cam in enumerate(rec.cameras) if all(ds._view_available(rec, cam, f) for f in frames)]
        K, E = camera_geometry(ds, seq_i, avail)
        vis = visibility(gt_joints(ds, seq_i), K, E, opt.margin)               # (C,F)
        frac = vis.mean(1)
        pick, sep = choose(frac, azimuths(E), opt.min_sep)
        views[name] = [rec.cameras[avail[i]].name for i in pick]
        both_all4 = float((vis[pick] == 1).all(0).mean())
        report[name] = {'views': views[name], 'visible_frac': [round(float(frac[i]), 3) for i in pick],
                        'frames_both_in_all_4': round(both_all4, 3), 'min_sep_deg': sep,
                        'best_single_cameras': {rec.cameras[avail[i]].name: round(float(frac[i]), 3)
                                                for i in np.argsort(-frac)[:6]}}
        print(f'{name:32s} {" ".join(views[name])}  vis {report[name]["visible_frac"]}  '
              f'both-in-all-4 {both_all4:.2f}  sep>={sep}', flush=True)
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps(views, indent=1))
    opt.out.with_suffix('.report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
