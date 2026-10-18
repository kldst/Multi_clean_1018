# Results

All numbers in mm (lower is better). Within one table every method is scored by the same evaluator on the same frames
and cameras; protocols differ between benchmarks, so do not compare across tables.

- **SMPL-22**: SMPL J_regressor on SMPL-topology vertices (SMPL-X → SMPL mapping), body joints 0–21, rooted on joint 0.
- **H36M-14**: J_regressor_h36m → 14 joints. Rooted on the regressed H36M pelvis (Dance, singles, Harmony4D) or on the
  SMPL-X kinematic joint 0 (CHI3D, as MV4DHMR `test.py`).
- **PA-**: after a similarity (Umeyama) alignment. **Transl**: pelvis position error.
- **Ours**: checkpoint `nano4` (trained on MAMMA synthetic, Harmony4D, BEDLAM; never on CHI3D), 4 views, T = 3.
  **+ TTO** uses the calibrated cameras, as do all multi-view methods below.

## MAMMA Dance — 18 sequences, 12 462 person-frames, 4 views (IOI_01/05/09/13)

Hungarian matching on the world H36M pelvis, no gate, every GT person scored.

| Method | SMPL-22 | SMPL-22 PA | H36M-14 | H36M-14 PA |
|---|---|---|---|---|
| Ours, network | 54.94 | 39.28 | 61.02 | 41.35 |
| **Ours + TTO** | 41.72 | **30.60** | 47.91 | 32.67 |
| MAMMA official pipeline (YOLO+SAM2 → MammaNet → multi-view SMPL-X fit), same 4 views | **37.99** | 30.72 | **42.77** | **31.49** |
| HeatFormer (GT boxes / person count / association) | 67.96 | 42.15 | 73.13 | 41.58 |
| MV4DHMR, 4 views | 75.00 | 47.01 | 81.35 | 51.49 |
| MV4DHMR, 8 views | 51.87 | 34.38 | 61.08 | 37.86 |

## CHI3D s04 — 127 sequences, 13 034 person-frames, MV4DHMR `test.py` protocol

4 cameras, every 3rd frame, camera-0 coordinates, greedy PVE matching.

**With the 500 mm gate (test.py default)**

| Method | MPJPE | PA-MPJPE | PVE | PA-PVE | Transl | SMPL-22 | SMPL-22 PA | recall |
|---|---|---|---|---|---|---|---|---|
| Ours, network | 57.82 | 34.13 | 65.74 | 44.70 | 234.39 | 53.34 | 31.08 | 99.6% |
| **Ours + TTO** | 44.37 | 29.31 | 52.87 | 41.77 | 27.95 | 40.20 | 25.61 | 99.8% |
| HeatFormer, GT boxes | 51.79 | 30.33 | 63.18 | 42.50 | 42.72 | – | – | 99.6% |
| HeatFormer, YOLO boxes | 48.36 | 29.03 | 58.76 | 40.58 | 35.78 | – | – | 99.8% |
| MAMMA official pipeline (all 50 fps frames) | **32.97** | **24.16** | **42.11** | **34.55** | **24.51** | – | – | 99.6% |
| MV4DHMR (paper, not fine-tuned) | 62.0 | 44.2 | 53.0 | 38.3 | 34.3 | – | – | – |
| Multi-HMR (paper) | 85.3 | 56.7 | 69.7 | 44.3 | 194.4 | – | – | – |

**Without the gate (every GT person scored)**

| Method | MPJPE | PA-MPJPE | PVE | PA-PVE | Transl | SMPL-22 | SMPL-22 PA |
|---|---|---|---|---|---|---|---|
| Ours, network | 58.11 | 34.33 | 66.10 | 45.01 | 242.20 | 53.57 | 31.22 |
| **Ours + TTO** | **44.67** | 29.45 | **53.21** | 41.97 | **30.34** | 40.46 | 25.71 |
| HeatFormer, GT boxes | 52.88 | 30.52 | 64.46 | 42.74 | 43.70 | – | – |
| HeatFormer, YOLO boxes | 49.28 | **29.09** | 59.84 | **40.66** | 35.90 | – | – |

## MAMMA singles — 22 sequences, 22 188 frames, one person

Gendered GT, no gate. The network's slot is chosen by causal tracking (`--select track`).

| Method | SMPL-22 | SMPL-22 PA | H36M-14 | H36M-14 PA |
|---|---|---|---|---|
| Ours, network | 60.91 | 42.80 | 66.51 | 44.03 |
| **Ours + TTO** | **49.00** | 36.38 | **54.33** | 37.86 |
| HeatFormer (GT boxes / person count / association) | 53.78 | **34.84** | 56.34 | **34.96** |
| MV4DHMR, 4 views | 122.34 | 59.15 | 139.35 | 65.28 |

## Harmony4D test — 1 429 scored frames, two people in close contact

`eval_h4d_nano.py` protocol: last frame of every 10th clip, camera-0 coordinates, Hungarian on PVE, one fixed draw of
4 cameras per sequence. TTO: candidate tracker with mask-centroid triangulation (clear-outlier rule). No other method
has been run on this split yet.

| Method | SMPL-22 | SMPL-22 PA | H36M-14 | H36M-14 PA | PVE | Transl | H36M-14, 500 mm gate | recall (gated) |
|---|---|---|---|---|---|---|---|---|
| Ours, network | 59.59 | 42.44 | 66.35 | 44.79 | 77.30 | 137.83 | 64.13 | 98.0% |
| **Ours + TTO** | **51.87** | **37.08** | **59.56** | **39.75** | **69.23** | **72.90** | **54.74** | 98.4% |

## Reproducing

```bash
OUT=outputs/<benchmark> scripts/run_benchmark.sh dance|singles|chi3d|harmony4d
```

The numbers above were produced by the research code this repository was extracted from; the extracted code was
checked to give bit-identical inference outputs, fits and scores on samples of every benchmark.
