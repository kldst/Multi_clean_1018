# MultiSMPL: multi-view, multi-person SMPL-X recovery

A feed-forward network (a [VGGT](https://github.com/facebookresearch/vggt) derivative) predicts every person's SMPL-X and
a per-person silhouette mask from a few calibrated views; a mask-based **test-time optimisation (TTO)** then refines the
bodies against those masks. No 2D keypoint detector, no differentiable renderer and no ground-truth poses are used.

```
videos (V views) ──► network: per frame, P person slots ──► tracking ──► TTO ──► SMPL-X per person, world frame
                     (presence, SMPL-X, pelvis, masks)      (who is who)   (fit the bodies to the masks)
```

## Results (MPJPE / PA-MPJPE, mm)

| Benchmark | Metric | Network | + TTO |
|---|---|---|---|
| MAMMA Dance (18 seq., 2 people, 4 views) | SMPL-22 | 54.94 / 39.28 | **41.72 / 30.60** |
| CHI3D s04 (MV4DHMR `test.py` protocol) | H36M-14 | 57.82 / 34.13 | **44.37 / 29.31** |
| MAMMA singles (22 seq.) | SMPL-22 | 60.91 / 42.80 | **49.00 / 36.38** |
| Harmony4D test (close interaction) | H36M-14 | 66.35 / 44.79 | **59.56 / 39.75** |

The model was not trained on CHI3D. Comparisons with other methods under the same protocols, and how every number is
computed, are in [docs/RESULTS.md](docs/RESULTS.md).

## Install

```bash
pip install -e .            # torch, numpy, scipy, opencv-python, pyyaml, smplx, Pillow
pip install -e .[dev]       # + pytest
```

Body models are not redistributable; download them yourself:
- SMPL-X (neutral / female / male) from https://smpl-x.is.tue.mpg.de
- SMPL from https://smpl.is.tue.mpg.de: `basicModel_{f,m,neutral}_lbs_10_207_0_v1.0.0.pkl` (Harmony4D ground truth is
  SMPL) and `SMPL_NEUTRAL.pkl` (its J_regressor defines the SMPL-22 metric)
- `smplx2smpl.pkl` (SMPL-X → SMPL vertex mapping) from the SMPL-X model transfer tools
- `J_regressor_h36m.npy` (H36M joint regressor, as used by SPIN / MV4DHMR)

Then copy `configs/paths.example.yaml` to `configs/paths.yaml` and fill in the checkpoint, body-model and dataset paths
(or point `MULTISMPL_PATHS` at another file).

## Run

```bash
# one benchmark end to end: inference -> network score -> TTO -> TTO score
OUT=outputs/dance DEVICE=cuda scripts/run_benchmark.sh dance        # dance | singles | chi3d | harmony4d

# or step by step
python -m multismpl.inference.mamma --out outputs/dance/infer                     # 1. network (+ masks)
python -m multismpl.tto.run --infer outputs/dance/infer --out outputs/dance/fit  # 2. tracking + TTO
python -m multismpl.eval.mamma --dataset dance --kind fit --pred-dir outputs/dance/fit --out outputs/dance/eval.json

# MAMMA markerless multi-person scenes (no GT; people count read from the sequence name)
OUT=outputs/markerless scripts/run_markerless.sh                                     # all sequences
OUT=outputs/markerless scripts/run_markerless.sh 260216_MultiMama_3_assist_walk_101100_1
```

Every command takes `--help`. Each step skips sequences whose output already exists, so runs can be resumed.

## Repository layout

```
multismpl/
  paths.py            all file locations, read from configs/paths.yaml
  model/              the network: VGGT aggregator + temporal SMPL head + DPT person-mask head (+ camera head)
  body/               SMPL-X decoding and coordinate conversions
  data/               image preprocessing and the Harmony4D loader
  inference/          step 1, one module per dataset (mamma, chi3d, harmony4d)
  tto/                step 2: silhouette losses, triangulation, tracking, the fit, CLI (run.py)
  eval/               step 3: one evaluator per benchmark; protocols/ holds the official protocol code
configs/              model_nano4.yaml (architecture of the released checkpoint), paths.example.yaml
scripts/              run_benchmark.sh, run_markerless.sh
tests/                CPU tests on synthetic data (pytest)
docs/                 RESULTS.md
```

## Tests

```bash
pytest            # CPU only, synthetic data; the end-to-end TTO test is skipped if SMPL-X is not configured
```

## Licenses

- Model code under `multismpl/model/` is derived from VGGT (Meta Platforms); see `LICENSE_VGGT.txt`. The released
  weights were initialised from the original VGGT-1B checkpoint, whose licence is **non-commercial**.
- SMPL / SMPL-X models and all datasets (MAMMA, CHI3D, Harmony4D, BEDLAM) are under their own licences and are not
  included.
- License of this repository: TBD.
