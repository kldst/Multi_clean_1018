# MultiSMPL: multi-view, multi-person SMPL-X recovery

A feed-forward network (a [VGGT](https://github.com/facebookresearch/vggt) derivative) predicts every person's SMPL-X and
a per-person silhouette mask from a few calibrated views; a mask-based **test-time optimisation (TTO)** then refines the
bodies against those masks. No 2D keypoint detector, no differentiable renderer and no ground-truth poses are used.

```
videos (V views) ─► network ──────────► top-k slots ─► Joint Candidate ─► TTO ────────────► SMPL-X per person
                    per frame: 20        (network       Tracking (JCT)     fit the bodies     (world frame)
                    person slots         score)         who is who         to the masks
                    (presence, SMPL-X,                  over time
                    pelvis, masks)
```

The same three steps run on every dataset:
1. **top-k**: per frame, the k most present slots (3D NMS where the benchmark's evaluator uses it; singles: causal
   top-1). This is the "network" score. The inference also stores 4 mask-ranked candidates per frame (the top-k slots
   first, then other slots whose masks are visible in at least half of the views).
2. **Joint Candidate Tracking**: places every candidate in 3D by triangulating its mask centroids and links candidates
   into P person tracks, all people assigned jointly per frame (Hungarian on motion + mask IoU + confidence). It can
   replace a slot that top-k picked wrongly (a bystander, a duplicate). Candidates closer than 25 cm count as one
   person, except that two top-k slots are never merged (people hugging), and among duplicates the one whose own body
   agrees with its mask triangulation is kept. A slot the network barely believes in (presence < 0.1) is not made
   confident by its masks, and a track that lost its person can take back a clearly visible, unclaimed candidate at
   any distance. Frames where a person is still not found are filled from the neighbouring frames. No ground truth
   is used.
3. **TTO**: fits each track's SMPL-X to its masks in all views over the whole sequence; temporal smoothness is
   switched off only for the person who is missing or touching someone, not for everyone in that frame.

Scores match predictions to the ground-truth people of each frame (the benchmark's protocol), for the network and the
TTO rows alike.

## Results (MPJPE / PA-MPJPE, mm)

| Benchmark | Metric | Network | + TTO |
|---|---|---|---|
| MAMMA Dance (18 seq., 2 people, 4 views) | SMPL-22 | 54.84 / 39.32 | **42.08 / 31.02** |
| CHI3D s04 (MV4DHMR `test.py` protocol) | H36M-14 | 57.82 / 34.13 | **44.44 / 29.40** |
| MAMMA singles (22 seq.) | SMPL-22 | 61.13 / 43.18 | **48.49 / 36.29** |
| Harmony4D test (close interaction) | SMPL-22 | 56.72 / 39.27 | **44.28 / 32.41** |

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
python -m multismpl.tto.run --infer outputs/dance/infer --out outputs/dance/fit  # 2. JCT + TTO
python -m multismpl.eval.mamma --dataset dance --kind fit --pred-dir outputs/dance/fit --out outputs/dance/eval.json

# MAMMA markerless multi-person scenes (no GT; people count read from the sequence name)
OUT=outputs/markerless scripts/run_markerless.sh                                     # all sequences
OUT=outputs/markerless scripts/run_markerless.sh 260216_MultiMama_3_assist_walk_101100_1
```

Every command takes `--help`. Each step skips sequences whose output already exists, so runs can be resumed.
`LEGACY=1 scripts/run_benchmark.sh ...` (or `multismpl.tto.run --tracker slots`) runs the older per-dataset trackers.

## Train

Trains on MAMMA (synthetic) + Harmony4D in the compose format (multi-view 518×518 frames + per-person SMPL-X,
cameras and masks). Edit the "local paths" block of `configs/train/train_mamma_harmony4d.yaml` first.

```bash
pip install -e .[train]                                        # hydra-core, fvcore, tensorboard, wandb
python -m multismpl.train.launch --config train_mamma_harmony4d
torchrun --nproc_per_node 4 -m multismpl.train.launch --config train_mamma_harmony4d     # multi-GPU
```

By default the aggregator and camera head start from the pretrained VGGT-1B checkpoint and stay frozen; the temporal
SMPL head and the person-mask head are trained. Set `init_checkpoint` to a trained model to fine-tune it instead.
Logs and checkpoints go to `logs/<exp_name>/`.

## Repository layout

```
multismpl/
  paths.py            all file locations, read from configs/paths.yaml
  model/              the network: VGGT aggregator + temporal SMPL head + DPT person-mask head (+ camera head)
  body/               SMPL-X decoding and coordinate conversions
  data/               image preprocessing and the Harmony4D loader
  inference/          step 1, one module per dataset (mamma, chi3d, harmony4d)
  tto/                step 2: silhouette losses, triangulation, tracking (JCT), the fit, CLI (run.py)
  eval/               step 3: one evaluator per benchmark; protocols/ holds the official protocol code
  train/              training: launch.py (entry), trainer.py, losses, utils/ (optimizer, checkpoints, logging)
configs/              model_nano4.yaml (architecture of the released checkpoint), paths.example.yaml,
                      train/train_mamma_harmony4d.yaml
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
