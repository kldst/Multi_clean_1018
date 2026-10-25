#!/usr/bin/env bash
# One benchmark end to end: inference -> network score -> tracking + TTO -> TTO score.
#   OUT=<dir> [CKPT=<checkpoint>] [DEVICE=cuda|cpu] [LEGACY=1] [PY=python] scripts/run_benchmark.sh dance|singles|chi3d|harmony4d
# Every dataset (default): top-k slots per frame (the network score; singles: causal top-1, --select track) ->
#   Joint Candidate Tracking over 4 mask candidates (multismpl.tto.run defaults: clear-outlier triangulation, no
#   silhouette-centre shift, consistent candidates kept among duplicates, top-k slots never merged, near-zero-presence
#   candidates not boosted by their masks, lost tracks re-acquire clear candidates, per-person smoothing) -> TTO.
# LEGACY=1: the per-dataset trackers of the older numbers (dance / chi3d / singles: slot tracker;
#   harmony4d: Joint Candidate Tracking with --up-axis z).
# GPU choice is the caller's (CUDA_VISIBLE_DEVICES). CPU: DEVICE=cpu, keep OMP_NUM_THREADS small.
set -euo pipefail
DS=${1:?dataset: dance|singles|chi3d|harmony4d}
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
CK=${CKPT:+--checkpoint $CKPT}               # default: configs/paths.yaml
OUT=${OUT:?set OUT=<output dir>}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: we cd to the repo
DEVICE=${DEVICE:-cuda}
LEGACY=${LEGACY:-0}
cd "$H"
run() { echo "$(date '+%F %T') | $*"; "$PY" -W ignore -m "$@" 2>&1 | { grep -vE "Warning|warn\(" || true; }; }

if [ "$LEGACY" = 1 ]; then
    CAND="--candidates 0"; TTO="--tracker slots"
    [ "$DS" = harmony4d ] && CAND="" && TTO="--tracker candidates --triangulation outlier --up-axis z --up-shift 0.15 --no-prefer-consistent --no-keep-topk-pairs --min-presence-boost 0 --no-reacquire --no-per-person-smooth"
else
    CAND="--candidates 4"; TTO="--tracker candidates --triangulation outlier --up-axis body"
fi
case $DS in
  dance)
    run multismpl.inference.mamma $CK --device $DEVICE $CAND --out "$OUT/infer"
    run multismpl.eval.mamma --dataset dance --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE $TTO
    run multismpl.eval.mamma --dataset dance --kind fit --pred-dir "$OUT/fit" --out "$OUT/eval_fit.json" ;;
  singles)
    run multismpl.inference.mamma $CK --device $DEVICE --dataset singles \
        --people 1 --select track $CAND --out "$OUT/infer"
    run multismpl.eval.mamma --dataset singles --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE $TTO
    run multismpl.eval.mamma --dataset singles --kind fit --pred-dir "$OUT/fit" --out "$OUT/eval_fit.json" ;;
  chi3d)
    run multismpl.inference.chi3d $CK --device $DEVICE $CAND --out "$OUT/infer"
    run multismpl.eval.chi3d --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE $TTO
    run multismpl.eval.chi3d --kind fit --pred-dir "$OUT/fit" --infer-dir "$OUT/infer" --out "$OUT/eval_fit.json"
    run multismpl.eval.chi3d --kind fit --pred-dir "$OUT/fit" --infer-dir "$OUT/infer" --gate-mm inf --out "$OUT/eval_fit_nogate.json" ;;
  harmony4d)
    run multismpl.inference.harmony4d $CK --device $DEVICE $CAND --out "$OUT/infer"
    run multismpl.eval.harmony4d --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE $TTO
    run multismpl.eval.harmony4d --kind fit --pred-dir "$OUT/fit" --infer-dir "$OUT/infer" --out "$OUT/eval_fit.json" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
echo "$(date '+%F %T') | $DS done -> $OUT"
