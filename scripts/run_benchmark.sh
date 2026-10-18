#!/usr/bin/env bash
# One benchmark end to end: inference -> network score -> tracking + TTO -> TTO score.
#   OUT=<dir> [CKPT=<checkpoint>] [DEVICE=cuda|cpu] [UNIFIED=1] [PY=python] scripts/run_benchmark.sh dance|singles|chi3d|harmony4d
# Default = the per-dataset method behind the reported numbers (docs/RESULTS.md):
#   dance / chi3d : slot tracker (keep/swap Viterbi)          singles : causal slot selection, P = 1
#   harmony4d     : v9 candidate tracker, clear-outlier triangulation, z-shift
# UNIFIED=1: the same v9 method everywhere (inference stores 4 mask candidates; --tracker candidates
#   --triangulation outlier --up-axis body). Not benchmarked yet.
# GPU choice is the caller's (CUDA_VISIBLE_DEVICES). CPU: DEVICE=cpu, keep OMP_NUM_THREADS small.
set -euo pipefail
DS=${1:?dataset: dance|singles|chi3d|harmony4d}
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
CK=${CKPT:+--checkpoint $CKPT}               # default: configs/paths.yaml
OUT=${OUT:?set OUT=<output dir>}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: we cd to the repo
DEVICE=${DEVICE:-cuda}
UNIFIED=${UNIFIED:-0}
cd "$H"
run() { echo "$(date '+%F %T') | $*"; "$PY" -W ignore -m "$@" 2>&1 | { grep -vE "Warning|warn\(" || true; }; }

if [ "$UNIFIED" = 1 ]; then
    CAND="--candidates 4"; TTO="--tracker candidates --triangulation outlier --up-axis body"
else
    CAND=""; TTO=""
fi
case $DS in
  dance)
    run multismpl.inference.mamma $CK --device $DEVICE $CAND --out "$OUT/infer"
    run multismpl.eval.mamma --dataset dance --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE $TTO
    run multismpl.eval.mamma --dataset dance --kind fit --pred-dir "$OUT/fit" --out "$OUT/eval_fit.json" ;;
  singles)
    SEL=$([ "$UNIFIED" = 1 ] && echo "$CAND" || echo "--select track")
    run multismpl.inference.mamma $CK --device $DEVICE --dataset singles \
        --people 1 $SEL --out "$OUT/infer"
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
    UP=$([ "$UNIFIED" = 1 ] && echo body || echo z)
    run multismpl.inference.harmony4d $CK --device $DEVICE --out "$OUT/infer"
    run multismpl.eval.harmony4d --kind infer --pred-dir "$OUT/infer" --out "$OUT/eval_network.json"
    run multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --device $DEVICE --tracker candidates --triangulation outlier --up-axis $UP
    run multismpl.eval.harmony4d --kind fit --pred-dir "$OUT/fit" --infer-dir "$OUT/infer" --out "$OUT/eval_fit.json" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
echo "$(date '+%F %T') | $DS done -> $OUT"
