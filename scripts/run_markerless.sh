#!/usr/bin/env bash
# MAMMA markerless multi-person scenes (no GT; configs/paths.yaml mamma_markerless): inference with mask candidates ->
# candidate tracking + TTO. The number of people is read from each sequence name (260216_MultiMama_<N>_...).
#   OUT=<dir> [DEVICE=cuda|cpu] [PY=python] scripts/run_markerless.sh [sequence ...]     # no sequence = all
# Markerless worlds are z-up, so the default --up-axis z is correct here.
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
OUT=${OUT:?set OUT=<output dir>}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: we cd to the repo
DEVICE=${DEVICE:-cuda}
SEQ=${*:+--sequences $*}
cd "$H"
"$PY" -W ignore -m multismpl.inference.mamma --dataset markerless --people auto --candidates auto $SEQ \
    --device $DEVICE --out "$OUT/infer"
"$PY" -W ignore -m multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" $SEQ --device $DEVICE --tracker candidates
