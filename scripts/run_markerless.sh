#!/usr/bin/env bash
# MAMMA markerless multi-person scene (no GT): inference with mask candidates -> v9 tracking + TTO.
#   OUT=<dir> [DEVICE=cuda|cpu] scripts/run_markerless.sh <seq_dir> <people>
# Markerless worlds are z-up, so the default --up-axis z is correct here.
set -euo pipefail
SEQ=$(cd "${1:?seq_dir}" && pwd); NP=${2:?people}
H=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
OUT=${OUT:?set OUT=<output dir>}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: we cd to the repo
DEVICE=${DEVICE:-cuda}
C=$(( NP + 4 > 6 ? NP + 4 : 6 ))
cd "$H"
"$PY" -W ignore -m multismpl.inference.mamma --seq-dirs "$SEQ" --people $NP --meta-dir meta --video-dir videos_crf24 --candidates $C \
    --device $DEVICE --out "$OUT/infer"
"$PY" -W ignore -m multismpl.tto.run --infer "$OUT/infer" --out "$OUT/fit" --sequences "$(basename "$SEQ")" --device $DEVICE \
    --tracker candidates --people $NP
