#!/bin/bash
#BSUB -J final_cmp_a8
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 120:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/final_cmp_a8.%J.out
#BSUB -e logs/final_cmp_a8.%J.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ACCEL=8
GRID_PRESET="final_comparison"
SPLIT_JSON=${FINAL_SPLIT_JSON:-final_split.json}
VAL_SLICES=${VAL_SLICES:-2}
TEST_SLICES=${TEST_SLICES:-4}
SLICE_OFFSET=${SLICE_OFFSET:-0}
RUN_ID=${PDAPS_RUN_ID:-fixed_split_seed123}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_${GRID_PRESET}_accel${ACCEL}_${RUN_ID}}
SEED=${SEED:-123}

mkdir -p "$OUT_ROOT"

if [[ ! -f "$SPLIT_JSON" ]]; then
    ./.venv/bin/python3 make_final_split.py --out "$SPLIT_JSON"
fi

mapfile -t VAL_FILENAMES < <(./.venv/bin/python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["val"]))' "$SPLIT_JSON")
mapfile -t TEST_FILENAMES < <(./.venv/bin/python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["test"]))' "$SPLIT_JSON")

echo "Final comparison: accel=${ACCEL}, val=${#VAL_FILENAMES[@]}, test=${#TEST_FILENAMES[@]}, out=${OUT_ROOT}"
echo "Run started at $(date -Is)" | tee -a "$OUT_ROOT/run.log"

./.venv/bin/python3 mri_validation.py \
    --grid-preset "$GRID_PRESET" \
    --filenames "${VAL_FILENAMES[@]}" \
    --test-filenames "${TEST_FILENAMES[@]}" \
    --val-slices "$VAL_SLICES" \
    --test-slices "$TEST_SLICES" \
    --slice-offset "$SLICE_OFFSET" \
    --seed "$SEED" \
    --acceleration "$ACCEL" \
    --pattern random \
    --mask-seed 0 \
    --out-dir "$OUT_ROOT" \
    --resume \
    --log-level VAL \
    2>&1 | tee -a "$OUT_ROOT/run.log"
