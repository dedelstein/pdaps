#!/bin/bash
#BSUB -J pdaps_prelaunch_c0_a4
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_prelaunch_c0_a4.%J.out
#BSUB -e logs/pdaps_prelaunch_c0_a4.%J.err

set -euo pipefail

ACCEL=4
GRID_PRESET="pdaps_prelaunch_c0"
METHOD_INDICES="0-6"
VAL_SLICES=2
SEEDS=(123)
FILENAMES=(file1000196.h5 file1001458.h5 file1002451.h5)
SLICE_OFFSET=0

ROOT_ID=${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT="results/mri_validation_${GRID_PRESET}_accel${ACCEL}_${ROOT_ID}"
mkdir -p "$OUT_ROOT"

./.venv/bin/python3 mri_validation.py \
    --grid-preset "$GRID_PRESET" \
    --filenames "${FILENAMES[@]}" \
    --val-slices "$VAL_SLICES" \
    --test-slices 0 \
    --slice-offset "$SLICE_OFFSET" \
    --seeds "${SEEDS[@]}" \
    --acceleration "$ACCEL" \
    --method-indices "$METHOD_INDICES" \
    --out-dir "$OUT_ROOT" \
    --log-level VAL \
    --evaluate-all \
    --skip-test \
    2>&1 | tee "$OUT_ROOT/run.log"
