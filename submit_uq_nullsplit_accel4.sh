#!/bin/bash
#BSUB -J uq_nullsplit_accel4
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/uq_nullsplit_accel4.%J.out
#BSUB -e logs/uq_nullsplit_accel4.%J.err

set -euo pipefail

ACCEL=4
ROOT_ID=${PDAPS_ROOT_ID:-${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_uq_nullsplit_accel${ACCEL}_${ROOT_ID}}
mkdir -p "$OUT_ROOT"

CELL_SELECTION=${CELL_INDICES:-0-7}
NUM_SAMPLES=${NUM_SAMPLES:-8}

echo "P-DAPS UQ null-split minitest: accel=${ACCEL}, cells=${CELL_SELECTION}, num_samples=${NUM_SAMPLES}, out=${OUT_ROOT}"

./.venv/bin/python3 mri_validation_minitest.py \
    --filename file1000196.h5 \
    --num-samples "$NUM_SAMPLES" \
    --accelerations "$ACCEL" \
    --cell-indices "$CELL_SELECTION" \
    --out-dir "$OUT_ROOT" \
    --log-level DEBUG \
    2>&1 | tee "$OUT_ROOT/run.log"
