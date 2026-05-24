#!/bin/bash
#BSUB -J pdaps_v8c_accel4
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_v8c_accel4.%J.out
#BSUB -e logs/pdaps_v8c_accel4.%J.err

set -euo pipefail

ACCEL=4
ROOT_ID=${PDAPS_ROOT_ID:-${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_pdaps_v8c_accel${ACCEL}_${ROOT_ID}}
mkdir -p "$OUT_ROOT"

METHOD_SELECTION=${METHOD_INDICES:-0-41}

SAVE_IMAGES_ARGS=()
if [[ "${SAVE_IMAGES:-0}" == "1" ]]; then
    SAVE_IMAGES_ARGS=(--save-images)
fi

echo "PDAPS v8c rescue-surface validation: accel=${ACCEL}, methods=${METHOD_SELECTION}, out=${OUT_ROOT}"

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_v8c \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 0 \
    --seeds 123 \
    --acceleration "$ACCEL" \
    --method-indices "$METHOD_SELECTION" \
    --out-dir "$OUT_ROOT" \
    --log-level DEBUG \
    --evaluate-all \
    --skip-test \
    "${SAVE_IMAGES_ARGS[@]}" \
    2>&1 | tee "$OUT_ROOT/run.log"
