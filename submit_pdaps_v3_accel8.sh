#!/bin/bash
#BSUB -J pdaps_v3_accel8
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 18:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_v3_accel8.%J.out
#BSUB -e logs/pdaps_v3_accel8.%J.err

set -euo pipefail

STAMP=${STAMP:-${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_pdaps_v3_accel8_${STAMP}}
mkdir -p "$OUT_ROOT"

echo "PDAPS v3 ablation: accel=8, out=${OUT_ROOT}"

./.venv/bin/python3 mri_validation_2.py \
    --grid-preset pdaps_v3 \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 0 \
    --test-same-as-val \
    --seeds 123 456 789 \
    --accelerations 8 \
    --out-dir "$OUT_ROOT" \
    --log-level DEBUG \
    --evaluate-all \
    2>&1 | tee "$OUT_ROOT/run.log"

./.venv/bin/python3 analyze_pdaps_v2_ablation.py "$OUT_ROOT" \
    2>&1 | tee "$OUT_ROOT/analyze.log"
