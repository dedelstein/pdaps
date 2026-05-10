#!/bin/bash
#BSUB -J pdaps_phaseB
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_phaseB.%J.out
#BSUB -e logs/pdaps_phaseB.%J.err

set -euo pipefail

STAMP=$(date +%Y%m%d_%H%M%S)

OUT=results/mri_validation_nullspace_focus_${STAMP}
mkdir -p "$OUT"

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_nullspace_focus \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 1 \
    --seeds 123 456 789 \
    --accelerations 4 8 \
    --out-dir "$OUT" \
    --log-level DEBUG \
    2>&1 | tee "$OUT/run.log"

OUT=results/mri_validation_pdaps_mechanism_${STAMP}
mkdir -p "$OUT"

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_mechanism \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 1 \
    --accelerations 4 8 \
    --out-dir "$OUT" \
    --log-level DEBUG \
    2>&1 | tee "$OUT/run.log"
