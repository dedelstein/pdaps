#!/bin/bash

set -euo pipefail

GRID_PRESET="pdaps_prelaunch_lrmin"
OUT_ROOT="results/mri_validation_${GRID_PRESET}_interactive_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_ROOT"

./.venv/bin/python3 mri_validation.py \
    --grid-preset "$GRID_PRESET" \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 0 \
    --seed 123 \
    --accelerations 4 8 \
    --out-dir "$OUT_ROOT" \
    --log-level VAL \
    --evaluate-all \
    --skip-test \
    2>&1 | tee "$OUT_ROOT/run.log"
