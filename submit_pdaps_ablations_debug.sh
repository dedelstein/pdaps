#!/bin/bash
#BSUB -J pdaps_v2
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_v2.%J.out
#BSUB -e logs/pdaps_v2.%J.err

set -euo pipefail

STAMP=$(date +%Y%m%d_%H%M%S)

OUT=results/mri_validation_pdaps_v2_${STAMP}
mkdir -p "$OUT"
mkdir -p logs

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_v2 \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 0 \
    --test-same-as-val \
    --seeds 123 456 789 \
    --accelerations 4 8 \
    --out-dir "$OUT" \
    --log-level DEBUG \
    --evaluate-all \
    2>&1 | tee "$OUT/run.log"

./.venv/bin/python3 analyze_pdaps_v2_ablation.py "$OUT" \
    2>&1 | tee "$OUT/analysis.log"

./.venv/bin/python3 plot_pdaps_trajectories.py "$OUT" \
    --out-dir "$OUT/trajectory_plots" \
    2>&1 | tee "$OUT/trajectory_plots.log"
