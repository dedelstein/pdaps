#!/bin/bash
#BSUB -J pdaps_ablate_dbg
#BSUB -q a100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o logs/pdaps_ablate_dbg.%J.out
#BSUB -e logs/pdaps_ablate_dbg.%J.err

cd /home/dan/DTU/Thesis/pdaps
mkdir -p logs results

OUT=results/pdaps_ablations_debug_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

# Targeted diagnostic run:
#   13 entries = DAPS + 12 P-DAPS ablations
#   2 accelerations = R4, R8
#   1 validation + 1 test slice per acceleration keeps DEBUG logs tractable
# while preserving the high-value trace numbers: gamma, lambda, drift/noise
# solve size, residual, null_idx, grow_tot, and grow_meas.
./.venv/bin/python3 mri_validation.py --grid-preset pdaps_ablations \
  --accelerations 4 8 --val-slices 1 --test-slices 1 \
  --out-dir "$OUT" \
  --log-level=DEBUG 2>&1 | tee "$OUT/run.log"
