#!/bin/bash
#BSUB -J pdaps_ablate_dbg
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 16:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_ablate_dbg.%J.out
#BSUB -e logs/pdaps_ablate_dbg.%J.err

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
