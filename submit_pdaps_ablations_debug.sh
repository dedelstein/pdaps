#!/bin/bash
#BSUB -J pdaps_ablate_dbg
#BSUB -q gpul40s
#BSUB -gpu 
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 06:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_ablate_dbg.%J.out
#BSUB -e logs/pdaps_ablate_dbg.%J.err

OUT=results/pdaps_ablations_debug_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_remediation \
    --val-slices 1 --test-slices 1 \
    --accelerations 4 8 \
    --log-level DEBUG
