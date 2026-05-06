#!/bin/bash
#BSUB -J pdaps_match_nfe
#BSUB -q a100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o logs/match_nfe.%J.out
#BSUB -e logs/match_nfe.%J.err

cd /home/dan/DTU/Thesis/pdaps
mkdir -p logs results

OUT=results/mri_validation_match-nfe_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

./.venv/bin/python3 mri_validation.py --grid-preset pdaps_match_nfe \
  --accelerations 4 8 --val-slices 3 --test-slices 5 \
  --out-dir "$OUT" \
  --log-level=DEBUG 2>&1 | tee "$OUT/run.log"
