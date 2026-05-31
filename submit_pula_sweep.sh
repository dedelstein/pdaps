#!/bin/bash
#BSUB -J pula_sweep
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pula_sweep.%J.out
#BSUB -e logs/pula_sweep.%J.err

set -euo pipefail

mkdir -p logs
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
unset PYTORCH_CUDA_ALLOC_CONF

ACCELS=${ACCELS:-"4 8"}
SLICES=${SLICES:-2}
FILENAME=${FILENAME:-file1000196.h5}
JOB_TAG=${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/pula_sweep_${JOB_TAG}}

echo "pULA sweep job: accels=${ACCELS}, slices=${SLICES}, file=${FILENAME}, out=${OUT_ROOT}"
echo "Allocator env: PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-<unset>} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "Run started at $(date -Is)"

./.venv/bin/python3 pula_sweep.py \
    --filename "$FILENAME" \
    --slices "$SLICES" \
    --accelerations $ACCELS \
    --regime 10:200:linear:poly-7 \
    --regime 10:60:linear:poly-7 \
    --regime 10:200:sqrt:log \
    --regime 1:200:linear:poly-7 \
    --gammas 0.25 0.5 1.0 \
    --out-dir "$OUT_ROOT" \
    --log-level VAL
