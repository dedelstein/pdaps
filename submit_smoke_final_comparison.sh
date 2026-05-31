#!/bin/bash
#BSUB -J smoke_final
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/smoke_final.%J.out
#BSUB -e logs/smoke_final.%J.err

set -euo pipefail

mkdir -p logs
export ACCEL=${ACCEL:-4}
JOB_TAG=${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}
export PDAPS_OUT_ROOT=${PDAPS_OUT_ROOT:-results/smoke_final_comparison_accel${ACCEL}_${JOB_TAG}}
./smoke_final_comparison.sh
