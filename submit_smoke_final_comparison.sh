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
./smoke_final_comparison.sh
