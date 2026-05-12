#!/bin/bash
#BSUB -J pdaps_v2_followup
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 24:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_v2_followup.%J.out
#BSUB -e logs/pdaps_v2_followup.%J.err

set -euo pipefail

mkdir -p logs

ROOT_ID=${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_pdaps_v2_followup_${ROOT_ID}}
mkdir -p "$OUT_ROOT"

run_chunk() {
    local accel=$1
    local method_indices=$2
    local chunk_name=$3
    local out="${OUT_ROOT}/${chunk_name}"
    mkdir -p "$out"

    echo "PDAPS v2 follow-up chunk: accel=${accel}, method_indices=${method_indices}, out=${out}"

    ./.venv/bin/python3 mri_validation.py \
        --grid-preset pdaps_v2 \
        --filename file1000196.h5 \
        --val-slices 1 \
        --test-slices 0 \
        --test-same-as-val \
        --seeds 123 456 789 \
        --acceleration "$accel" \
        --method-indices "$method_indices" \
        --out-dir "$out" \
        --log-level DEBUG \
        --evaluate-all \
        --skip-test \
        2>&1 | tee "$out/run.log"
}

run_chunk 8 "0-5,19" "accel8_priority_lam"
run_chunk 8 "6-13" "accel8_warm_gate"
run_chunk 8 "14-18,20" "accel8_noise_init_reanchor"
run_chunk 4 "17-20" "accel4_unfinished"
