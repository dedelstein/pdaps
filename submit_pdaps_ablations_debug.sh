#!/bin/bash
#BSUB -J "pdaps_v2_followup[1-4]"
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 6
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 12:00
#BSUB -u s243446@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/pdaps_v2_followup.%J.%I.out
#BSUB -e logs/pdaps_v2_followup.%J.%I.err

set -euo pipefail

mkdir -p logs

TASK_ID=${LSB_JOBINDEX:-${PDAPS_FOLLOWUP_TASK:-1}}
ROOT_ID=${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/mri_validation_pdaps_v2_followup_${ROOT_ID}}

case "$TASK_ID" in
    1)
        ACCEL=8
        METHOD_INDICES="0-5,19"
        CHUNK_NAME="accel8_priority_lam"
        ;;
    2)
        ACCEL=8
        METHOD_INDICES="6-13"
        CHUNK_NAME="accel8_warm_gate"
        ;;
    3)
        ACCEL=8
        METHOD_INDICES="14-18,20"
        CHUNK_NAME="accel8_noise_init_reanchor"
        ;;
    4)
        ACCEL=4
        METHOD_INDICES="17-20"
        CHUNK_NAME="accel4_unfinished"
        ;;
    *)
        echo "Unknown PDAPS follow-up task index: $TASK_ID" >&2
        exit 2
        ;;
esac

OUT="${OUT_ROOT}/${CHUNK_NAME}"
mkdir -p "$OUT"

echo "PDAPS v2 follow-up task ${TASK_ID}: accel=${ACCEL}, method_indices=${METHOD_INDICES}, out=${OUT}"

./.venv/bin/python3 mri_validation.py \
    --grid-preset pdaps_v2 \
    --filename file1000196.h5 \
    --val-slices 1 \
    --test-slices 0 \
    --test-same-as-val \
    --seeds 123 456 789 \
    --acceleration "$ACCEL" \
    --method-indices "$METHOD_INDICES" \
    --out-dir "$OUT" \
    --log-level DEBUG \
    --evaluate-all \
    --skip-test \
    2>&1 | tee "$OUT/run.log"

./.venv/bin/python3 analyze_pdaps_v2_ablation.py "$OUT" \
    2>&1 | tee "$OUT/analysis.log"

./.venv/bin/python3 plot_pdaps_trajectories.py "$OUT" \
    --out-dir "$OUT/trajectory_plots" \
    2>&1 | tee "$OUT/trajectory_plots.log"

(
    flock 9
    touch "$OUT_ROOT/.done_${TASK_ID}"
    DONE_COUNT=$(find "$OUT_ROOT" -maxdepth 1 -name '.done_*' | wc -l)
    if [ "$DONE_COUNT" -ge 4 ]; then
        ./.venv/bin/python3 analyze_pdaps_v2_ablation.py "$OUT_ROOT" \
            2>&1 | tee "$OUT_ROOT/analysis.log"
        ./.venv/bin/python3 plot_pdaps_trajectories.py "$OUT_ROOT" \
            --out-dir "$OUT_ROOT/trajectory_plots" \
            2>&1 | tee "$OUT_ROOT/trajectory_plots.log"
    fi
) 9>"$OUT_ROOT/.postprocess.lock"
