#!/bin/bash
set -euo pipefail

# Run MRI validation grids over one or more accelerations.
# All four methods (DPS, DAPS, pULA, P-DAPS) are evaluated; per-accel
# val→select→test runs into accel_<R>/ subdirs.
#
# Usage:
#   ./run_mri_validation.sh [preset] [-- extra args]
#
# Presets (early-stage exploratory; full defensible runs come later):
#   robust         Strong introductory validation: R∈{1,4,8}, 3 patient files,
#                  2 val + 6 test slices per file (18 test slices/accel).
#                  ≈9 h wall on L40s. R=1 is a fully-sampled sanity ceiling.
#   probe          Wider grid at R=4 and R=8; extends stiff end so R=8 is
#                  actually probed. Single-file exploratory run.
#   tiny           Narrower grid (R=4-tuned) at R=4 and R=8.
#   tiny-r4        Narrow grid, R=4 only.
#   tiny-r8        Narrow grid, R=8 only.
#   smoke          Single point per method at R=4. Fast sanity check.
#   warm-sweep     warm_fraction sweep at R=4 and R=8.
#   iso-nfe        lgvd_num_steps sweep at R=4 and R=8.
#   inner-sweep    inner_sigma_max sweep at R=4 and R=8.
#
# When --filenames is set (multi-file presets like `robust`), --val-slices
# and --test-slices are interpreted *per file*. Otherwise per-run.
# Override slice counts with `-- --val-slices N --test-slices M`.
#
# Examples:
#   ./run_mri_validation.sh probe                   # main exploratory run
#   ./run_mri_validation.sh probe -- --list-grid    # show grid, don't run
#   ./run_mri_validation.sh probe -- --test-slices 10
#   ./run_mri_validation.sh smoke -- --verbose
#
# Env overrides:
#   PYTHON=path/to/python   (default ./.venv/bin/python3)
#   DRY_RUN=1               Print the command and exit without running.

PRESET=${1:-"probe"}
shift || true

# Strip optional `--` separator before user-supplied extra args.
if [[ ${1:-} == "--" ]]; then shift; fi
EXTRA=("$@")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PYTHON_BIN=${PYTHON:-"./.venv/bin/python3"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

OUT_DIR="results/mri_validation_${PRESET}_${TIMESTAMP}"

# Default slice counts. Overridable via EXTRA args (argparse takes the last
# value, so user-supplied --val-slices / --test-slices win).
DEFAULT_SLICES=(--val-slices 2 --test-slices 3)
case "$PRESET" in
    robust)
        # 3 patient files × {1, 4, 8} accelerations.
        # NOTE: confirm filenames exist in --maps-dir before launching;
        # override with `-- --filenames <a.h5> <b.h5> <c.h5>` if needed.
        ARGS=(--grid-preset probe --accelerations 1 4 8 \
              --filenames file1000196.h5 file1000206.h5 file1000229.h5)
        DEFAULT_SLICES=(--val-slices 2 --test-slices 6)
        ;;
    probe)
        ARGS=(--grid-preset probe --accelerations 4 8)
        DEFAULT_SLICES=(--val-slices 3 --test-slices 5)
        ;;
    tiny)
        ARGS=(--grid-preset tiny --accelerations 4 8)
        ;;
    tiny-r4)
        ARGS=(--grid-preset tiny --acceleration 4)
        ;;
    tiny-r8)
        ARGS=(--grid-preset tiny --acceleration 8)
        ;;
    smoke)
        ARGS=(--grid-preset smoke --acceleration 4)
        DEFAULT_SLICES=(--val-slices 1 --test-slices 1)
        ;;
    warm-sweep)
        ARGS=(--grid-preset warm_sweep --accelerations 4 8)
        DEFAULT_SLICES=(--val-slices 3 --test-slices 5)
        ;;
    iso-nfe)
        ARGS=(--grid-preset iso_nfe --accelerations 4 8)
        DEFAULT_SLICES=(--val-slices 3 --test-slices 5)
        ;;
    inner-sweep)
        ARGS=(--grid-preset pdaps_inner_sweep --accelerations 4 8)
        DEFAULT_SLICES=(--val-slices 3 --test-slices 5)
        ;;
    *)
        echo "Unknown preset: $PRESET" >&2
        echo "See header for valid presets." >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

# Compose the full argument list. User-supplied EXTRA goes last so it
# overrides DEFAULT_SLICES if --val-slices / --test-slices are repeated.
CMD=("$PYTHON_BIN" mri_validation.py
     "${ARGS[@]}"
     "${DEFAULT_SLICES[@]}"
     --out-dir "$OUT_DIR"
     "${EXTRA[@]}")

echo "Preset:       $PRESET"
echo "Out dir:      $OUT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Command:      ${CMD[*]}"
echo "Log:          $LOG"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 — exiting without launching."
    exit 0
fi

# Snapshot HEAD-ish state (best-effort; repo isn't necessarily git).
{
    echo "preset=$PRESET"
    echo "timestamp=$TIMESTAMP"
    echo "host=$(hostname)"
    echo "python=$PYTHON_BIN"
    echo "command=${CMD[*]}"
    if command -v nvidia-smi &> /dev/null; then
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    fi
} > "$OUT_DIR/run_meta.txt" 2>&1 || true

START_EPOCH=$(date +%s)
"${CMD[@]}" 2>&1 | tee "$LOG"
END_EPOCH=$(date +%s)

ELAPSED=$((END_EPOCH - START_EPOCH))
printf '\nElapsed: %dh %dm %ds\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) | tee -a "$LOG"
echo "Done. Results in $OUT_DIR"
