#!/bin/bash

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ACCEL=${ACCEL:-4}
GRID_PRESET="final_comparison"
SPLIT_JSON=${FINAL_SPLIT_JSON:-final_split.json}
SMOKE_VAL_PATIENTS=${SMOKE_VAL_PATIENTS:-1}
SMOKE_TEST_PATIENTS=${SMOKE_TEST_PATIENTS:-1}
VAL_SLICES=${VAL_SLICES:-1}
TEST_SLICES=${TEST_SLICES:-1}
SLICE_OFFSET=${SLICE_OFFSET:-0}
SEED=${SEED:-123}

# One representative config per method in final_comparison:
#   0=DPS, 4=DAPS, 6=pULA, 9=P-DAPS-core
METHOD_INDICES=${METHOD_INDICES:-0,4,6,9}
OUT_ROOT=${PDAPS_OUT_ROOT:-results/smoke_${GRID_PRESET}_accel${ACCEL}}

mkdir -p "$OUT_ROOT"

if [[ ! -f "$SPLIT_JSON" ]]; then
    ./.venv/bin/python3 make_final_split.py --out "$SPLIT_JSON"
fi

mapfile -t VAL_FILENAMES < <(
    ./.venv/bin/python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("\n".join(s["val"][:int(sys.argv[2])]))' \
        "$SPLIT_JSON" "$SMOKE_VAL_PATIENTS"
)
mapfile -t TEST_FILENAMES < <(
    ./.venv/bin/python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("\n".join(s["test"][:int(sys.argv[2])]))' \
        "$SPLIT_JSON" "$SMOKE_TEST_PATIENTS"
)

echo "Smoke final comparison: accel=${ACCEL}, methods=${METHOD_INDICES}, val=${VAL_FILENAMES[*]}, test=${TEST_FILENAMES[*]}, out=${OUT_ROOT}"
echo "Run started at $(date -Is)" | tee -a "$OUT_ROOT/run.log"

./.venv/bin/python3 mri_validation.py \
    --grid-preset "$GRID_PRESET" \
    --method-indices "$METHOD_INDICES" \
    --filenames "${VAL_FILENAMES[@]}" \
    --test-filenames "${TEST_FILENAMES[@]}" \
    --val-slices "$VAL_SLICES" \
    --test-slices "$TEST_SLICES" \
    --slice-offset "$SLICE_OFFSET" \
    --seed "$SEED" \
    --acceleration "$ACCEL" \
    --pattern random \
    --mask-seed 0 \
    --out-dir "$OUT_ROOT" \
    --resume \
    --log-level VAL \
    2>&1 | tee -a "$OUT_ROOT/run.log"

./.venv/bin/python3 - <<'PY' "$OUT_ROOT"
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
required = ["validation_raw.csv", "validation_summary.csv", "selected.json", "test_raw.csv", "test_summary.csv"]
missing = [name for name in required if not (out / name).exists()]
if missing:
    raise SystemExit(f"Missing smoke outputs: {missing}")

selected = json.load(open(out / "selected.json"))
if set(selected) != {"DPS", "DAPS", "pULA", "P-DAPS-core"}:
    raise SystemExit(f"Unexpected selected methods: {sorted(selected)}")

with open(out / "test_raw.csv", newline="") as f:
    test_rows = list(csv.DictReader(f))
if not test_rows:
    raise SystemExit("test_raw.csv is empty")
failed = [row for row in test_rows if str(row.get("failed", "")).lower() == "true"]
if failed:
    raise SystemExit(f"Smoke test has failed rows; inspect {out / 'test_raw.csv'}")

print(f"Smoke OK: {len(test_rows)} test rows, selected methods {sorted(selected)}")
PY
