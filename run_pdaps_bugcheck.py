#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "pdaps_bugcheck_accel4"


def run_validation():
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'libs' / 'inversebench'}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(ROOT / "mri_validation.py"),
        "--grid-preset", "pdaps_bugcheck",
        "--filename", "file1000196.h5",
        "--val-slices", "1",
        "--test-slices", "0",
        "--seeds", "123",
        "--acceleration", "4",
        "--out-dir", str(OUT_DIR),
        "--log-level", "DEBUG",
        "--evaluate-all",
        "--skip-test",
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def summarize():
    raw_csv = OUT_DIR / "validation_raw.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    df = pd.read_csv(raw_csv)
    print("\nCSV gate_stats_json:")
    for _, row in df.iterrows():
        gates = json.loads(row.get("gate_stats_json") or "[]")
        print(f"  {row['method']}: failed={row['failed']} gate_records={len(gates)}")
        if not gates:
            print("    FAIL: gate_stats_json is empty")

    traj_dir = OUT_DIR / "trajectories" / "accel_4"
    if not traj_dir.exists():
        raise FileNotFoundError(traj_dir)

    print("\nTrajectory checks:")
    for p in sorted(traj_dir.glob("P-DAPS_bug_*.npz")):
        z = np.load(p)
        name = p.stem
        print(f"\n{name}")

        if "bug_E_mask_fixed" in name:
            final_null = float(z["null_idx"][-1]) if "null_idx" in z else float("nan")
            max_null_range = (
                float(np.nanmax(z["null_range_ratio"]))
                if "null_range_ratio" in z
                else float("nan")
            )
            print(f"  final_null_idx={final_null:.6g}")
            print(f"  max_null_range_ratio={max_null_range:.6g}")
            print(f"  pass_null_idx={final_null <= 0.2}")
            print(f"  pass_null_range_ratio={max_null_range <= 1.0}")

        if "bug_noisegate" in name:
            if "noise_scale" not in z:
                print("  FAIL: no noise_scale in trajectory")
                continue
            noise_scale = z["noise_scale"]
            vals = sorted(set(float(v) for v in noise_scale if np.isfinite(v)))
            inner = z["inner"] >= 0 if "inner" in z else np.ones_like(noise_scale, dtype=bool)
            inner_noise = noise_scale[inner]
            early = inner_noise[:200] if len(inner_noise) >= 200 else inner_noise
            late = inner_noise[-200:] if len(inner_noise) >= 200 else inner_noise
            print(f"  noise_scale_values={vals}")
            print(f"  early_noise_scale_mean={float(np.nanmean(early)):.6g}")
            print(f"  late_noise_scale_mean={float(np.nanmean(late)):.6g}")
            print(f"  pass_has_0_and_1={0.0 in vals and 1.0 in vals}")

        if "bug_full_nt0p005_sigmaearly1p0" in name:
            if "noise_scale" not in z:
                print("  FAIL: no noise_scale in trajectory")
                continue
            inner = z["inner"] >= 0 if "inner" in z else np.ones_like(z["noise_scale"], dtype=bool)
            sigma = z["sigma"][inner]
            noise_scale = z["noise_scale"][inner]
            expected = (sigma >= 1.0).astype(float)
            final_null = float(z["null_idx"][-1]) if "null_idx" in z else float("nan")
            max_null_range = (
                float(np.nanmax(z["null_range_ratio"]))
                if "null_range_ratio" in z
                else float("nan")
            )
            print(f"  sigmaearly_noise_scale_values={sorted(set(float(v) for v in noise_scale if np.isfinite(v)))}")
            print(f"  pass_sigmaearly_pattern={bool(np.all(noise_scale == expected))}")
            print(f"  final_null_idx={final_null:.6g}")
            print(f"  max_null_range_ratio={max_null_range:.6g}")
            if "noise_gate_resid" in z:
                resid = z["noise_gate_resid"][inner]
                print(f"  noise_gate_resid_min={float(np.nanmin(resid)):.6g}")
                print(f"  noise_gate_resid_max={float(np.nanmax(resid)):.6g}")


def main():
    run_validation()
    summarize()


if __name__ == "__main__":
    main()
