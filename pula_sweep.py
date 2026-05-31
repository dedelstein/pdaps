#!/usr/bin/env python3
"""
Tiny standalone pULA hyperparameter sweep.

Runs pULA-only validation rows over a small paired (sigma_max, N) regime grid
and gamma sweep. This intentionally avoids adding another preset to
mri_validation.py.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from dataloader import MultiCoilMRIDataset
from mri_validation import (
    PROJECT_ROOT,
    _load_resume_rows,
    _select_samples,
    _successful_row_keys,
    load_model,
    run_one,
    summarize,
    write_ablation_artifacts,
    write_csv,
)


def parse_regime(raw):
    parts = raw.split(":")
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError(
            f"Expected SIGMA_MAX:NUM_STEPS[:SCHEDULE:TIMESTEP], got {raw!r}"
        )
    sigma_s, steps_s = parts[:2]
    schedule, timestep = ("sqrt", "log") if len(parts) == 2 else parts[2:]
    try:
        sigma_max = float(sigma_s)
        num_steps = int(steps_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid regime {raw!r}") from exc
    if sigma_max <= 0 or num_steps <= 0:
        raise argparse.ArgumentTypeError(f"Regime values must be positive, got {raw!r}")
    if schedule not in {"sqrt", "linear", "vp"}:
        raise argparse.ArgumentTypeError(
            f"Invalid schedule {schedule!r}; expected one of sqrt, linear, vp"
        )
    if timestep not in {"log", "vp"} and not (
        timestep.startswith("poly-") and timestep.removeprefix("poly-").isdigit()
    ):
        raise argparse.ArgumentTypeError(
            f"Invalid timestep {timestep!r}; expected log, vp, or poly-<int>"
        )
    return sigma_max, num_steps, schedule, timestep


def regime_label(sigma_max, num_steps, schedule, timestep):
    sigma = f"{sigma_max:g}".replace(".", "p")
    timestep_label = timestep.replace("-", "")
    return f"smax{sigma}_N{num_steps}_{schedule}_{timestep_label}"


def make_pula_entries(regimes, gammas, log_level, k_steps, cg_iter):
    entries = []
    for sigma_max, num_steps, schedule, timestep in regimes:
        scheduler = {
            "num_steps": int(num_steps),
            "sigma_max": float(sigma_max),
            "sigma_min": 0.01,
            "sigma_final": 0,
            "schedule": schedule,
            "timestep": timestep,
        }
        for gamma in gammas:
            label = regime_label(sigma_max, num_steps, schedule, timestep)
            gamma_label = f"{gamma:g}".replace(".", "p")
            params = {
                "sigma_max": float(sigma_max),
                "num_steps": int(num_steps),
                "gamma": float(gamma),
                "K": int(k_steps),
                "cg_iter": int(cg_iter),
                "schedule": schedule,
                "timestep": timestep,
                "candidate": f"pULA[{label}_g{gamma_label}]",
            }
            entries.append({
                "method": "pULA",
                "params": params,
                "algorithm": {
                    "_target_": "algo.pula.pULA",
                    "noise_scheduler_config": scheduler,
                    "K": int(k_steps),
                    "gamma": float(gamma),
                    "cg_iter": int(cg_iter),
                    "log_level": log_level,
                },
            })
    return entries


def parse_args():
    parser = argparse.ArgumentParser(description="Two-slice pULA sweep for MRI validation.")
    parser.add_argument("--models-dir", default="/dtu/blackhole/1d/214141/Thesis/models")
    parser.add_argument("--ckpt-name", default="MRI-knee.pt")
    parser.add_argument("--kspace-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val")
    parser.add_argument("--maps-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit")
    parser.add_argument("--filename", default="file1000196.h5")
    parser.add_argument("--filenames", nargs="+", default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--accelerations", nargs="+", type=int, default=None)
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--slices", type=int, default=2, help="Number of validation slices per file.")
    parser.add_argument("--regime", dest="regimes", type=parse_regime, action="append",
                        default=None, metavar="SIGMA_MAX:NUM_STEPS[:SCHEDULE:TIMESTEP]",
                        help="Paired pULA regime. Repeatable. Defaults use sqrt/log: 0.1:20, 1:40, 10:60.")
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--cg-iter", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "VAL"], default="VAL")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-grid", action="store_true")
    return parser.parse_args()


def row_key(entry, sample_idx, filename, seed, acceleration):
    return (
        entry["method"],
        json.dumps(entry["params"], sort_keys=True),
        "validation",
        str(sample_idx),
        filename,
        str(seed),
        str(acceleration),
    )


def run_acceleration(args, entries, net, dataset, samples, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(entries, f, indent=2)

    rows = _load_resume_rows(out_dir / "validation_raw.csv", args.resume)
    done = _successful_row_keys(rows)
    seed_values = list(args.seeds) if args.seeds else [args.seed]

    for entry in entries:
        for seed in seed_values:
            args.seed = int(seed)
            for sample_idx, filename in samples:
                key = row_key(entry, sample_idx, filename, args.seed, args.acceleration)
                if key in done:
                    continue
                sample = dataset[sample_idx]
                row = run_one(
                    entry,
                    sample,
                    sample_idx,
                    "validation",
                    net,
                    args,
                    out_dir,
                    save_image=args.save_images,
                    filename=filename,
                )
                del sample
                rows.append(row)
                done.add(row_key(entry, sample_idx, filename, args.seed, args.acceleration))
                write_csv(out_dir / "validation_raw.csv", rows)

    write_csv(out_dir / "validation_summary.csv", summarize(rows))
    write_ablation_artifacts(out_dir, rows, [])
    print(f"Wrote {out_dir}")


def main():
    args = parse_args()
    regimes = args.regimes or [
        (0.1, 20, "sqrt", "log"),
        (1.0, 40, "sqrt", "log"),
        (10.0, 60, "sqrt", "log"),
    ]
    entries = make_pula_entries(regimes, args.gammas, args.log_level, args.K, args.cg_iter)
    if args.list_grid:
        print(json.dumps(entries, indent=2))
        return

    out_dir = Path(args.out_dir or f"results/pula_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args, device)

    filenames = args.filenames if args.filenames else [args.filename]
    dataset = MultiCoilMRIDataset(args.kspace_dir, args.maps_dir, args.image_size, filenames=filenames)
    samples, _ = _select_samples(dataset, args.slice_offset, args.slices, 0)
    print(f"pULA sweep: {len(entries)} cells x {len(samples)} slice(s), files={', '.join(filenames)}")

    accelerations = args.accelerations if args.accelerations else [args.acceleration]
    if len(accelerations) == 1:
        args.acceleration = int(accelerations[0])
        run_acceleration(args, entries, net, dataset, samples, out_dir)
    else:
        for acceleration in accelerations:
            args.acceleration = int(acceleration)
            run_acceleration(args, entries, net, dataset, samples, out_dir / f"accel_{acceleration}")


if __name__ == "__main__":
    main()
