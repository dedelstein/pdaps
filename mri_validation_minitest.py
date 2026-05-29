#!/usr/bin/env python3
"""
Standalone UQ null-split probe for P-DAPS on one fastMRI slice.

This intentionally avoids adding a grid preset or CLI surface to
`mri_validation.py`. It reuses that file's model/data helpers, then writes a
small probe table and mean/std/error panels for the null-split UQ question.
"""

import argparse
import csv
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import MultiCoilMRIDataset
from mri_validation import DAPS_TAU, load_model, make_forward_op, move_to_device, pdaps_entry
from utilities import compute_metrics_dict


UQ_CELLS = [
    ("drift_anchor", "none", 0.0, 1e-3),
    ("full_nullp3_nt0p01", "full", 0.01, 0.3),
    ("full_nullp3_nt0p05", "full", 0.05, 0.3),
    ("full_null1_nt0p05", "full", 0.05, 1.0),
    ("full_null3_nt0p05", "full", 0.05, 3.0),
    ("full_null1_nt0p1", "full", 0.1, 1.0),
    ("nullonly_null1_nt0p05", "null_only", 0.05, 1.0),
    ("full_null1_nt0p05_nosplit", "full", 0.05, 1e-3),
]


def parse_index_selection(raw, n_entries):
    selected = []
    seen = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, stop_s = part.split("-", 1)
            values = range(int(start_s), int(stop_s) + 1)
        else:
            values = [int(part)]
        for idx in values:
            if idx < 0 or idx >= n_entries:
                raise ValueError(f"Cell index {idx} out of range for {n_entries} cells")
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    return selected


def sanitize(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def make_entry(label, noise_mode, noise_tau, target_null_lam_floor, log_level):
    entry = pdaps_entry(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=50,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=noise_tau,
        noise_mode=noise_mode,
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
        sigma_stop_truncate=0.17,
        label_suffix=f"uq_{label}",
    )
    entry["params"]["noise_mode"] = noise_mode
    entry["params"]["target_null_lam_floor"] = float(target_null_lam_floor)
    entry["algorithm"]["lgvd_config"]["target_null_lam_floor"] = float(target_null_lam_floor)
    return entry


def to_complex(x):
    if torch.is_complex(x):
        return x
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())


def to_real(x):
    return torch.stack([x.real, x.imag], dim=1)


def pearson_corr(a, b):
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    if denom.item() <= 0.0:
        return float("nan")
    return float((a * b).sum().div(denom).item())


def save_panel(path, x_bar_complex, std_map, error_map, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_img = x_bar_complex.abs().squeeze().detach().cpu().numpy()
    std_img = std_map.squeeze().detach().cpu().numpy()
    err_img = error_map.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (mean_img, "posterior mean", "gray"),
        (std_img, "empirical std", "magma"),
        (err_img, "|mean - target|", "magma"),
    ]
    for ax, (img, name, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    out = []
    by_accel = {}
    for row in rows:
        if not row.get("failed"):
            by_accel.setdefault(int(row["acceleration"]), {})[row["label"]] = row

    for row in rows:
        summary = dict(row)
        anchor = by_accel.get(int(row["acceleration"]), {}).get("drift_anchor")
        if anchor and not row.get("failed"):
            psnr_drop = float(anchor["psnr"]) - float(row["psnr"])
            ssim_drop = float(anchor["ssim"]) - float(row["ssim"])
            null_gain = float(row["std_null_mean"]) / max(float(anchor["std_null_mean"]), 1e-12)
            summary["psnr_drop_vs_anchor"] = psnr_drop
            summary["ssim_drop_vs_anchor"] = ssim_drop
            summary["std_null_gain_vs_anchor"] = null_gain
            summary["quality_disqualified"] = bool(psnr_drop > 0.36 or ssim_drop > 0.012)
            summary["diversity_win"] = bool(
                not summary["quality_disqualified"]
                and null_gain > 2.0
                and float(row["std_error_corr"]) > float(anchor["std_error_corr"])
            )
        out.append(summary)
    return out


def run_cell(entry, label, accel, sample, sample_idx, filename, net, args, out_dir):
    device = next(net.parameters()).device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    forward_op = make_forward_op(args, device)
    algo = hydra.utils.instantiate(OmegaConf.create(entry["algorithm"]), forward_op=forward_op, net=net)
    data = move_to_device(sample, device)
    data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    observation = forward_op(data)
    target = data["target"]

    start = time.perf_counter()
    samples = algo.inference(observation, num_samples=args.num_samples, verbose=args.verbose)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_s = time.perf_counter() - start

    samples_complex = to_complex(samples)
    x_bar = samples_complex.mean(dim=0, keepdim=True)
    x_bar_real = to_real(x_bar)
    target_complex = to_complex(target)

    centered = samples_complex - x_bar
    std_map = centered.abs().square().mean(dim=0, keepdim=True).sqrt()
    std_range = algo.project_range(centered).abs().square().mean(dim=0, keepdim=True).sqrt()
    std_null = algo.project_null(centered).abs().square().mean(dim=0, keepdim=True).sqrt()
    error_map = (x_bar - target_complex).abs()

    metrics = compute_metrics_dict(forward_op, x_bar_real, target, observation)
    null_idx = algo.nullspace_energy(x_bar).mean().item()
    figure_path = out_dir / "figures" / f"accel_{accel}" / f"{sanitize(label)}.png"
    save_panel(
        figure_path,
        x_bar,
        std_map,
        error_map,
        f"{label}  R={accel}  n={args.num_samples}",
    )

    row = {
        "label": label,
        "method": entry["method"],
        "params_json": json.dumps(entry["params"], sort_keys=True),
        "acceleration": accel,
        "filename": filename,
        "sample_idx": sample_idx,
        "h5_slice": int(args.h5_slice),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "failed": False,
        "runtime_s": runtime_s,
        "std_mean": float(std_map.mean().item()),
        "std_range_mean": float(std_range.mean().item()),
        "std_null_mean": float(std_null.mean().item()),
        "std_null_range_ratio": float(std_null.mean().div(std_range.mean().clamp_min(1e-12)).item()),
        "std_error_corr": pearson_corr(std_map, error_map),
        "coverage_2std": float((error_map < 2.0 * std_map).float().mean().item()),
        "nullspace_energy_xbar": float(null_idx),
        "figure_path": str(figure_path),
    }
    row.update(metrics)
    row["failed"] = not bool(metrics.get("finite", False)) or not all(
        math.isfinite(float(row[name]))
        for name in ("psnr", "ssim", "nmse", "std_mean", "std_null_mean", "max_abs")
    )
    if row["failed"]:
        row["error"] = "nonfinite_reconstruction_or_metrics"
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone one-slice P-DAPS UQ null-split probe.")
    parser.add_argument("--models-dir", default="/dtu/blackhole/1d/214141/Thesis/models")
    parser.add_argument("--ckpt-name", default="MRI-knee.pt")
    parser.add_argument("--kspace-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val")
    parser.add_argument("--maps-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit")
    parser.add_argument("--filename", default="file1000196.h5")
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--accelerations", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cell-indices", default=None, help="Comma/range selection, e.g. 0,2,5-7.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "VAL"], default="VAL")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-grid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cells = UQ_CELLS
    if args.cell_indices:
        indices = parse_index_selection(args.cell_indices, len(cells))
        cells = [cells[idx] for idx in indices]
    entries = [
        make_entry(label, noise_mode, noise_tau, target_null_lam_floor, args.log_level)
        for label, noise_mode, noise_tau, target_null_lam_floor in cells
    ]

    if args.list_grid:
        print(json.dumps(entries, indent=2))
        return

    out_dir = Path(args.out_dir or f"results/mri_validation_minitest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(entries, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args, device)

    dataset = MultiCoilMRIDataset(
        args.kspace_dir,
        args.maps_dir,
        args.image_size,
        filenames=[Path(args.filename).name],
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No usable slices found for {args.filename}")
    if args.slice_offset < 0 or args.slice_offset >= len(dataset):
        raise ValueError(f"--slice-offset {args.slice_offset} out of range for {len(dataset)} usable slices")
    sample = dataset[args.slice_offset]
    sample_idx = args.slice_offset
    filename = Path(args.filename).name
    args.h5_slice = int(dataset.samples[args.slice_offset][2])

    rows = []
    for accel in args.accelerations:
        args.acceleration = int(accel)
        print(f"=== acceleration {accel}x ===", flush=True)
        for entry, (label, _noise_mode, _noise_tau, _target_null) in zip(entries, cells):
            print(f"[{label}] running {args.num_samples} samples", flush=True)
            try:
                row = run_cell(entry, label, int(accel), sample, sample_idx, filename, net, args, out_dir)
            except Exception as exc:
                row = {
                    "label": label,
                    "method": entry["method"],
                    "params_json": json.dumps(entry["params"], sort_keys=True),
                    "acceleration": int(accel),
                    "filename": filename,
                    "sample_idx": sample_idx,
                    "h5_slice": int(args.h5_slice),
                    "seed": args.seed,
                    "num_samples": args.num_samples,
                    "failed": True,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            rows.append(row)
            write_csv(out_dir / "raw.csv", rows)
            if row.get("failed"):
                print(f"[{label}] FAILED: {row.get('error', 'unknown')}", flush=True)
            else:
                print(
                    f"[{label}] PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} "
                    f"std_null={row['std_null_mean']:.3e} corr={row['std_error_corr']:.3f}",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    write_csv(out_dir / "summary.csv", summary)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
