#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENDPOINT_METRICS = ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed", "runtime_s")
DETOUR_METRICS = (
    "null_idx_peak",
    "null_idx_auc",
    "inner_null_growth_peak",
    "resid_plateau",
    "resid_min",
    "inner_dist_peak",
    "inner_active_outer_count",
    "x_max_peak",
    "x_max_endpoint",
    "gamma_eff_at_sigma0p1",
    "step_total_max_endpoint",
    "first_bad_outer",
    "last_finite_x_max",
    "last_finite_gamma_eff",
    "last_finite_step_total_max",
    "alpha_min",
    "alpha_max",
    "alpha_mean",
    "alpha_endpoint",
)
METRICS = DETOUR_METRICS + ENDPOINT_METRICS


def read_rows(root, name):
    rows = []
    for path in sorted(root.rglob(name)):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source_csv"] = str(path)
                row["failed"] = str(row.get("failed", "")).lower() == "true"
                try:
                    params = json.loads(row.get("params_json", "{}"))
                except json.JSONDecodeError:
                    params = {}
                for key, value in params.items():
                    row[f"params.{key}"] = value
                rows.append(row)
    return rows


def resolve_path(root, raw_path):
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def compute_detour_metrics(npz_path):
    if npz_path is None or not npz_path.exists():
        return {}
    try:
        data = np.load(npz_path)
    except Exception:
        return {}
    if "inner" not in data or "outer" not in data:
        return {}

    outer_mask = data["inner"] == -1
    inner_mask = data["inner"] >= 0
    if not np.any(outer_mask):
        return {}

    def arr(name):
        if name not in data:
            return None
        return np.asarray(data[name][outer_mask], dtype=float)

    def inner_arr(name):
        if name not in data or not np.any(inner_mask):
            return None
        return np.asarray(data[name][inner_mask], dtype=float)

    outer = arr("outer")
    sigma = arr("sigma")
    null_idx = arr("null_idx")
    growth = arr("inner_null_growth")
    resid = arr("resid")
    inner_dist = arr("inner_dist")
    active = arr("inner_active")
    x_abs_max = arr("x_abs_max")
    gamma_eff = arr("gamma_eff")
    step_total_max = inner_arr("step_total_max")
    step_outer = inner_arr("outer")

    out = {}
    if null_idx is not None and null_idx.size:
        out["null_idx_peak"] = float(np.nanmax(null_idx))
        out["null_idx_auc"] = float(np.trapezoid(null_idx, outer)) if outer is not None and outer.size > 1 else ""
    if growth is not None and growth.size:
        out["inner_null_growth_peak"] = float(np.nanmax(growth))
    if resid is not None and resid.size:
        out["resid_min"] = float(np.nanmin(resid))
        if sigma is not None:
            plateau = resid[(sigma > 1.0) & (sigma <= 5.0) & np.isfinite(resid)]
            out["resid_plateau"] = float(np.nanmedian(plateau)) if plateau.size else ""
    if inner_dist is not None and inner_dist.size:
        out["inner_dist_peak"] = float(np.nanmax(inner_dist))
    if active is not None and active.size:
        out["inner_active_outer_count"] = int(np.nansum(active > 0.5))
    if x_abs_max is not None and x_abs_max.size:
        finite = np.isfinite(x_abs_max)
        out["x_max_peak"] = float(np.nanmax(x_abs_max))
        if np.any(finite):
            out["x_max_endpoint"] = float(x_abs_max[np.where(finite)[0][-1]])
            out["last_finite_x_max"] = float(x_abs_max[np.where(finite)[0][-1]])
    if sigma is not None and gamma_eff is not None and gamma_eff.size:
        finite = np.isfinite(sigma) & np.isfinite(gamma_eff)
        if np.any(finite):
            near_final = np.where(finite & (sigma <= 0.1000001))[0]
            idx = near_final[-1] if near_final.size else np.where(finite)[0][-1]
            out["gamma_eff_at_sigma0p1"] = float(gamma_eff[idx])
            out["last_finite_gamma_eff"] = float(gamma_eff[np.where(finite)[0][-1]])
    if step_total_max is not None and step_total_max.size:
        finite = np.isfinite(step_total_max)
        if np.any(finite):
            out["step_total_max_endpoint"] = float(step_total_max[np.where(finite)[0][-1]])
            out["last_finite_step_total_max"] = float(step_total_max[np.where(finite)[0][-1]])
    bad_masks = []
    for values in (resid, x_abs_max, gamma_eff):
        if values is not None and values.size and outer is not None and values.size == outer.size:
            bad_masks.append(~np.isfinite(values))
    if bad_masks and outer is not None:
        bad = np.logical_or.reduce(bad_masks)
        if np.any(bad):
            out["first_bad_outer"] = int(outer[np.where(bad)[0][0]])
    if step_total_max is not None and step_total_max.size and step_outer is not None:
        bad = ~np.isfinite(step_total_max)
        if np.any(bad):
            first_bad = int(step_outer[np.where(bad)[0][0]])
            previous = out.get("first_bad_outer", "")
            out["first_bad_outer"] = first_bad if previous == "" else min(int(previous), first_bad)
    return out


def compute_alpha_metrics(row):
    try:
        stats = json.loads(row.get("gate_stats_json", "[]") or "[]")
    except json.JSONDecodeError:
        return {}
    if not isinstance(stats, list) or not stats:
        return {}
    means = []
    mins = []
    maxes = []
    for item in stats:
        if not isinstance(item, dict):
            continue
        for target, key in ((means, "alpha_mean"), (mins, "alpha_min"), (maxes, "alpha_max")):
            try:
                target.append(float(item[key]))
            except (KeyError, TypeError, ValueError):
                pass
    out = {}
    if means:
        out["alpha_mean"] = float(sum(means) / len(means))
        out["alpha_endpoint"] = float(means[-1])
    if mins:
        out["alpha_min"] = float(min(mins))
    if maxes:
        out["alpha_max"] = float(max(maxes))
    return out


def add_detour_metrics(root, rows):
    for row in rows:
        npz_path = resolve_path(root, row.get("trajectory_npz", ""))
        metrics = compute_detour_metrics(npz_path)
        metrics.update(compute_alpha_metrics(row))
        for key, value in metrics.items():
            row[key] = value
    return rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_csv(path, failures, all_rows):
    if failures:
        write_csv(path, failures)
        return
    fields = sorted({key for row in all_rows for key in row}) or ["failed"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row.get("method"), row.get("acceleration"), row.get("split"))
        groups.setdefault(key, []).append(row)
    out = []
    for (method, acceleration, split), group in groups.items():
        ok = [row for row in group if not row.get("failed")]
        summary = {
            "method": method,
            "acceleration": acceleration,
            "split": split,
            "n": len(group),
            "n_ok": len(ok),
            "failure_rate": 1.0 - len(ok) / max(1, len(group)),
        }
        for metric in METRICS:
            vals = []
            for row in ok:
                try:
                    vals.append(float(row[metric]))
                except (KeyError, TypeError, ValueError):
                    pass
            if vals:
                summary[f"{metric}_mean"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mu = summary[f"{metric}_mean"]
                    summary[f"{metric}_std"] = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                else:
                    summary[f"{metric}_std"] = 0.0
        out.append(summary)
    return sorted(out, key=lambda row: (row.get("acceleration", ""), row.get("split", ""), row.get("method", "")))


def print_summary(summary):
    print("Detour-first summary:")
    for row in summary:
        print(
            f"  accel={row.get('acceleration')} split={row.get('split')} "
            f"{row.get('method')}: n_ok={row.get('n_ok')}/{row.get('n')} "
            f"null_peak={row.get('null_idx_peak_mean', float('nan')):.3f} "
            f"null_auc={row.get('null_idx_auc_mean', float('nan')):.2f} "
            f"growth_peak={row.get('inner_null_growth_peak_mean', float('nan')):.2f} "
            f"resid_plateau={row.get('resid_plateau_mean', float('nan')):.3e} "
            f"inner_dist_peak={row.get('inner_dist_peak_mean', float('nan')):.3e} "
            f"PSNR={row.get('psnr_mean', float('nan')):.2f} "
            f"SSIM={row.get('ssim_mean', float('nan')):.4f}"
        )


def block_label(method):
    if method == "DAPS":
        return "DAPS"
    if "anchor" in method:
        return "anchor"
    if "outer" in method or "lgvd" in method:
        return "schedule_budget"
    if "noise_temp" in method:
        return "noise"
    if "gamma" in method:
        return "gamma"
    if "warm" in method or "cgsense" in method:
        return "warm"
    if "reanchor" in method:
        return "reanchor"
    if "inner_sigma" in method or "solve_floor" in method:
        return "gate_solve"
    if "target_floor" in method:
        return "regularization"
    return "other"


def write_pareto_plot(root, summary):
    rows = [
        row for row in summary
        if row.get("psnr_mean") not in (None, "")
        and row.get("runtime_s_mean") not in (None, "")
    ]
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping pareto plot: matplotlib unavailable ({exc})")
        return

    colors = {
        "DAPS": "black",
        "anchor": "tab:blue",
        "schedule_budget": "tab:orange",
        "noise": "tab:red",
        "gamma": "tab:purple",
        "warm": "tab:green",
        "reanchor": "tab:brown",
        "gate_solve": "tab:cyan",
        "regularization": "tab:pink",
        "other": "tab:gray",
    }
    groups = {}
    for row in rows:
        key = (row.get("acceleration", ""), row.get("split", ""))
        groups.setdefault(key, []).append(row)

    for (accel, split), group in groups.items():
        plt.figure(figsize=(10, 7))
        seen = set()
        for row in group:
            block = block_label(row.get("method", ""))
            label = block if block not in seen else None
            seen.add(block)
            runtime = float(row["runtime_s_mean"])
            psnr = float(row["psnr_mean"])
            ssim = row.get("ssim_mean", "")
            misfit = row.get("data_misfit_per_observed_mean", "")
            plt.scatter(runtime, psnr, c=colors.get(block, "tab:gray"), label=label, s=55)
            short = row.get("method", "").replace("P-DAPS[", "").replace("]", "")
            if len(short) > 28:
                short = short[:25] + "..."
            suffix = ""
            try:
                suffix = f"\nSSIM={float(ssim):.3f} mis={float(misfit):.3f}"
            except (TypeError, ValueError):
                pass
            plt.annotate(short + suffix, (runtime, psnr), fontsize=6, xytext=(4, 4),
                         textcoords="offset points")
        plt.xlabel("Runtime mean (s)")
        plt.ylabel("PSNR mean")
        title = f"P-DAPS v6 Pareto"
        if accel != "":
            title += f" accel={accel}"
        if split != "":
            title += f" split={split}"
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        suffix = f"accel_{accel}_{split}".strip("_").replace("/", "_")
        plt.savefig(root / f"pareto_psnr_runtime_{suffix}.png", dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Combine and summarize P-DAPS v2 ablation outputs.")
    parser.add_argument("root", help="Run root containing validation_raw.csv/test_raw.csv files.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root, "validation_raw.csv") + read_rows(root, "test_raw.csv")
    rows = add_detour_metrics(root, rows)
    write_csv(root / "ablation_table.csv", rows)
    write_failure_csv(root / "failure_table.csv", [row for row in rows if row.get("failed")], rows)
    summary = summarize(rows)
    write_csv(root / "ablation_summary_by_accel_split.csv", summary)
    write_pareto_plot(root, summary)
    print_summary(summary)
    print(f"rows={len(rows)} summary_rows={len(summary)} root={root}")


if __name__ == "__main__":
    main()
