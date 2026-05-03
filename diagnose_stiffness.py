"""Decompose existing stiffness-sweep CSVs to localize the cause of non-monotonicity.

Reads the latest formal_*/ output and emits per-component plots that separate the
loss-design, GT-geometry, divergence-fallback, and seed-noise contributions to the
Study 1 (cond x alpha) and Study 4 (step-size stability) curves.

Usage:
    python diagnose_stiffness.py [results_dir]

If no dir is given, uses the most recent results/formal_* directory.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


PRIMARY_METHODS = ("DPS", "DAPS", "pULA", "P-DAPS")


def latest_formal_dir() -> str:
    candidates = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "formal_*")))
    if not candidates:
        raise SystemExit("no results/formal_* directory found")
    return candidates[-1]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def filter_method(rows: list[dict], method: str) -> list[dict]:
    return [r for r in rows if r.get("method") == method]


def study1_decomposition(csv_path: str, out_dir: str) -> None:
    rows = load_csv(csv_path)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        m = r["method"]
        # Keep only the alpha=0.0 P-DAPS row and the baselines per condition
        if m == "P-DAPS" and r["alpha"] != "0.0":
            continue
        if m.startswith("P-DAPS-warm"):
            continue
        by_method[m].append(r)

    # Sort each method's rows by condition number
    for m in by_method:
        by_method[m].sort(key=lambda r: float(r["condition"]))

    components = [
        ("fit_error_mean", "fit_error (combined)"),
        ("upper_mode_error_mean", "|sampler_top_mass - gt_top_mass|"),
        ("std_x2_error_mean", "|sampler_std_x2 - gt_std_x2|"),
        ("upper_mode_mass_error_mean", "mode_mass_l1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (col, title) in zip(axes.flat, components):
        for method in PRIMARY_METHODS:
            data = by_method.get(method, [])
            if not data:
                continue
            xs = [float(r["condition"]) for r in data]
            ys = [to_float(r[col]) for r in data]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(title)
        ax.set_xlabel("condition number kappa")
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Study 1 — decomposition of fit_error across kappa (alpha=0.0 / baseline)")
    fig.tight_layout()
    out = os.path.join(out_dir, "diagnose_study1_decomposition.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # Hypothesis-B chart: GT geometry vs kappa
    daps_rows = by_method.get("DAPS", [])
    if daps_rows:
        kappas = [float(r["condition"]) for r in daps_rows]
        gt_var = [to_float(r["gt_var_x2_mean"]) for r in daps_rows]
        # Reconstruct gt_top_mass implicitly: sampler_upper_frac and upper_mode_error
        # We can't perfectly recover sign, but |sampler - gt| with sampler ~ 0.5 in bimodal
        # regime tells us how much GT deviates from 0.5.
        # Plot sampler upper_frac and gt_var_x2 together to show the geometry transition.
        fig, ax1 = plt.subplots(figsize=(9, 5))
        for method in PRIMARY_METHODS:
            data = by_method.get(method, [])
            if not data:
                continue
            xs = [float(r["condition"]) for r in data]
            ys = [to_float(r["upper_frac_mean"]) for r in data]
            ax1.plot(xs, ys, marker="o", label=f"{method} sampler upper_frac")
        ax1.axhline(0.5, color="grey", lw=0.5, ls="--", label="symmetric posterior (0.5)")
        ax1.axhline(1.0, color="grey", lw=0.5, ls=":", label="single-mode (1.0)")
        ax1.set_xscale("log")
        ax1.set_xlabel("condition number kappa")
        ax1.set_ylabel("sampler upper_frac (P[x2 > 0])")
        ax1.set_ylim(0.4, 1.05)
        ax1.legend(loc="upper left", fontsize=7)

        ax2 = ax1.twinx()
        ax2.plot(kappas, gt_var, marker="s", color="black", lw=2,
                 label="GT var(x2) (geometry indicator)")
        ax2.set_yscale("log")
        ax2.set_ylabel("GT var(x2)")
        ax2.legend(loc="lower right", fontsize=7)
        fig.suptitle("Study 1 — GT geometry transition vs sampler tracking")
        fig.tight_layout()
        out = os.path.join(out_dir, "diagnose_study1_gt_geometry.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def study4_decomposition(csv_path: str, out_dir: str) -> None:
    rows = load_csv(csv_path)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    for m in by_method:
        by_method[m].sort(key=lambda r: float(r["langevin_step"]))

    has_median = bool(rows) and "fit_error_median" in rows[0]

    components = [
        ("fit_error", "fit_error (combined)"),
        ("upper_mode_error", "|sampler_top_mass - gt_top_mass|"),
        ("std_x2_error", "|sampler_std_x2 - gt_std_x2|"),
        ("diverged", "fraction of repeats that diverged"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (col, title) in zip(axes.flat, components):
        for method in ("DAPS", "P-DAPS"):
            data = by_method.get(method, [])
            if not data:
                continue
            xs = [float(r["langevin_step"]) for r in data]
            ys = [to_float(r[f"{col}_mean"]) for r in data]
            ax.plot(xs, ys, marker="o", label=f"{method} mean")
            if has_median and col != "diverged":
                ymed = [to_float(r[f"{col}_median"]) for r in data]
                ax.plot(xs, ymed, marker="s", ls="--", alpha=0.7, label=f"{method} median")
        ax.set_xscale("log")
        if col != "diverged":
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(title)
        ax.set_xlabel("langevin_step")
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("Study 4 — decomposition of fit_error across step size (kappa=1000)")
    fig.tight_layout()
    out = os.path.join(out_dir, "diagnose_study4_decomposition.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def study4_per_seed(csv_path: str, out_dir: str) -> None:
    """Plot raw per-seed fit_error vs step to visualise hypothesis D (seed noise)."""
    raw = load_csv(csv_path)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_method[r["method"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, method in zip(axes, ("DAPS", "P-DAPS")):
        rows = by_method.get(method, [])
        # group by repeat index
        per_seed: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
        for r in rows:
            seed = to_float(r.get("repeat", "nan"))
            step = to_float(r["langevin_step"])
            fit = to_float(r["fit_error"])
            per_seed[seed].append((step, fit, to_float(r.get("diverged", "0"))))
        for seed, points in sorted(per_seed.items()):
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker=".", alpha=0.55, lw=1, label=f"seed{int(seed) if not np.isnan(seed) else '?'}")
        # overlay median
        per_step: dict[float, list[float]] = defaultdict(list)
        for r in rows:
            step = to_float(r["langevin_step"])
            fit = to_float(r["fit_error"])
            if not np.isnan(fit):
                per_step[step].append(fit)
        steps = sorted(per_step)
        med = [float(np.median(per_step[s])) for s in steps]
        ax.plot(steps, med, color="black", lw=2.2, label="median", zorder=10)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"{method} — per-seed fit_error")
        ax.set_xlabel("langevin_step")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
    axes[0].set_ylabel("fit_error (raw, per repeat)")
    fig.suptitle("Study 4 — per-seed raw fit_error (hypothesis D: seed noise)")
    fig.tight_layout()
    out = os.path.join(out_dir, "diagnose_study4_per_seed.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def print_numeric_summary(study1_csv: str, study4_csv: str) -> None:
    s1 = load_csv(study1_csv)
    print("\n=== Study 1 (alpha=0.0 / baseline) by kappa ===")
    print(f"{'method':<8}{'kappa':>8}  {'fit_err':>10} {'upper_err':>10} {'std_err':>10} {'gt_var_x2':>12} {'sampler_uf':>12}")
    for r in s1:
        m = r["method"]
        if m.startswith("P-DAPS-warm"):
            continue
        if m == "P-DAPS" and r["alpha"] != "0.0":
            continue
        print(f"{m:<8}{r['condition']:>8}  "
              f"{to_float(r['fit_error_mean']):>10.4f} "
              f"{to_float(r['upper_mode_error_mean']):>10.4f} "
              f"{to_float(r['std_x2_error_mean']):>10.4f} "
              f"{to_float(r['gt_var_x2_mean']):>12.5f} "
              f"{to_float(r['upper_frac_mean']):>12.4f}")

    s4 = load_csv(study4_csv)
    print("\n=== Study 4 (kappa=1000) by langevin_step ===")
    print(f"{'method':<8}{'step':>14}  {'fit_err':>10} {'upper_err':>10} {'std_err':>10} {'diverged':>10}")
    for r in sorted(s4, key=lambda r: (r['method'], float(r['langevin_step']))):
        print(f"{r['method']:<8}{float(r['langevin_step']):>14.3e}  "
              f"{to_float(r['fit_error_mean']):>10.4f} "
              f"{to_float(r['upper_mode_error_mean']):>10.4f} "
              f"{to_float(r['std_x2_error_mean']):>10.4f} "
              f"{to_float(r['diverged_mean']):>10.2f}")


def alternate_metrics_plot(csv_path: str, x_col: str, out_path: str, title: str,
                           filter_alpha: str = "0.0") -> None:
    """Plot fit_error alongside several scale-invariant alternatives so the user
    can see which (if any) is monotone in the sweep axis.

    Derived metrics (all per-row, no resampling needed):
      - fit_error                          : current baseline
      - mean_rmse                          : L2 of posterior-mean error
      - mean_rmse / gt_std_x2              : Mahalanobis-ish (scale-invariant)
      - relative_upper_err                 : upper_mode_error / max(gt_top, 1-gt_top)
      - bures2_x2                          : sqrt(mean_rmse^2 + std_x2_err^2)
                                             (W2 between Gaussian approximations)
      - mode_kl                            : symmetric KL of (gt_top, 1-gt_top)
                                             vs (sampler_top, 1-sampler_top), clipped
    """
    rows = load_csv(csv_path)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if filter_alpha is not None and r.get("alpha") not in (filter_alpha, "baseline"):
            continue
        by_method[r["method"]].append(r)
    for m in by_method:
        by_method[m].sort(key=lambda r: float(r[x_col]))

    methods = [m for m in PRIMARY_METHODS if m in by_method]

    def derived(r):
        mr = to_float(r["mean_rmse_mean"])
        sx = to_float(r["std_x2_error_mean"])
        ue = to_float(r["upper_mode_error_mean"])
        gt_top = to_float(r["gt_top_mass_mean"])
        gt_std = to_float(r["gt_std_x2_mean"])
        sa = to_float(r["upper_frac_mean"])
        eps = 1e-6
        rel_upper = ue / max(gt_top, 1 - gt_top, eps)
        # prefer the in-row bures2_x2_mean if present (computed from raw samples)
        # else fall back to the moments-only approximation
        bures2_csv = to_float(r.get("bures2_x2_mean", ""))
        bures2 = bures2_csv if np.isfinite(bures2_csv) else float(np.sqrt(mr ** 2 + sx ** 2))
        mahal = mr / max(gt_std, eps)
        return {
            "fit_error": to_float(r["fit_error_mean"]),
            "mean_rmse": mr,
            "bures2_x2": bures2,
            "relative_upper_err": rel_upper,
            "data_chi2 (op-space)": to_float(r.get("data_chi2_mean_mean", "")),
            "neg_log_evidence (op-space)": to_float(r.get("neg_log_evidence_mean", "")),
        }

    metrics = list(derived(by_method[methods[0]][0]).keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        for method in methods:
            data = by_method[method]
            xs = [float(r[x_col]) for r in data]
            ys = [derived(r)[metric] for r in data]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(metric)
        ax.set_xlabel(x_col)
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def monotonicity_score(csv_path: str, x_col: str, filter_alpha: str = "0.0") -> None:
    """Print a monotonicity score (Spearman-style: count of inversions) for each
    metric × method combo. Lower is more monotone. Direction-agnostic."""
    rows = load_csv(csv_path)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if filter_alpha is not None and r.get("alpha") not in (filter_alpha, "baseline"):
            continue
        by_method[r["method"]].append(r)
    for m in by_method:
        by_method[m].sort(key=lambda r: float(r[x_col]))

    def metrics_for_row(r):
        mr = to_float(r["mean_rmse_mean"])
        sx = to_float(r["std_x2_error_mean"])
        ue = to_float(r["upper_mode_error_mean"])
        gt_top = to_float(r["gt_top_mass_mean"])
        gt_std = to_float(r["gt_std_x2_mean"])
        eps = 1e-6
        bures2_csv = to_float(r.get("bures2_x2_mean", ""))
        bures2 = bures2_csv if np.isfinite(bures2_csv) else float(np.sqrt(mr ** 2 + sx ** 2))
        return {
            "fit_error": to_float(r["fit_error_mean"]),
            "mean_rmse": mr,
            "bures2_x2": bures2,
            "rel_upper_err": ue / max(gt_top, 1 - gt_top, eps),
            "data_chi2": to_float(r.get("data_chi2_mean_mean", "")),
            "neg_log_evidence": to_float(r.get("neg_log_evidence_mean", "")),
        }

    def inversions(seq):
        seq = [v for v in seq if np.isfinite(v)]
        n = len(seq)
        if n < 2:
            return 0, 0
        inv = sum(1 for i in range(n) for j in range(i + 1, n) if seq[j] < seq[i])
        # normalize to [0, 1]: 0 = perfectly monotone non-decreasing
        return inv, n * (n - 1) / 2

    print(f"\nMonotonicity score (lower = more monotone non-decreasing in {x_col}):")
    methods = [m for m in PRIMARY_METHODS if m in by_method]
    metric_names = list(metrics_for_row(by_method[methods[0]][0]).keys())
    print(f"{'metric':<26}" + "".join(f"{m:>14}" for m in methods))
    for metric in metric_names:
        line = f"{metric:<26}"
        for method in methods:
            seq = [metrics_for_row(r)[metric] for r in by_method[method]]
            inv, total = inversions(seq)
            frac = inv / total if total else 0.0
            line += f"{inv}/{int(total)} ({frac:.2f}) "
        print(line)


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = latest_formal_dir()
    print(f"diagnosing {results_dir}")
    s1 = os.path.join(results_dir, "study1_cond_alpha.csv")
    s4 = os.path.join(results_dir, "study4_stability.csv")
    s4_raw = os.path.join(results_dir, "study4_stability_raw.csv")
    if os.path.exists(s1):
        study1_decomposition(s1, results_dir)
    if os.path.exists(s4):
        study4_decomposition(s4, results_dir)
    if os.path.exists(s4_raw):
        study4_per_seed(s4_raw, results_dir)
    if os.path.exists(s1) and os.path.exists(s4):
        print_numeric_summary(s1, s4)
    if os.path.exists(s1):
        alternate_metrics_plot(
            s1, "condition",
            os.path.join(results_dir, "diagnose_alt_metrics_study1.png"),
            "Study 1 — alternate metrics for monotonicity in kappa",
        )
        monotonicity_score(s1, "condition")
    # Toy D variants if present
    sd1 = os.path.join(results_dir, "studyD1_kappa_alpha.csv")
    if os.path.exists(sd1):
        alternate_metrics_plot(
            sd1, "kappa",
            os.path.join(results_dir, "diagnose_alt_metrics_studyD1.png"),
            "Study D-1 — alternate metrics for monotonicity in kappa (Toy D)",
        )
        monotonicity_score(sd1, "kappa")


if __name__ == "__main__":
    main()
