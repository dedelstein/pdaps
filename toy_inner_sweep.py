"""
Toy 2D inner_sigma_max sweep.

Validates the MRI finding that gating P-DAPS's inner Langevin correction
on sigma > inner_sigma_max improves quality and runtime. Runs across the
threshold range on a single toy scenario, reports fit_error vs threshold.

Usage:
    ./.venv/bin/python3 toy_inner_sweep.py \
        --scenario toy_b_stiffness --repeats 5 --nb 1500 \
        --out-dir results/toy_inner_sweep
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

import toy_2d as toy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="toy_b_stiffness")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nb", type=int, default=1500)
    parser.add_argument("--gt-samples", type=int, default=100_000)
    parser.add_argument("--warm-fraction", type=float, default=0.3)
    parser.add_argument("--out-dir", default="results/toy_inner_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toy.configure_scenario(args.scenario)
    rp = dict(toy.CURRENT_SCENARIO["run_params"])
    gt_samples = toy.sample_ground_truth(n_samples=args.gt_samples)
    gt_summary = toy.summarize_samples(gt_samples)

    sigma_max = rp["sigma_max"]
    sigma_min = rp["sigma_min"]
    N_inner = rp["pdaps_langevin_steps"]
    gamma = rp["pdaps_langevin_step_size"]

    theory_thr = 1.0 / math.sqrt(N_inner * gamma) if N_inner * gamma > 0 else float("inf")

    sweep_values = sorted(set(
        [sigma_min * 1.01, theory_thr * 0.5, theory_thr, theory_thr * 2.0,
         sigma_max * 0.25, sigma_max * 0.5, sigma_max, 1e9]
    ))
    sweep_values = [v for v in sweep_values if v >= sigma_min]

    print(f"=== inner_sigma_max sweep on {args.scenario} ===")
    print(f"  sigma_max={sigma_max}  sigma_min={sigma_min}  N_inner={N_inner}  gamma={gamma}")
    print(f"  theory threshold 1/sqrt(N*gamma) = {theory_thr:.3f}")
    print(f"  sweep over inner_sigma_max: {[f'{v:.3g}' for v in sweep_values]}")
    print()

    rows = []
    for variant_name, run_fn in [
        ("P-DAPS", lambda inner_max, seed: toy.run_pdaps(
            nb=args.nb, N=rp["N"], sigma_max=sigma_max, sigma_min=sigma_min,
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=N_inner,
            lgvd_step_size=gamma, inner_sigma_max=inner_max,
        )),
        ("P-DAPS-warm", lambda inner_max, seed: toy.run_pdaps_warm(
            nb=args.nb, N=rp["N"], sigma_max=sigma_max, sigma_min=sigma_min,
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=N_inner,
            lgvd_step_size=gamma, warm_fraction=args.warm_fraction,
            inner_sigma_max=inner_max,
        )),
    ]:
        for inner_max in sweep_values:
            errs, times, divs = [], [], 0
            for r in range(args.repeats):
                torch.manual_seed(42 + r)
                np.random.seed(42 + r)
                t0 = time.perf_counter()
                result = run_fn(inner_max, 42 + r)
                wall = time.perf_counter() - t0
                if result.get("diverged_at") is not None:
                    divs += 1
                    continue
                fit = toy.compare_to_truth(result["final"], gt_summary)
                errs.append(fit["fit_error"])
                times.append(wall)
            ok = [e for e in errs if np.isfinite(e)]
            row = {
                "method": variant_name,
                "inner_sigma_max": inner_max,
                "fit_error_mean": float(np.mean(ok)) if ok else float("nan"),
                "fit_error_std": float(np.std(ok)) if len(ok) > 1 else 0.0,
                "runtime_s_mean": float(np.mean(times)) if times else float("nan"),
                "n_ok": len(ok),
                "n_diverged": divs,
                "n": args.repeats,
            }
            rows.append(row)
            print(f"  {variant_name:13s} inner_max={inner_max:7.3g}  "
                  f"fit_error={row['fit_error_mean']:.4f}  "
                  f"runtime={row['runtime_s_mean']:.2f}s  "
                  f"div={divs}/{args.repeats}")

    with open(out_dir / "inner_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(11, 4))
        for variant in ["P-DAPS", "P-DAPS-warm"]:
            sub = [r for r in rows if r["method"] == variant]
            xs = [r["inner_sigma_max"] for r in sub]
            ys = [r["fit_error_mean"] for r in sub]
            ts = [r["runtime_s_mean"] for r in sub]
            ax_err.plot(xs, ys, "o-", label=variant)
            ax_time.plot(xs, ts, "o-", label=variant)
        for ax in (ax_err, ax_time):
            ax.set_xscale("log")
            ax.set_xlabel("inner_sigma_max")
            ax.axvline(theory_thr, color="gray", ls="--", alpha=0.5,
                       label=f"theory 1/√(Nγ)={theory_thr:.3g}")
            ax.axvline(sigma_max, color="red", ls=":", alpha=0.4,
                       label=f"σ_max={sigma_max}")
            ax.legend(fontsize=8)
        ax_err.set_ylabel("fit_error_mean")
        ax_time.set_ylabel("runtime (s)")
        fig.suptitle(f"P-DAPS inner-correction gating sweep — {args.scenario}")
        fig.tight_layout()
        fig.savefig(out_dir / "inner_sweep.png", dpi=120)
        print(f"\nWrote {out_dir / 'inner_sweep.png'}")
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nWrote {out_dir / 'inner_sweep.csv'}")


if __name__ == "__main__":
    main()
