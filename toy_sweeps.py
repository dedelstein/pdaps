#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import toy_2d as toy


DTYPE = torch.float64
DEVICE = torch.device("cpu")

CONDITION_NUMBERS = (1, 10, 100, 1000, 10000)
SDE_VARIANTS = ("VE", "VP")
WARM_FRACTIONS_5 = (0.0, 0.25, 0.5, 0.75, 1.0)
WARM_FRACTIONS_3 = (0.0, 0.5, 1.0)
WARM_FRACTIONS_11 = tuple(np.linspace(0.0, 1.0, 11))
P_DAPS_WARM_ALPHA_SUITE = tuple(round(float(alpha), 1) for alpha in np.linspace(0.1, 1.0, 10))
SIGMA_NOISES = (0.01, 0.05, 0.1, 0.2)
STABILITY_STEPS = tuple(np.logspace(-6, 0, 13))
# Eighth-decade fine grid spanning the suspected DAPS breakdown band. Used by
# run_daps_breakdown_bracket only; cheaper than re-running full Study 4.
DAPS_BREAKDOWN_STEPS = tuple(np.logspace(-5, -3, 17))
# Canonical scenarios whose daps_langevin_lr defaults must be bracketed against
# the measured breakdown. Order is the order the breakdown sweep visits them.
DAPS_BREAKDOWN_SCENARIOS = (
    "toy_a_mode_recovery",
    "toy_b_stiffness",
    "toy_c_score_bias",
    "toy_d_high_d_stiffness",
)
SCORE_BIAS_LEVELS = {
    "none": {"bias": [0.0, 0.0], "sigma_gate": 0.24, "sharpness": 16.0},
    "mild": {"bias": [0.24, 0.0], "sigma_gate": 0.24, "sharpness": 16.0},
    "harsh": {"bias": [0.42, 0.0], "sigma_gate": 0.18, "sharpness": 20.0},
}
CONVERGENCE_THRESHOLD = 0.05
VALIDATION_SCENARIOS = ("toy_a_mode_recovery", "toy_b_stiffness")
P_DAPS_WARM_ALPHA_PREFIX = "P-DAPS-warm alpha="
P_DAPS_ADAPTIVE = "P-DAPS-adaptive"


def warm_alpha_method(alpha):
    return f"{P_DAPS_WARM_ALPHA_PREFIX}{float(alpha):.1f}"


def is_warm_alpha_method(method):
    return method == "P-DAPS-warm" or str(method).startswith(P_DAPS_WARM_ALPHA_PREFIX)


def is_adaptive_method(method):
    return method == P_DAPS_ADAPTIVE


def label_warm_alpha_rows(rows, alpha):
    for row in rows:
        if row["method"] == "P-DAPS-warm":
            row["method"] = warm_alpha_method(alpha)
        row["alpha"] = float(alpha)
    return rows


def label_pdaps_zero_alpha(rows):
    for row in rows:
        row["alpha"] = 0.0 if row["method"] == "P-DAPS" else "baseline"
    return rows


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


@contextmanager
def pushd(path):
    old_cwd = os.getcwd()
    ensure_dir(path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def sde_context(sde):
    prev = toy.SDE
    toy.SDE = sde
    try:
        yield
    finally:
        toy.SDE = prev


def save_to_csv(filepath, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_grid(**axes):
    names = tuple(axes.keys())
    values = tuple(axes[name] for name in names)
    for combo in itertools.product(*values):
        yield dict(zip(names, combo))


VALIDATION_GRIDS = {
    "DPS": list(iter_grid(dps_guidance_scale=(1.4, 2.0, 2.6, 3.2))),
    "DAPS": list(iter_grid(daps_langevin_lr=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
                           daps_langevin_steps=(40, 70, 120))),
    "pULA": list(iter_grid(pula_step_size=(0.25, 0.5, 0.75, 1.0),
                           pula_nb_langevin=(3, 5, 8))),
    "P-DAPS": list(iter_grid(pdaps_langevin_step_size=(0.25, 0.5, 0.75, 1.0),
                             pdaps_langevin_steps=(8, 16, 24))),
    "P-DAPS-warm": list(iter_grid(pdaps_langevin_step_size=(0.25, 0.5, 0.75, 1.0),
                                  pdaps_langevin_steps=(8, 16, 24),
                                  pdaps_warm_fraction=(0.0, 0.25, 0.5))),
}


def budget_steps_for(N, stride=10):
    steps = set(range(stride, N + 1, stride))
    steps.add(N)
    return tuple(sorted(steps))


def nfe_per_outer_step(method, rp):
    if method == "DPS":
        return 1
    if method == "DAPS":
        return rp["daps_ode_steps"]
    if method == "pULA":
        return rp["pula_nb_langevin"]
    if method == "P-DAPS" or is_warm_alpha_method(method) or is_adaptive_method(method):
        return rp["pdaps_ode_steps"]
    return 1


def nfe_to_converge(method, result, gt_summary, rp, threshold=CONVERGENCE_THRESHOLD):
    per_step = nfe_per_outer_step(method, rp)
    for step in sorted(result["progress"]):
        fit = toy.compare_to_truth(result["progress"][step]["samples"], gt_summary)["fit_error"]
        if np.isfinite(fit) and fit <= threshold:
            return int(step * per_step)
    return np.nan


def add_nfe_metrics(row, method, result, gt_summary, rp):
    per_step = nfe_per_outer_step(method, rp)
    row["nfe_total"] = int(rp["N"] * per_step)
    row["nfe_per_outer_step"] = int(per_step)
    row["nfe_to_converge"] = nfe_to_converge(method, result, gt_summary, rp)
    return row


def aggregate_rows(rows, key_fields):
    key_fields = tuple(key_fields)
    groups = {}
    order = []
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    aggregated = []
    for key in order:
        group = groups[key]
        out = {field: value for field, value in zip(key_fields, key)}
        out["n_runs"] = len(group)
        for field in group[0]:
            if field in key_fields or field == "method":
                continue
            values = []
            for row in group:
                value = row.get(field)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))
            if len(values) != len(group):
                continue
            values = np.array(values, dtype=float)
            if np.all(np.isnan(values)):
                out[f"{field}_mean"] = np.nan
                out[f"{field}_std"] = np.nan
                out[f"{field}_median"] = np.nan
                out[f"{field}_iqr"] = np.nan
                continue
            out[f"{field}_mean"] = float(np.nanmean(values))
            out[f"{field}_std"] = float(np.nanstd(values))
            out[f"{field}_median"] = float(np.nanmedian(values))
            q75, q25 = np.nanpercentile(values, [75, 25])
            out[f"{field}_iqr"] = float(q75 - q25)
        aggregated.append(out)
    return aggregated


def run_methods(nb, rp, budget_steps, daps_lr, pdaps_step, warm_fraction,
                pula_step=None,
                adversarial_init=False, methods=None):
    methods = set(methods or ("DPS", "DAPS", "pULA", "P-DAPS", "P-DAPS-warm"))
    results = {}
    if "DPS" in methods:
        results["DPS"] = toy.run_dps(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            guidance_scale=rp["dps_guidance_scale"], budget_steps=budget_steps,
            adversarial_init=adversarial_init,
        )
    if "DAPS" in methods:
        results["DAPS"] = toy.run_daps(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["daps_ode_steps"], lgvd_steps=rp["daps_langevin_steps"],
            lgvd_lr=daps_lr, budget_steps=budget_steps, adversarial_init=adversarial_init,
        )
    if "pULA" in methods:
        if pula_step is None:
            pula_step = rp["pula_step_size"]
        results["pULA"] = toy.run_pula(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            step_size=pula_step, nb_langevin=rp["pula_nb_langevin"],
            budget_steps=budget_steps, adversarial_init=adversarial_init,
        )
    if "P-DAPS" in methods:
        results["P-DAPS"] = toy.run_pdaps(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=pdaps_step, budget_steps=budget_steps, adversarial_init=adversarial_init,
        )
    if "P-DAPS-warm" in methods:
        results["P-DAPS-warm"] = toy.run_pdaps_warm(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=pdaps_step, warm_fraction=warm_fraction,
            budget_steps=budget_steps, adversarial_init=adversarial_init,
        )
    if P_DAPS_ADAPTIVE in methods:
        results[P_DAPS_ADAPTIVE] = toy.run_pdaps_adaptive(
            nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=pdaps_step, warm_fraction=warm_fraction,
            budget_steps=budget_steps, adversarial_init=adversarial_init,
        )
    return results


def apply_validated_params(rp, scenario_key, best_params_by_scenario=None, methods=None):
    rp = dict(rp)
    if not best_params_by_scenario:
        return rp
    scenario_params = best_params_by_scenario.get(scenario_key, {})
    for method in methods or scenario_params:
        for key, value in scenario_params.get(method, {}).items():
            if value != "":
                rp[key] = value
    return rp


def run_study_point(scenario_key, repeats=5, base_seed=42, nb=1000, gt_samples=100_000,
                    a_diag_override=None, sigma_noise_override=None, score_bias_override=None,
                    kappa_override=None,
                    lgvd_lr_override=None, lgvd_step_size_override=None,
                    pula_step_size_override=None,
                    warm_fraction_override=None, adversarial_init=False, methods=None,
                    convergence_threshold=CONVERGENCE_THRESHOLD, best_params_by_scenario=None):
    del convergence_threshold  # Kept in the signature so callers document the study threshold.
    rows = []
    for rep in range(repeats):
        toy.set_global_seed(base_seed + rep)
        toy.configure_scenario(
            scenario_key,
            quiet=True,
            a_diag_override=a_diag_override,
            sigma_noise_override=sigma_noise_override,
            score_bias_override=score_bias_override,
            kappa_override=kappa_override,
        )

        rp = apply_validated_params(
            toy.CURRENT_SCENARIO["run_params"],
            scenario_key,
            best_params_by_scenario=best_params_by_scenario,
            methods=methods,
        )
        budgets = budget_steps_for(rp["N"])
        gt = toy.sample_ground_truth(n_samples=gt_samples)
        gt_summary = toy.summarize_samples(gt)

        daps_lr = lgvd_lr_override if lgvd_lr_override is not None else rp["daps_langevin_lr"]
        pdaps_step = lgvd_step_size_override if lgvd_step_size_override is not None else rp["pdaps_langevin_step_size"]
        pula_step = pula_step_size_override if pula_step_size_override is not None else rp["pula_step_size"]
        warm_f = warm_fraction_override if warm_fraction_override is not None else rp.get("pdaps_warm_fraction", 0.5)

        results = run_methods(
            nb, rp, budgets, daps_lr, pdaps_step, warm_f,
            pula_step=pula_step,
            adversarial_init=adversarial_init, methods=methods,
        )
        for method, result in results.items():
            row = toy._summarize_method_result(method, result, gt_summary)
            add_nfe_metrics(row, method, result, gt_summary, rp)
            row["repeat"] = rep
            rows.append(row)
    return rows


def run_sweep_point(**kwargs):
    rows = run_study_point(**kwargs)
    aggregated = aggregate_rows(rows, key_fields=("method",))
    return {row["method"]: row for row in aggregated}


def run_validation_candidate(scenario_key, method, params, repeats, nb, gt_samples, base_seed=42):
    rows = []
    for rep in range(repeats):
        toy.set_global_seed(base_seed + rep)
        toy.configure_scenario(scenario_key, quiet=True)
        rp = dict(toy.CURRENT_SCENARIO["run_params"])
        rp.update(params)
        budgets = budget_steps_for(rp["N"])
        gt = toy.sample_ground_truth(n_samples=gt_samples)
        gt_summary = toy.summarize_samples(gt)

        if method == "DPS":
            result = toy.run_dps(
                nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                guidance_scale=rp["dps_guidance_scale"], budget_steps=budgets,
            )
        elif method == "DAPS":
            result = toy.run_daps(
                nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                ode_steps=rp["daps_ode_steps"], lgvd_steps=rp["daps_langevin_steps"],
                lgvd_lr=rp["daps_langevin_lr"], budget_steps=budgets,
            )
        elif method == "pULA":
            result = toy.run_pula(
                nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                step_size=rp["pula_step_size"], nb_langevin=rp["pula_nb_langevin"],
                budget_steps=budgets,
            )
        elif method == "P-DAPS":
            result = toy.run_pdaps(
                nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
                lgvd_step_size=rp["pdaps_langevin_step_size"], budget_steps=budgets,
            )
        elif method == "P-DAPS-warm":
            result = toy.run_pdaps_warm(
                nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
                lgvd_step_size=rp["pdaps_langevin_step_size"],
                warm_fraction=rp["pdaps_warm_fraction"], budget_steps=budgets,
            )
        else:
            raise ValueError(f"Unknown validation method: {method}")

        row = toy._summarize_method_result(method, result, gt_summary)
        add_nfe_metrics(row, method, result, gt_summary, rp)
        row["scenario"] = scenario_key
        row["repeat"] = rep
        for key, value in params.items():
            row[key] = value
        rows.append(row)
    return rows


def run_validation_tuning(repeats, nb, gt_samples, out_dir):
    print("\n--- Validation Hyperparameter Tuning ---")
    raw_rows = []
    best_rows = []
    for scenario in VALIDATION_SCENARIOS:
        for method, grid in VALIDATION_GRIDS.items():
            print(f"Validation grid: {scenario} / {method} / {len(grid)} candidates")
            for params in tqdm(grid):
                raw_rows.extend(run_validation_candidate(
                    scenario_key=scenario,
                    method=method,
                    params=params,
                    repeats=repeats,
                    nb=nb,
                    gt_samples=gt_samples,
                ))

    key_fields = ("scenario", "method")
    param_fields = sorted({key for row in raw_rows for key in row if key.endswith(("_scale", "_lr", "_steps", "_step_size", "_langevin")) or key == "pdaps_warm_fraction"})
    for row in raw_rows:
        for field in param_fields:
            row.setdefault(field, "")
    candidate_key_fields = tuple(list(key_fields) + param_fields)
    agg = aggregate_rows(raw_rows, key_fields=candidate_key_fields)
    for scenario in VALIDATION_SCENARIOS:
        for method in VALIDATION_GRIDS:
            candidates = [row for row in agg if row["scenario"] == scenario and row["method"] == method]
            finite = [row for row in candidates if np.isfinite(row.get("fit_error_mean", np.nan))]
            pool = finite if finite else candidates
            if pool:
                best_rows.append(min(pool, key=lambda row: row.get("fit_error_mean", np.inf)))

    save_to_csv(os.path.join(out_dir, "validation_tuning_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "validation_tuning.csv"), agg)
    save_to_csv(os.path.join(out_dir, "validation_best_by_method.csv"), best_rows)

    for scenario in VALIDATION_SCENARIOS:
        rows = [row for row in best_rows if row["scenario"] == scenario]
        plt.figure(figsize=(8, 5.5))
        plt.bar([row["method"] for row in rows], [row["fit_error_mean"] for row in rows])
        plt.ylabel("best validation fit_error_mean")
        plt.title(f"Validation Best by Method: {scenario}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"validation_best_{scenario}.png"), dpi=200)
        plt.close()
    return best_params_map(best_rows)


def best_params_map(best_rows):
    param_fields = sorted({
        key
        for row in best_rows
        for key, value in row.items()
        if key in {
            "dps_guidance_scale",
            "daps_langevin_lr",
            "daps_langevin_steps",
            "pula_step_size",
            "pula_nb_langevin",
            "pdaps_langevin_step_size",
            "pdaps_langevin_steps",
            "pdaps_warm_fraction",
        } and value != ""
    })
    by_scenario = {}
    for row in best_rows:
        scenario = row["scenario"]
        method = row["method"]
        by_scenario.setdefault(scenario, {})[method] = {
            field: row[field]
            for field in param_fields
            if field in row and row[field] != ""
        }
    return by_scenario


def plot_sweep(x_values, sweep_results, x_label, title, filename, log_x=False, metric="fit_error_mean"):
    methods = sweep_results[0].keys()
    plt.figure(figsize=(10, 6))
    for method in methods:
        y = [res[method][metric] for res in sweep_results]
        yerr = [res[method].get(metric.replace("_mean", "_std"), 0.0) for res in sweep_results]
        plt.errorbar(x_values, y, yerr=yerr, label=method, fmt="-o", capsize=4)
    if log_x:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_heatmap(rows, x_field, y_field, value_field, title, filename, log_x=False):
    xs = sorted({row[x_field] for row in rows})
    ys = sorted({row[y_field] for row in rows})
    grid = np.full((len(ys), len(xs)), np.nan)
    for row in rows:
        grid[ys.index(row[y_field]), xs.index(row[x_field])] = row[value_field]

    plt.figure(figsize=(8, 5.5))
    im = plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(im, label=value_field)
    plt.xticks(range(len(xs)), [f"{x:g}" if isinstance(x, float) else str(x) for x in xs])
    plt.yticks(range(len(ys)), [f"{y:g}" if isinstance(y, float) else str(y) for y in ys])
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    if log_x:
        plt.xlabel(f"{x_field} (log-spaced)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


METRIC_PANEL_FIELDS = (
    ("fit_error_mean",     "fit_error (posterior-space)"),
    ("mean_rmse_mean",     "mean_rmse (posterior-space)"),
    ("bures2_x2_mean",     "bures2_x2 (W2 of moments)"),
    ("data_chi2_mean_mean","data_chi2 (operator-space)"),
)


def plot_metric_panels(rows, x_field, methods, filename, title, log_x=True,
                       extra_field=None):
    """Four-panel comparison of fit_error, mean_rmse, bures2_x2, data_chi2 vs
    the sweep axis. Lets us read posterior-space and operator-space side by side
    without committing to a single headline metric."""
    panels = list(METRIC_PANEL_FIELDS)
    if extra_field is not None:
        panels = [extra_field, *panels[:3]]  # swap one panel for an extra view
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (col, label) in zip(axes.flat, panels):
        for method in methods:
            mrows = [row for row in rows if row["method"] == method]
            if not mrows:
                continue
            mrows = sorted(mrows, key=lambda r: r[x_field])
            xs = [row[x_field] for row in mrows]
            ys = [row.get(col, np.nan) for row in mrows]
            ax.plot(xs, ys, marker="o", label=method,
                    color=METHOD_COLORS_TOY.get(method))
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel(x_field)
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


METHOD_COLORS_TOY = {
    "DPS": "#c8553d",
    "DAPS": "#7f3c8d",
    "pULA": "#2c7fb8",
    "P-DAPS": "#1b9e77",
    "P-DAPS-warm": "#8c6d1f",
    P_DAPS_ADAPTIVE: "#e66101",
}


def plot_pareto(rows, filename, title="Accuracy vs NFE Pareto"):
    plt.figure(figsize=(8, 5.5))
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        x = [row["nfe_total_mean"] for row in method_rows]
        y = [row["fit_error_mean"] for row in method_rows]
        plt.scatter(x, y, label=method, alpha=0.75)
    plt.xlabel("total NFE")
    plt.ylabel("fit_error_mean")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_3d_posterior_samples(scenario, nb, gt_samples, out_dir, grid_size=120):
    toy.set_global_seed(42)
    toy.configure_scenario(scenario, quiet=True)
    rp = toy.CURRENT_SCENARIO["run_params"]
    xx, yy, posterior = toy.density_on_grid(toy.posterior_log_prob, grid_size=grid_size)
    gt = toy.sample_ground_truth(n_samples=min(gt_samples, 10_000))
    result = toy.run_pdaps_warm(
        nb=nb,
        N=rp["N"],
        sigma_max=rp["sigma_max"],
        sigma_min=rp["sigma_min"],
        ode_steps=rp["pdaps_ode_steps"],
        lgvd_steps=rp["pdaps_langevin_steps"],
        lgvd_step_size=rp["pdaps_langevin_step_size"],
        warm_fraction=rp["pdaps_warm_fraction"],
        budget_steps=(rp["N"],),
    )
    samples, _ = toy.samples_in_view(result["final"])

    z_floor = -0.08 * float(np.nanmax(posterior))
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, posterior, cmap="viridis", linewidth=0, antialiased=True, alpha=0.88)

    if len(gt):
        gt_plot = gt[::max(1, len(gt) // 1200)]
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], zs=z_floor, zdir="z", s=5, alpha=0.16,
                   c="#111111", label="true posterior")
    if len(samples):
        sample_plot = samples[::max(1, len(samples) // 1200)]
        ax.scatter(sample_plot[:, 0], sample_plot[:, 1], zs=z_floor, zdir="z", s=8, alpha=0.45,
                   c="#d95f02", label="P-DAPS-warm")

    centers = toy.CLUSTER_CENTERS.detach().cpu().numpy()
    ax.scatter(centers[:, 0], centers[:, 1], zs=z_floor, zdir="z", s=55, marker="+",
               c="#111111", linewidths=1.2, label="prior centers")
    ax.scatter([toy.X_TRUE[0].item()], [toy.X_TRUE[1].item()], zs=z_floor, zdir="z",
               s=70, marker="x", c="#e41a1c", linewidths=2.0, label="x_true")

    ax.set_title(f"{toy.CURRENT_SCENARIO['name']}: exact posterior surface", pad=18)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("posterior density")
    ax.set_xlim(-toy.LIM, toy.LIM)
    ax.set_ylim(-toy.LIM, toy.LIM)
    ax.view_init(elev=34, azim=-54)
    ax.legend(loc="upper left")
    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"{scenario}_posterior_3d.png"), dpi=220)
    plt.close(fig)


def plot_3d_pooled_posterior_samples(scenario, pooled_gt, pooled_samples, out_dir, repeats, grid_size=120):
    toy.configure_scenario(scenario, quiet=True)
    xx, yy, posterior = toy.density_on_grid(toy.posterior_log_prob, grid_size=grid_size)
    samples, _ = toy.samples_in_view(pooled_samples)

    z_floor = -0.08 * float(np.nanmax(posterior))
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, posterior, cmap="viridis", linewidth=0, antialiased=True, alpha=0.88)

    if len(pooled_gt):
        gt_plot = pooled_gt[::max(1, len(pooled_gt) // 1500)]
        ax.scatter(gt_plot[:, 0], gt_plot[:, 1], zs=z_floor, zdir="z", s=5, alpha=0.14,
                   c="#111111", label="true posterior, pooled")
    if len(samples):
        sample_plot = samples[::max(1, len(samples) // 1500)]
        ax.scatter(sample_plot[:, 0], sample_plot[:, 1], zs=z_floor, zdir="z", s=8, alpha=0.42,
                   c="#d95f02", label="P-DAPS-warm, pooled")

    centers = toy.CLUSTER_CENTERS.detach().cpu().numpy()
    ax.scatter(centers[:, 0], centers[:, 1], zs=z_floor, zdir="z", s=55, marker="+",
               c="#111111", linewidths=1.2, label="prior centers")
    ax.scatter([toy.X_TRUE[0].item()], [toy.X_TRUE[1].item()], zs=z_floor, zdir="z",
               s=70, marker="x", c="#e41a1c", linewidths=2.0, label="x_true")

    ax.set_title(f"{toy.CURRENT_SCENARIO['name']}: pooled posterior view over {repeats} repeats", pad=18)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("posterior density")
    ax.set_xlim(-toy.LIM, toy.LIM)
    ax.set_ylim(-toy.LIM, toy.LIM)
    ax.view_init(elev=34, azim=-54)
    ax.legend(loc="upper left")
    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, f"{scenario}_posterior_3d_repeat_pooled.png"), dpi=220)
    plt.close(fig)


def run_repeat_averaged_qualitative(scenario, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print(f"\n--- Repeat-Averaged Qualitative: {scenario} ---")
    toy.configure_scenario(scenario, quiet=True)
    methods = ("DPS", "DAPS", "pULA", "P-DAPS", "P-DAPS-warm")
    pooled_gt = []
    pooled_samples = {method: [] for method in methods}
    raw_rows = []

    for rep in range(repeats):
        toy.set_global_seed(42 + rep)
        toy.configure_scenario(scenario, quiet=True)
        rp = apply_validated_params(
            toy.CURRENT_SCENARIO["run_params"],
            scenario,
            best_params_by_scenario=best_params_by_scenario,
            methods=methods,
        )
        gt = toy.sample_ground_truth(n_samples=gt_samples)
        gt_summary = toy.summarize_samples(gt)
        pooled_gt.append(gt)
        results = run_methods(
            nb=nb,
            rp=rp,
            budget_steps=(rp["N"],),
            daps_lr=rp["daps_langevin_lr"],
            pdaps_step=rp["pdaps_langevin_step_size"],
            warm_fraction=rp["pdaps_warm_fraction"],
            methods=methods,
        )
        for method, result in results.items():
            pooled_samples[method].append(result["final"])
            row = toy._summarize_method_result(method, result, gt_summary)
            add_nfe_metrics(row, method, result, gt_summary, rp)
            row["scenario"] = scenario
            row["repeat"] = rep
            raw_rows.append(row)

    pooled_gt = np.concatenate(pooled_gt, axis=0)
    pooled_samples = {method: np.concatenate(samples, axis=0) for method, samples in pooled_samples.items()}
    agg = aggregate_rows(raw_rows, key_fields=("scenario", "method"))
    ensure_dir(out_dir)
    save_to_csv(os.path.join(out_dir, f"{scenario}_repeat_averaged_metrics_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, f"{scenario}_repeat_averaged_metrics.csv"), agg)

    toy.configure_scenario(scenario, quiet=True)
    xx, yy, posterior_density = toy.density_on_grid(toy.posterior_log_prob, grid_size=250)
    gt_contour = (xx, yy, posterior_density)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    axes = axes.ravel()
    toy.plot_sample_density(axes[0], pooled_gt, title=f"True posterior pooled, n={len(pooled_gt)}")
    for idx, method in enumerate(methods, start=1):
        in_view, frac = toy.samples_in_view(pooled_samples[method])
        row = next(row for row in agg if row["method"] == method)
        if len(in_view) >= 40 and frac >= 0.10:
            toy.plot_sample_density(
                axes[idx],
                in_view,
                title=f"{method}\nfit={row['fit_error_mean']:.3f}, in-view={row['in_view_mean']:.2f}",
                gt_contour=gt_contour,
            )
        else:
            toy._style_density_axis(axes[idx])
            axes[idx].set_title(f"{method}\noff-window, in-view={frac:.2f}")
    fig.suptitle(f"{toy.CURRENT_SCENARIO['name']}: repeat-pooled final samples over {repeats} repeats", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{scenario}_repeat_averaged_algorithms.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_3d_pooled_posterior_samples(
        scenario,
        pooled_gt=pooled_gt,
        pooled_samples=pooled_samples["P-DAPS-warm"],
        out_dir=out_dir,
        repeats=repeats,
    )


def run_study1_cond_alpha(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 1: Conditioning x Alpha Grid ---")
    raw_rows = []
    for condition in tqdm(CONDITION_NUMBERS):
        baseline_rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            a_diag_override=[1.0, 1.0 / condition],
            methods=("DPS", "DAPS", "pULA", "P-DAPS"),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_pdaps_zero_alpha(baseline_rows)
        for row in baseline_rows:
            row["condition"] = condition
            raw_rows.append(row)

    for point in tqdm(list(iter_grid(condition=CONDITION_NUMBERS, alpha=P_DAPS_WARM_ALPHA_SUITE))):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            a_diag_override=[1.0, 1.0 / point["condition"]],
            warm_fraction_override=point["alpha"],
            methods=("P-DAPS-warm",),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_warm_alpha_rows(rows, point["alpha"])
        for row in rows:
            row.update(point)
            raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("condition", "alpha", "method"))
    save_to_csv(os.path.join(out_dir, "study1_cond_alpha_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study1_cond_alpha.csv"), agg)
    warm_rows = [row for row in agg if is_warm_alpha_method(row["method"]) or row["method"] == "P-DAPS"]
    plot_heatmap(warm_rows, "condition", "alpha", "fit_error_mean",
                 "Study 1: P-DAPS-warm Fit Error", os.path.join(out_dir, "study1_heatmap_fit_error.png"), log_x=True)
    plot_heatmap(warm_rows, "condition", "alpha", "nfe_to_converge_mean",
                 "Study 1: P-DAPS-warm NFE to Converge", os.path.join(out_dir, "study1_heatmap_nfe_to_converge.png"), log_x=True)
    plot_pareto(agg, os.path.join(out_dir, "study1_pareto_fit_error_nfe.png"),
                title="Study 1: Fit Error vs Total NFE")
    # Posterior-space vs operator-space comparison at alpha=0.0 / baseline
    panel_rows = [r for r in agg if r.get("alpha") in ("0.0", "baseline")]
    plot_metric_panels(
        panel_rows, "condition", ("DPS", "DAPS", "pULA", "P-DAPS"),
        os.path.join(out_dir, "study1_metric_panels.png"),
        "Study 1 — Stiffness Signature: posterior-space vs operator-space",
    )


def run_study2_mode_phase(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 2: Alpha x Score-Bias Phase Transition ---")
    raw_rows = []
    for score_bias in tqdm(list(SCORE_BIAS_LEVELS.keys())):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            score_bias_override=SCORE_BIAS_LEVELS[score_bias],
            adversarial_init=True,
            methods=("DPS", "DAPS", "pULA", "P-DAPS"),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_pdaps_zero_alpha(rows)
        for row in rows:
            row["score_bias"] = score_bias
            row["adversarial_init"] = True
            raw_rows.append(row)

    for point in tqdm(list(iter_grid(alpha=P_DAPS_WARM_ALPHA_SUITE, score_bias=list(SCORE_BIAS_LEVELS.keys())))):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            warm_fraction_override=point["alpha"],
            score_bias_override=SCORE_BIAS_LEVELS[point["score_bias"]],
            adversarial_init=True,
            methods=("P-DAPS-warm",),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_warm_alpha_rows(rows, point["alpha"])
        for row in rows:
            row.update(point)
            row["adversarial_init"] = True
            raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("score_bias", "alpha", "method"))
    save_to_csv(os.path.join(out_dir, "study2_alpha_score_bias_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study2_alpha_score_bias.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for level in SCORE_BIAS_LEVELS:
        rows = [row for row in agg if row["score_bias"] == level and row["method"] == "P-DAPS" or (
            row["score_bias"] == level and is_warm_alpha_method(row["method"])
        )]
        rows.sort(key=lambda row: float(row["alpha"]))
        plt.plot([row["alpha"] for row in rows], [row["upper_mode_error_mean"] for row in rows],
                 marker="o", label=level)
    plt.xlabel("warm_fraction alpha")
    plt.ylabel("upper_mode_error_mean")
    plt.title("Study 2: Mode Correction Phase Transition")
    plt.grid(True, alpha=0.4)
    plt.legend(title="score bias")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study2_phase_transition.png"), dpi=200)
    plt.close()


def run_study3_snr_alpha(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 3: SNR x Alpha Grid ---")
    raw_rows = []
    for sigma_noise in tqdm(SIGMA_NOISES):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            sigma_noise_override=sigma_noise,
            methods=("DPS", "DAPS", "pULA", "P-DAPS"),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_pdaps_zero_alpha(rows)
        for row in rows:
            row["sigma_noise"] = sigma_noise
            raw_rows.append(row)

    for point in tqdm(list(iter_grid(sigma_noise=SIGMA_NOISES, alpha=P_DAPS_WARM_ALPHA_SUITE))):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            sigma_noise_override=point["sigma_noise"],
            warm_fraction_override=point["alpha"],
            methods=("P-DAPS-warm",),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_warm_alpha_rows(rows, point["alpha"])
        for row in rows:
            row.update(point)
            raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sigma_noise", "alpha", "method"))
    save_to_csv(os.path.join(out_dir, "study3_snr_alpha_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study3_snr_alpha.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for alpha in (0.0, *P_DAPS_WARM_ALPHA_SUITE):
        rows = [
            row for row in agg
            if (row["method"] == "P-DAPS" or is_warm_alpha_method(row["method"]))
            and np.isclose(float(row["alpha"]), alpha)
        ]
        rows.sort(key=lambda row: row["sigma_noise"])
        plt.plot([row["sigma_noise"] for row in rows], [row["fit_error_mean"] for row in rows],
                 marker="o", label=f"alpha={alpha:g}")
    plt.xlabel("sigma_noise")
    plt.ylabel("fit_error_mean")
    plt.title("Study 3: SNR Sensitivity by Alpha")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study3_snr_alpha_lines.png"), dpi=200)
    plt.close()


def run_study4_stability(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 4: Step-Size Stability Boundary ---")
    raw_rows = []
    for step in tqdm(STABILITY_STEPS):
        rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            lgvd_lr_override=step,
            lgvd_step_size_override=step,
            pula_step_size_override=step,
            methods=("DAPS", "pULA", "P-DAPS"),
            best_params_by_scenario=best_params_by_scenario,
        )
        label_pdaps_zero_alpha(rows)
        for row in rows:
            row["langevin_step"] = step
            raw_rows.append(row)

        for alpha in P_DAPS_WARM_ALPHA_SUITE:
            rows = run_study_point(
                scenario_key=scenario_key,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                lgvd_step_size_override=step,
                warm_fraction_override=alpha,
                methods=("P-DAPS-warm",),
                best_params_by_scenario=best_params_by_scenario,
            )
            label_warm_alpha_rows(rows, alpha)
            for row in rows:
                row["langevin_step"] = step
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("langevin_step", "method"))
    save_to_csv(os.path.join(out_dir, "study4_stability_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study4_stability.csv"), agg)

    study4_methods = ("DAPS", "pULA", "P-DAPS", *(warm_alpha_method(alpha) for alpha in P_DAPS_WARM_ALPHA_SUITE))

    plt.figure(figsize=(8, 5.5))
    for method in study4_methods:
        rows = [row for row in agg if row["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda row: row["langevin_step"])
        plt.plot([row["langevin_step"] for row in rows], [1.0 - row["diverged_mean"] for row in rows],
                 marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Langevin step size / lr")
    plt.ylabel("survival rate (1 - diverged_mean)")
    plt.ylim(-0.05, 1.05)
    plt.title("Study 4: Stability Boundary")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study4_stability_survival.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for method in study4_methods:
        rows = [row for row in agg if row["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda row: row["langevin_step"])
        plt.plot([row["langevin_step"] for row in rows], [row["nan_count_mean"] for row in rows],
                 marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Langevin step size / lr")
    plt.ylabel("nan_count_mean")
    plt.title("Study 4: NaN Breakdown")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study4_nan_count.png"), dpi=200)
    plt.close()

    plot_metric_panels(
        agg, "langevin_step", ("DAPS", "pULA", "P-DAPS"),
        os.path.join(out_dir, "study4_metric_panels.png"),
        "Study 4 — Step-Size Stability: posterior-space vs operator-space",
    )


def run_daps_breakdown_bracket(repeats, nb, gt_samples, out_dir,
                               scenarios=DAPS_BREAKDOWN_SCENARIOS,
                               step_grid=DAPS_BREAKDOWN_STEPS):
    """DAPS-only fine sub-grid sweep across canonical scenarios. Brackets the
    step-size breakdown more precisely than Study 4's half-decade grid, at a
    fraction of the cost (single sampler, no warm-fraction inner sweep).
    """
    print("\n--- DAPS breakdown bracket (fine sub-grid, DAPS only) ---")
    raw_rows = []
    for scenario in scenarios:
        for step in tqdm(step_grid, desc=f"DAPS {scenario}", leave=False):
            rows = run_study_point(
                scenario_key=scenario,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                lgvd_lr_override=step,
                methods=("DAPS",),
            )
            for row in rows:
                row["langevin_step"] = step
                row["scenario"] = scenario
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("scenario", "langevin_step", "method"))
    save_to_csv(os.path.join(out_dir, "daps_breakdown_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "daps_breakdown.csv"), agg)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for scenario in scenarios:
        rows = sorted([r for r in agg if r.get("scenario") == scenario],
                      key=lambda r: r["langevin_step"])
        if not rows:
            continue
        steps = [r["langevin_step"] for r in rows]
        survival = [1.0 - float(r.get("diverged_mean", 0.0) or 0.0) for r in rows]
        ax.plot(steps, survival, marker="o", label=scenario)
    ax.set_xscale("log")
    ax.set_xlabel("DAPS Langevin step (log)")
    ax.set_ylabel("survival rate (1 - diverged_mean)")
    ax.set_title("DAPS Breakdown Bracket — survival vs step, per scenario")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "daps_breakdown_survival.png"), dpi=200)
    plt.close(fig)

    print("\nMeasured DAPS breakdown bracket per scenario:")
    print("  scenario                       max_stable_step  min_broken_step")
    for scenario in scenarios:
        rows = sorted([r for r in agg if r.get("scenario") == scenario],
                      key=lambda r: r["langevin_step"])
        stable = [r for r in rows if float(r.get("diverged_mean", 0.0) or 0.0) < 0.5]
        broken = [r for r in rows if float(r.get("diverged_mean", 0.0) or 0.0) >= 0.5]
        max_stable = stable[-1]["langevin_step"] if stable else float("nan")
        min_broken = broken[0]["langevin_step"] if broken else float("nan")
        print(f"  {scenario:<30s} {max_stable:>14.3e}  {min_broken:>14.3e}")


def validate_scenario_daps_defaults(out_dir, safety_margin=0.5):
    """If a daps_breakdown.csv exists in out_dir, assert each scenario's default
    daps_langevin_lr is below safety_margin x the measured breakdown step.
    Raises RuntimeError if any scenario default is above the safe bound — DAPS
    must never run past breakdown in a comparison study.
    Returns dict of measured breakdown values (or None if no CSV)."""
    csv_path = os.path.join(out_dir, "daps_breakdown.csv")
    if not os.path.exists(csv_path):
        # Fall back to the most recent results/daps-breakdown_* directory.
        results_root = os.path.dirname(os.path.abspath(out_dir)) or "results"
        candidates = []
        if os.path.isdir(results_root):
            for entry in os.listdir(results_root):
                if entry.startswith("daps-breakdown_"):
                    p = os.path.join(results_root, entry, "daps_breakdown.csv")
                    if os.path.exists(p):
                        candidates.append((os.path.getmtime(p), p))
        if candidates:
            candidates.sort()
            csv_path = candidates[-1][1]
            print(f"[validate] using most recent daps_breakdown.csv at {csv_path}")
        else:
            print(f"[validate] no daps_breakdown.csv in {out_dir} or results/daps-breakdown_*; "
                  "skipping scenario-default audit. Run --mode daps-breakdown to refresh.")
            return None
    rows_by_scenario = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("method") != "DAPS":
                continue
            rows_by_scenario.setdefault(row.get("scenario"), []).append(row)
    breakdowns = {}
    violations = []
    for scenario in DAPS_BREAKDOWN_SCENARIOS:
        rows = rows_by_scenario.get(scenario, [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r["langevin_step"]))
        stable = [r for r in rows if float(r.get("diverged_mean") or 0.0) < 0.5]
        if not stable:
            continue
        max_stable = float(stable[-1]["langevin_step"])
        breakdowns[scenario] = max_stable
        toy.configure_scenario(scenario, quiet=True)
        default = float(toy.CURRENT_SCENARIO["run_params"]["daps_langevin_lr"])
        if default > safety_margin * max_stable:
            violations.append((scenario, default, max_stable))
    if violations:
        msg = ["DAPS scenario defaults violate safety margin "
               f"({safety_margin}x measured breakdown):"]
        for scenario, default, max_stable in violations:
            msg.append(
                f"  {scenario}: default daps_langevin_lr={default:.2e} "
                f"> {safety_margin} * measured breakdown {max_stable:.2e}"
            )
        msg.append("Lower the offending defaults in toy_2d.py and re-run.")
        raise RuntimeError("\n".join(msg))
    print(f"[validate] DAPS defaults below {safety_margin}x measured breakdown "
          f"for {len(breakdowns)} scenario(s). Safe to proceed.")
    return breakdowns


def run_study5_sde_robustness(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 5: SDE Robustness ---")
    raw_rows = []
    methods = ("DPS", "DAPS", "pULA", "P-DAPS", "P-DAPS-warm")
    for sde in SDE_VARIANTS:
        for condition in tqdm(CONDITION_NUMBERS, desc=f"Study 5 {sde}"):
            with sde_context(sde):
                rows = run_study_point(
                    scenario_key=scenario_key,
                    repeats=repeats,
                    nb=nb,
                    gt_samples=gt_samples,
                    a_diag_override=[1.0, 1.0 / condition],
                    methods=methods,
                    best_params_by_scenario=best_params_by_scenario,
                )
            for row in rows:
                row["sde"] = sde
                row["condition"] = condition
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sde", "condition", "method"))
    save_to_csv(os.path.join(out_dir, "study5_sde_robustness_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study5_sde_robustness.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for sde in SDE_VARIANTS:
        for method in methods:
            rows = [row for row in agg if row["sde"] == sde and row["method"] == method]
            if not rows:
                continue
            rows.sort(key=lambda row: row["condition"])
            plt.plot(
                [row["condition"] for row in rows],
                [row["fit_error_mean"] for row in rows],
                marker="o",
                label=f"{sde} {method}",
            )
    plt.xscale("log")
    plt.xlabel("condition")
    plt.ylabel("fit_error_mean")
    plt.title("Study 5: VE vs VP Robustness")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study5_sde_robustness_fit_error.png"), dpi=200)
    plt.close()


def run_study5_sde_stability(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study 5b: SDE Step-Size Stability ---")
    raw_rows = []
    hardest_condition = max(CONDITION_NUMBERS)
    methods = ("DAPS", "P-DAPS")
    for sde in SDE_VARIANTS:
        for step in tqdm(STABILITY_STEPS, desc=f"Study 5b {sde}"):
            with sde_context(sde):
                rows = run_study_point(
                    scenario_key=scenario_key,
                    repeats=repeats,
                    nb=nb,
                    gt_samples=gt_samples,
                    a_diag_override=[1.0, 1.0 / hardest_condition],
                    lgvd_lr_override=step,
                    lgvd_step_size_override=step,
                    methods=methods,
                    best_params_by_scenario=best_params_by_scenario,
                )
            for row in rows:
                row["sde"] = sde
                row["condition"] = hardest_condition
                row["langevin_step"] = step
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sde", "condition", "langevin_step", "method"))
    save_to_csv(os.path.join(out_dir, "study5_sde_stability_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "study5_sde_stability.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for sde in SDE_VARIANTS:
        for method in methods:
            rows = [row for row in agg if row["sde"] == sde and row["method"] == method]
            rows.sort(key=lambda row: row["langevin_step"])
            plt.plot(
                [row["langevin_step"] for row in rows],
                [1.0 - row["diverged_mean"] for row in rows],
                marker="o",
                label=f"{sde} {method}",
            )
    plt.xscale("log")
    plt.xlabel("Langevin step size / lr")
    plt.ylabel("survival rate (1 - diverged_mean)")
    plt.ylim(-0.05, 1.05)
    plt.title("Study 5b: VE vs VP Stability")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study5_sde_stability_survival.png"), dpi=200)
    plt.close()


def run_study5_sde_suite(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    run_study5_sde_robustness(
        scenario_key, repeats, nb, gt_samples, out_dir,
        best_params_by_scenario=best_params_by_scenario,
    )
    run_study5_sde_stability(
        scenario_key, repeats, nb, gt_samples, out_dir,
        best_params_by_scenario=best_params_by_scenario,
    )


def _fit_spread(rows):
    values = [float(row["fit_error_mean"]) for row in rows if np.isfinite(row.get("fit_error_mean", np.nan))]
    if not values:
        return np.nan
    return float(max(values) - min(values))


def run_adaptive_vs_fixed(repeats, nb, gt_samples, out_dir,
                          scenarios=("toy_b_stiffness", "toy_c_score_bias"),
                          best_params_by_scenario=None):
    print("\n--- Adaptive vs Fixed Warm-Start ---")
    raw_rows = []
    gate_rows = []
    methods = ("P-DAPS-warm", P_DAPS_ADAPTIVE)
    for scenario in scenarios:
        for alpha in tqdm(P_DAPS_WARM_ALPHA_SUITE, desc=f"adaptive {scenario}"):
            for rep in range(repeats):
                toy.set_global_seed(42 + rep)
                toy.configure_scenario(scenario, quiet=True)
                rp = apply_validated_params(
                    toy.CURRENT_SCENARIO["run_params"],
                    scenario,
                    best_params_by_scenario=best_params_by_scenario,
                    methods=methods,
                )
                budgets = budget_steps_for(rp["N"])
                gt = toy.sample_ground_truth(n_samples=gt_samples)
                gt_summary = toy.summarize_samples(gt)
                results = run_methods(
                    nb=nb,
                    rp=rp,
                    budget_steps=budgets,
                    daps_lr=rp["daps_langevin_lr"],
                    pdaps_step=rp["pdaps_langevin_step_size"],
                    warm_fraction=alpha,
                    methods=methods,
                )
                for method, result in results.items():
                    row = toy._summarize_method_result(method, result, gt_summary)
                    add_nfe_metrics(row, method, result, gt_summary, rp)
                    row["scenario"] = scenario
                    row["alpha_max"] = float(alpha)
                    row["repeat"] = rep
                    raw_rows.append(row)

                    if method == P_DAPS_ADAPTIVE:
                        for stat in result.get("gate_stats", []):
                            alpha_actual_max = stat.get("alpha_max", np.nan)
                            gate_rows.append({
                                "scenario": scenario,
                                "repeat": rep,
                                "alpha_cap": float(alpha),
                                "step": int(stat["step"]),
                                "alpha_mean": stat.get("alpha_mean", np.nan),
                                "alpha_min": stat.get("alpha_min", np.nan),
                                "alpha_max_actual": alpha_actual_max,
                                "alpha_mean_fraction": (
                                    stat.get("alpha_mean", np.nan) / float(alpha) if alpha > 0 else np.nan
                                ),
                                "alpha_max_fraction": (
                                    alpha_actual_max / float(alpha) if alpha > 0 else np.nan
                                ),
                                "drift_mean": stat.get("drift_mean", np.nan),
                                "r_hat_mean": stat.get("r_hat_mean", np.nan),
                                "r_prev_mean": stat.get("r_prev_mean", np.nan),
                            })

    agg = aggregate_rows(raw_rows, key_fields=("scenario", "method", "alpha_max"))
    save_to_csv(os.path.join(out_dir, "adaptive_vs_fixed_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "adaptive_vs_fixed.csv"), agg)
    save_to_csv(os.path.join(out_dir, "adaptive_gate_stats.csv"), gate_rows)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8.5, 4.8 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        for method in methods:
            rows = sorted(
                [row for row in agg if row["scenario"] == scenario and row["method"] == method],
                key=lambda row: row["alpha_max"],
            )
            if not rows:
                continue
            spread = _fit_spread(rows)
            label = f"{method} (spread={spread:.4g})" if np.isfinite(spread) else method
            ax.errorbar(
                [row["alpha_max"] for row in rows],
                [row["fit_error_mean"] for row in rows],
                yerr=[row.get("fit_error_std", 0.0) for row in rows],
                marker="o",
                capsize=4,
                label=label,
                color=METHOD_COLORS_TOY.get(method),
            )
        ax.set_xlabel("alpha_max")
        ax.set_ylabel("fit_error_mean")
        ax.set_title(f"Adaptive vs Fixed Warm-Start: {scenario}")
        ax.grid(True, alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "adaptive_vs_fixed.png"), dpi=200)
    plt.close(fig)

    if gate_rows:
        gate_agg = aggregate_rows(gate_rows, key_fields=("scenario", "alpha_cap", "step"))
        save_to_csv(os.path.join(out_dir, "adaptive_gate_stats_by_step.csv"), gate_agg)
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 4.8 * len(scenarios)), squeeze=False)
        for ax, scenario in zip(axes[:, 0], scenarios):
            for alpha in P_DAPS_WARM_ALPHA_SUITE:
                rows = sorted(
                    [
                        row for row in gate_agg
                        if row["scenario"] == scenario and np.isclose(row["alpha_cap"], alpha)
                    ],
                    key=lambda row: row["step"],
                )
                if not rows:
                    continue
                ax.plot(
                    [row["step"] for row in rows],
                    [row["alpha_mean_mean"] for row in rows],
                    label=f"{alpha:.1f}",
                    alpha=0.85,
                )
            ax.set_xlabel("outer step")
            ax.set_ylabel("mean adaptive alpha")
            ax.set_title(f"Adaptive Gate Alpha Schedule: {scenario}")
            ax.grid(True, alpha=0.35)
            ax.legend(title="alpha cap", ncol=5, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "adaptive_gate_alpha_schedule.png"), dpi=200)
        plt.close(fig)


def run_studyD1_kappa_alpha(repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study D-1: Toy D Kappa x Alpha ---")
    raw_rows = []
    scenario_key = "toy_d_high_d_stiffness"
    methods = ("DPS", "DAPS", "pULA", "P-DAPS")
    with sde_context("VE"):
        for kappa in tqdm(CONDITION_NUMBERS, desc="Study D-1 baselines"):
            rows = run_study_point(
                scenario_key=scenario_key,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                kappa_override=kappa,
                methods=methods,
                best_params_by_scenario=best_params_by_scenario,
            )
            label_pdaps_zero_alpha(rows)
            for row in rows:
                row["kappa"] = kappa
                row["sde"] = "VE"
                raw_rows.append(row)

        for point in tqdm(list(iter_grid(kappa=CONDITION_NUMBERS, alpha=P_DAPS_WARM_ALPHA_SUITE)),
                          desc="Study D-1 warm"):
            rows = run_study_point(
                scenario_key=scenario_key,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                kappa_override=point["kappa"],
                warm_fraction_override=point["alpha"],
                methods=("P-DAPS-warm",),
                best_params_by_scenario=best_params_by_scenario,
            )
            label_warm_alpha_rows(rows, point["alpha"])
            for row in rows:
                row.update(point)
                row["sde"] = "VE"
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sde", "kappa", "alpha", "method"))
    save_to_csv(os.path.join(out_dir, "studyD1_kappa_alpha_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "studyD1_kappa_alpha.csv"), agg)
    warm_rows = [row for row in agg if is_warm_alpha_method(row["method"]) or row["method"] == "P-DAPS"]
    plot_heatmap(warm_rows, "kappa", "alpha", "fit_error_mean",
                 "Study D-1: Toy D Fit Error", os.path.join(out_dir, "studyD1_heatmap_fit_error.png"), log_x=True)
    panel_rows = [r for r in agg if r.get("alpha") in ("0.0", "baseline")]
    plot_metric_panels(
        panel_rows, "kappa", ("DPS", "DAPS", "pULA", "P-DAPS"),
        os.path.join(out_dir, "studyD1_metric_panels.png"),
        "Study D-1 — Toy D Stiffness Signature: posterior-space vs operator-space",
    )


def run_studyD2_stability(repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study D-2: Toy D Step-Size Stability ---")
    raw_rows = []
    scenario_key = "toy_d_high_d_stiffness"
    methods = ("DAPS", "P-DAPS")
    with sde_context("VE"):
        for point in tqdm(list(iter_grid(kappa=CONDITION_NUMBERS, langevin_step=STABILITY_STEPS)),
                          desc="Study D-2"):
            rows = run_study_point(
                scenario_key=scenario_key,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                kappa_override=point["kappa"],
                lgvd_lr_override=point["langevin_step"],
                lgvd_step_size_override=point["langevin_step"],
                methods=methods,
                best_params_by_scenario=best_params_by_scenario,
            )
            label_pdaps_zero_alpha(rows)
            for row in rows:
                row.update(point)
                row["sde"] = "VE"
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sde", "kappa", "langevin_step", "method"))
    save_to_csv(os.path.join(out_dir, "studyD2_stability_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "studyD2_stability.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for kappa in CONDITION_NUMBERS:
        for method in methods:
            rows = [row for row in agg if row["kappa"] == kappa and row["method"] == method]
            rows.sort(key=lambda row: row["langevin_step"])
            plt.plot(
                [row["langevin_step"] for row in rows],
                [1.0 - row["diverged_mean"] for row in rows],
                marker="o",
                label=f"k={kappa} {method}",
            )
    plt.xscale("log")
    plt.xlabel("Langevin step size / lr")
    plt.ylabel("survival rate (1 - diverged_mean)")
    plt.ylim(-0.05, 1.05)
    plt.title("Study D-2: Toy D Stability")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "studyD2_stability_survival.png"), dpi=200)
    plt.close()

    # Per-kappa metric panels (one PNG per kappa) so each curve is readable
    for kappa in CONDITION_NUMBERS:
        rows = [r for r in agg if r["kappa"] == kappa]
        if not rows:
            continue
        plot_metric_panels(
            rows, "langevin_step", ("DAPS", "P-DAPS"),
            os.path.join(out_dir, f"studyD2_metric_panels_k{kappa}.png"),
            f"Study D-2 — Toy D Step-Size Stability (kappa={kappa})",
        )


def run_studyD3_sde_kappa(repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Study D-3: Toy D SDE x Kappa ---")
    raw_rows = []
    scenario_key = "toy_d_high_d_stiffness"
    methods = ("DPS", "DAPS", "pULA", "P-DAPS", "P-DAPS-warm")
    for sde in SDE_VARIANTS:
        for kappa in tqdm(CONDITION_NUMBERS, desc=f"Study D-3 {sde}"):
            with sde_context(sde):
                rows = run_study_point(
                    scenario_key=scenario_key,
                    repeats=repeats,
                    nb=nb,
                    gt_samples=gt_samples,
                    kappa_override=kappa,
                    methods=methods,
                    best_params_by_scenario=best_params_by_scenario,
                )
            for row in rows:
                row["sde"] = sde
                row["kappa"] = kappa
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows, key_fields=("sde", "kappa", "method"))
    save_to_csv(os.path.join(out_dir, "studyD3_sde_kappa_raw.csv"), raw_rows)
    save_to_csv(os.path.join(out_dir, "studyD3_sde_kappa.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    for sde in SDE_VARIANTS:
        for method in methods:
            rows = [row for row in agg if row["sde"] == sde and row["method"] == method]
            rows.sort(key=lambda row: row["kappa"])
            plt.plot(
                [row["kappa"] for row in rows],
                [row["fit_error_mean"] for row in rows],
                marker="o",
                label=f"{sde} {method}",
            )
    plt.xscale("log")
    plt.xlabel("kappa")
    plt.ylabel("fit_error_mean")
    plt.title("Study D-3: Toy D VE vs VP")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "studyD3_sde_kappa_fit_error.png"), dpi=200)
    plt.close()


def run_toy_d_suite(repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    run_studyD1_kappa_alpha(
        repeats, nb, gt_samples, out_dir,
        best_params_by_scenario=best_params_by_scenario,
    )
    run_studyD2_stability(
        repeats, nb, gt_samples, out_dir,
        best_params_by_scenario=best_params_by_scenario,
    )
    run_studyD3_sde_kappa(
        repeats, nb, gt_samples, out_dir,
        best_params_by_scenario=best_params_by_scenario,
    )


def run_fidelity_study(scenario_key, repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Fidelity: Posterior Variance Comparison ---")
    rows = run_study_point(
        scenario_key=scenario_key,
        repeats=repeats,
        nb=nb,
        gt_samples=gt_samples,
        methods=("DPS", "DAPS", "pULA", "P-DAPS"),
        best_params_by_scenario=best_params_by_scenario,
    )
    label_pdaps_zero_alpha(rows)
    for alpha in P_DAPS_WARM_ALPHA_SUITE:
        alpha_rows = run_study_point(
            scenario_key=scenario_key,
            repeats=repeats,
            nb=nb,
            gt_samples=gt_samples,
            warm_fraction_override=alpha,
            methods=("P-DAPS-warm",),
            best_params_by_scenario=best_params_by_scenario,
        )
        rows.extend(label_warm_alpha_rows(alpha_rows, alpha))
    agg = aggregate_rows(rows, key_fields=("method",))
    save_to_csv(os.path.join(out_dir, "study_fidelity_variance_raw.csv"), rows)
    save_to_csv(os.path.join(out_dir, "study_fidelity_variance.csv"), agg)

    plt.figure(figsize=(8, 5.5))
    methods = [row["method"] for row in agg]
    values = [row["posterior_var_x2_error_mean"] for row in agg]
    plt.bar(methods, values)
    plt.ylabel("posterior_var_x2_error_mean")
    plt.title("Fidelity: Posterior Variance Bias Along x2")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study_fidelity_variance_error.png"), dpi=200)
    plt.close()


def run_qualitative_figures(repeats, nb, gt_samples, out_dir, best_params_by_scenario=None):
    print("\n--- Qualitative Scenario Figures ---")
    fig_dir = os.path.join(out_dir, "qualitative")
    averaged_dir = os.path.join(fig_dir, "repeat_averaged")
    with pushd(fig_dir):
        for scenario in ("toy_a_mode_recovery", "toy_b_stiffness", "toy_c_score_bias"):
            run_repeat_averaged_qualitative(
                scenario,
                repeats=repeats,
                nb=nb,
                gt_samples=gt_samples,
                out_dir=averaged_dir,
                best_params_by_scenario=best_params_by_scenario,
            )
            toy.set_global_seed(42)
            toy.run_scenario(
                scenario,
                nb=nb,
                gt_samples_n=gt_samples,
                make_plots=True,
                print_rows=True,
            )
            plot_3d_posterior_samples(
                scenario,
                nb=nb,
                gt_samples=gt_samples,
                out_dir=os.path.join(os.getcwd(), "3d"),
            )
            toy.run_warm_fraction_sweep(
                scenario,
                nb=nb,
                gt_samples_n=gt_samples,
                make_plot=True,
                print_rows=True,
            )


def run_legacy_sweeps(scenario_key, repeats, nb, gt_samples, out_dir):
    print("\n--- Legacy SNR/Conditioning Sweeps ---")
    snr_res = []
    for sigma in tqdm(SIGMA_NOISES):
        snr_res.append(run_sweep_point(
            scenario_key=scenario_key, repeats=repeats, nb=nb, gt_samples=gt_samples,
            sigma_noise_override=sigma,
        ))
    plot_sweep(SIGMA_NOISES, snr_res, "Sigma Noise", "SNR Sweep",
               os.path.join(out_dir, "sweep_snr.png"))

    cond_res = []
    for condition in tqdm(CONDITION_NUMBERS):
        cond_res.append(run_sweep_point(
            scenario_key=scenario_key, repeats=repeats, nb=nb, gt_samples=gt_samples,
            a_diag_override=[1.0, 1.0 / condition],
        ))
    plot_sweep(CONDITION_NUMBERS, cond_res, "Conditioning Number", "Conditioning Sweep",
               os.path.join(out_dir, "sweep_cond.png"), log_x=True)


def main():
    parser = argparse.ArgumentParser(description="Formal toy sweeps and study grids.")
    parser.add_argument(
        "--mode",
        choices=(
            "sweeps", "stability", "alpha-sweep", "cond-alpha", "snr-alpha",
            "score-bias", "sde-robustness", "toy-d", "fidelity", "qualitative", "validation",
            "daps-breakdown", "adaptive-vs-fixed",
            "formal", "formal-with-validation", "all",
        ),
        default="formal",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nb", type=int, default=1500)
    parser.add_argument("--gt-samples", type=int, default=100_000)
    parser.add_argument("--scenario", type=str, default="toy_b_stiffness")
    parser.add_argument("--out-dir", type=str, default="results/formal_study")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    best_params_by_scenario = None

    if args.mode in ("daps-breakdown", "all"):
        run_daps_breakdown_bracket(args.repeats, args.nb, args.gt_samples, args.out_dir)

    if args.mode in ("cond-alpha", "alpha-sweep", "score-bias", "snr-alpha",
                     "stability", "sde-robustness", "toy-d", "fidelity",
                     "qualitative", "formal", "formal-with-validation", "all"):
        validate_scenario_daps_defaults(args.out_dir)

    if args.mode in ("validation", "formal-with-validation", "all"):
        best_params_by_scenario = run_validation_tuning(args.repeats, args.nb, args.gt_samples, args.out_dir)
    if args.mode in ("sweeps",):
        run_legacy_sweeps(args.scenario, args.repeats, args.nb, args.gt_samples, args.out_dir)
    if args.mode in ("cond-alpha", "formal", "formal-with-validation", "all"):
        run_study1_cond_alpha(
            "toy_b_stiffness", args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("alpha-sweep", "score-bias", "formal", "formal-with-validation", "all"):
        run_study2_mode_phase(
            "toy_a_mode_recovery", args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("snr-alpha", "formal", "formal-with-validation", "all"):
        run_study3_snr_alpha(
            "toy_b_stiffness", args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("stability", "formal", "formal-with-validation", "all"):
        run_study4_stability(
            "toy_b_stiffness", args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("sde-robustness", "formal", "formal-with-validation", "all"):
        run_study5_sde_suite(
            "toy_b_stiffness", args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("adaptive-vs-fixed", "formal", "formal-with-validation"):
        run_adaptive_vs_fixed(
            args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("toy-d", "formal", "formal-with-validation", "all"):
        run_toy_d_suite(
            args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("fidelity", "formal", "formal-with-validation", "all"):
        run_fidelity_study(
            args.scenario, args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )
    if args.mode in ("qualitative", "formal", "formal-with-validation", "all"):
        run_qualitative_figures(
            args.repeats, args.nb, args.gt_samples, args.out_dir,
            best_params_by_scenario=best_params_by_scenario,
        )

    print(f"\nStudies complete. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
