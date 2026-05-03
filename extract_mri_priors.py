#!/usr/bin/env python3
"""Translate toy-study sweep outputs into a fastMRI algorithm config.

Reads the CSVs produced by toy_sweeps.py, extracts:
  - optimal warm-start fraction alpha (from study1_cond_alpha)
  - QUALITATIVE stability finding: DAPS has a measured breakdown step
    (from daps_breakdown.csv if present, else study4_stability.csv);
    P-DAPS shows no divergence across the tested step range
    (from study4_stability)
  - efficiency multiplier = DAPS nfe_to_converge / P-DAPS nfe_to_converge
  - score-bias sensitivity knee (from study2_alpha_score_bias, if present)
and writes a YAML config that overlays a fastMRI DAPS template with the
P-DAPS-warm priors. Absolute step sizes do not transfer across problem
scales; the qualitative stability finding is recorded in provenance only,
and the template's fastMRI lr is NOT scaled by a toy-derived ratio (the
old "stability ratio" was grid-censored on both ends and meaningless).

Precondition: study1/study4 inputs are the VE-only toy_b formal-study CSVs.
SDE robustness and Toy D studies intentionally write separate CSV schemas.
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

P_DAPS_WARM_ALPHA_PREFIX = "P-DAPS-warm alpha="
P_DAPS_ADAPTIVE = "P-DAPS-adaptive"


def _read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def _is_pdaps_warm_method(method):
    return method == "P-DAPS-warm" or str(method).startswith(P_DAPS_WARM_ALPHA_PREFIX)


def _pdaps_warm_rows(df):
    return df[df["method"].map(_is_pdaps_warm_method)]


def find_optimal_alpha(study1_df, tolerance=1.05, safety_buffer=0.10):
    """Smallest alpha such that P-DAPS-warm fit_error matches DAPS baseline.

    Uses the hardest condition number available (worst-case for MRI transfer).
    """
    if study1_df is None or "alpha" not in study1_df.columns:
        return {"status": "missing_study1"}

    worst_cond = study1_df["condition"].max()
    slice_df = study1_df[study1_df["condition"] == worst_cond]

    daps_rows = slice_df[slice_df["method"] == "DAPS"]
    pdaps_rows = _pdaps_warm_rows(slice_df)
    if daps_rows.empty or pdaps_rows.empty:
        return {"status": "missing_methods", "condition": float(worst_cond)}

    daps_baseline = float(daps_rows["fit_error_mean"].mean())
    threshold = daps_baseline * tolerance

    successful = pdaps_rows[pdaps_rows["fit_error_mean"] <= threshold]
    if successful.empty:
        best_row = pdaps_rows.loc[pdaps_rows["fit_error_mean"].idxmin()]
        return {
            "status": "baseline_unreached",
            "daps_baseline": daps_baseline,
            "threshold": threshold,
            "best_toy_alpha": float(best_row["alpha"]),
            "best_fit_error": float(best_row["fit_error_mean"]),
            "safe_mri_alpha": float(min(1.0, best_row["alpha"] + safety_buffer)),
            "condition": float(worst_cond),
        }

    optimal_alpha = float(successful["alpha"].min())
    return {
        "status": "ok",
        "daps_baseline": daps_baseline,
        "threshold": threshold,
        "optimal_toy_alpha": optimal_alpha,
        "safe_mri_alpha": float(min(1.0, optimal_alpha + safety_buffer)),
        "condition": float(worst_cond),
    }


def extract_qualitative_stability(study4_df, daps_breakdown_df=None):
    """Qualitative stability summary — replaces the old 'stability ratio'.

    The previous extract_step_size_ratio returned a ratio whose numerator and
    denominator were both pinned to the edges of STABILITY_STEPS:
    P-DAPS never diverged across the entire sweep (max-stable = grid ceiling)
    and DAPS's last survivor was one half-decade below its true breakdown.
    The ratio (~31 622) was a grid artefact, not a measurement.

    This replacement reports honest claims:
      - DAPS breakdown bracket per scenario (from daps_breakdown_df if
        provided, else from study4_df if it has DAPS rows). Bracketed as
        (max_stable_step, min_broken_step).
      - P-DAPS no-divergence claim across study4's tested step range.
      - Optima of both samplers (argmin fit_error in study4) for context.

    No ratio is computed and no lr scaling is applied to the MRI YAML.
    """
    out = {"status": "ok"}

    if daps_breakdown_df is not None and "scenario" in daps_breakdown_df.columns:
        per_scenario = {}
        for scenario, sub in daps_breakdown_df.groupby("scenario"):
            sub = sub[sub["method"] == "DAPS"].sort_values("langevin_step")
            if sub.empty:
                continue
            stable = sub[sub["diverged_mean"].astype(float) < 0.5]
            broken = sub[sub["diverged_mean"].astype(float) >= 0.5]
            entry = {}
            if not stable.empty:
                entry["max_stable_step"] = float(stable["langevin_step"].max())
            if not broken.empty:
                entry["min_broken_step"] = float(broken["langevin_step"].min())
            if entry:
                per_scenario[str(scenario)] = entry
        if per_scenario:
            out["daps_breakdown_per_scenario"] = per_scenario
            out["daps_breakdown_source"] = "daps_breakdown.csv (fine sub-grid)"

    if study4_df is not None and "langevin_step" in study4_df.columns:
        s4 = study4_df

        if "daps_breakdown_per_scenario" not in out:
            daps = s4[s4["method"] == "DAPS"].sort_values("langevin_step")
            if not daps.empty and "diverged_mean" in daps.columns:
                stable = daps[daps["diverged_mean"].astype(float) < 0.5]
                broken = daps[daps["diverged_mean"].astype(float) >= 0.5]
                entry = {}
                if not stable.empty:
                    entry["max_stable_step"] = float(stable["langevin_step"].max())
                if not broken.empty:
                    entry["min_broken_step"] = float(broken["langevin_step"].min())
                if entry:
                    out["daps_breakdown_study4"] = entry
                    out["daps_breakdown_source"] = "study4_stability.csv (half-decade grid; fine sub-grid not run)"

        pdaps = s4[s4["method"] == "P-DAPS"].sort_values("langevin_step")
        if not pdaps.empty and "diverged_mean" in pdaps.columns:
            broken = pdaps[pdaps["diverged_mean"].astype(float) >= 0.5]
            tested_steps = pdaps["langevin_step"].astype(float)
            out["pdaps_no_divergence_in_tested_range"] = bool(broken.empty)
            out["pdaps_step_range_tested"] = [float(tested_steps.min()), float(tested_steps.max())]

        if "fit_error_mean" in s4.columns:
            for method, key in (("DAPS", "daps_optimum_step"),
                                ("P-DAPS", "pdaps_optimum_step")):
                rows = s4[s4["method"] == method]
                if "diverged_mean" in rows.columns:
                    rows = rows[rows["diverged_mean"].astype(float) < 0.5]
                rows = rows[rows["fit_error_mean"].notna()]
                if rows.empty:
                    continue
                best = rows.loc[rows["fit_error_mean"].astype(float).idxmin()]
                out[key] = {
                    "step": float(best["langevin_step"]),
                    "fit_error_mean": float(best["fit_error_mean"]),
                }

    if "daps_breakdown_per_scenario" not in out and "daps_breakdown_study4" not in out:
        out["status"] = "no_breakdown_data"
    return out


def extract_efficiency_multiplier(study1_df):
    """nfe_to_converge(DAPS) / nfe_to_converge(P-DAPS-warm) at worst condition."""
    if study1_df is None:
        return {"status": "missing_study1"}
    if "nfe_to_converge_mean" not in study1_df.columns:
        return {"status": "missing_nfe_column"}

    worst_cond = study1_df["condition"].max()
    slice_df = study1_df[study1_df["condition"] == worst_cond]
    daps_nfe = slice_df[slice_df["method"] == "DAPS"]["nfe_to_converge_mean"].mean()
    pdaps_nfe = _pdaps_warm_rows(slice_df)["nfe_to_converge_mean"].min()

    if not (np.isfinite(daps_nfe) and np.isfinite(pdaps_nfe) and pdaps_nfe > 0):
        return {"status": "non_finite", "daps_nfe": float(daps_nfe) if np.isfinite(daps_nfe) else None,
                "pdaps_nfe": float(pdaps_nfe) if np.isfinite(pdaps_nfe) else None}

    return {
        "status": "ok",
        "daps_nfe_to_converge": float(daps_nfe),
        "pdaps_nfe_to_converge": float(pdaps_nfe),
        "efficiency_multiplier": float(daps_nfe / pdaps_nfe),
    }


def extract_score_bias_knee(study2_df):
    """Lowest alpha that keeps upper_mode_error below the 'none' bias baseline."""
    if study2_df is None:
        return {"status": "missing_study2"}
    cols = study2_df.columns
    if "score_bias" not in cols or "alpha" not in cols:
        return {"status": "bad_schema"}

    none_rows = study2_df[study2_df["score_bias"] == "none"]
    harsh_rows = study2_df[study2_df["score_bias"] == "harsh"]
    if none_rows.empty or harsh_rows.empty:
        return {"status": "missing_levels"}

    none_baseline = float(none_rows["upper_mode_error_mean"].min())
    # Only the P-DAPS-warm sweep rows have a numeric alpha; DPS/DAPS/pULA
    # rows carry alpha="baseline" and must not be considered when picking
    # the lowest warm fraction that tolerates harsh bias.
    harsh_alpha_numeric = pd.to_numeric(harsh_rows["alpha"], errors="coerce")
    harsh_rows = harsh_rows[harsh_alpha_numeric.notna()].assign(alpha=harsh_alpha_numeric.dropna())
    tolerated = harsh_rows[harsh_rows["upper_mode_error_mean"] <= none_baseline * 1.25]
    if tolerated.empty:
        return {"status": "harsh_never_matches", "none_baseline": none_baseline}
    return {
        "status": "ok",
        "none_baseline": none_baseline,
        "min_alpha_under_harsh_bias": float(tolerated["alpha"].min()),
    }


def extract_adaptive_vs_fixed(adaptive_df, tolerance=1.05, warm_mode="fixed"):
    if adaptive_df is None:
        return {"status": "missing", "warm_mode": warm_mode}
    required = {"scenario", "method", "alpha_max", "fit_error_mean"}
    if not required.issubset(set(adaptive_df.columns)):
        return {
            "status": "bad_schema",
            "warm_mode": warm_mode,
            "missing_columns": sorted(required - set(adaptive_df.columns)),
        }

    scenarios = {}
    for scenario, sub in adaptive_df.groupby("scenario"):
        fixed = sub[sub["method"] == "P-DAPS-warm"].copy()
        adaptive = sub[sub["method"] == P_DAPS_ADAPTIVE].copy()
        if fixed.empty or adaptive.empty:
            scenarios[str(scenario)] = {"status": "missing_methods"}
            continue
        fixed["fit_error_mean"] = pd.to_numeric(fixed["fit_error_mean"], errors="coerce")
        adaptive["fit_error_mean"] = pd.to_numeric(adaptive["fit_error_mean"], errors="coerce")
        fixed = fixed[fixed["fit_error_mean"].notna()]
        adaptive = adaptive[adaptive["fit_error_mean"].notna()]
        if fixed.empty or adaptive.empty:
            scenarios[str(scenario)] = {"status": "non_finite_fit_error"}
            continue

        fixed_spread = float(fixed["fit_error_mean"].max() - fixed["fit_error_mean"].min())
        adaptive_spread = float(adaptive["fit_error_mean"].max() - adaptive["fit_error_mean"].min())
        fixed_best = float(fixed["fit_error_mean"].min())
        adaptive_best_idx = adaptive["fit_error_mean"].idxmin()
        adaptive_best_row = adaptive.loc[adaptive_best_idx]
        adaptive_best = float(adaptive_best_row["fit_error_mean"])
        scenarios[str(scenario)] = {
            "status": "ok",
            "fixed_spread": fixed_spread,
            "adaptive_spread": adaptive_spread,
            "fixed_best": fixed_best,
            "adaptive_best": adaptive_best,
            "alpha_at_adaptive_best": float(adaptive_best_row["alpha_max"]),
            "matches_best_within_tolerance": bool(adaptive_best <= tolerance * fixed_best),
            "lower_spread_than_fixed": bool(adaptive_spread < fixed_spread),
        }

    return {"status": "ok", "warm_mode": warm_mode, "scenarios": scenarios}


def build_config(template_path, alpha_info, stability_info, efficiency_info, bias_info,
                 adaptive_info=None, warm_mode="fixed"):
    with open(template_path) as f:
        cfg = yaml.safe_load(f)

    if warm_mode == "adaptive":
        cfg["algorithm"]["_target_"] = "algo.pdaps.PDAPSAdaptive"
        cfg["algorithm"]["warm_mode"] = "adaptive"
    else:
        cfg["algorithm"]["_target_"] = "algo.pdaps.PDAPSWarm"
        cfg["algorithm"]["warm_mode"] = "fixed"

    alpha = None
    if alpha_info.get("status") == "ok":
        alpha = alpha_info["safe_mri_alpha"]
    elif alpha_info.get("status") == "baseline_unreached":
        alpha = alpha_info["safe_mri_alpha"]
    if alpha is not None:
        cfg["algorithm"]["warm_fraction"] = float(alpha)

    cfg["algorithm"].setdefault("lgvd_config", {})
    # Note: we deliberately do NOT scale the template's Langevin lr by a
    # toy-derived ratio. The previous "stability ratio" was a grid artefact
    # (both endpoints censored). The qualitative stability finding lives in
    # _extraction_provenance.stability instead.

    anneal = cfg["algorithm"].setdefault("annealing_scheduler_config", {})
    if efficiency_info.get("status") == "ok":
        mult = efficiency_info["efficiency_multiplier"]
        if mult > 1.0:
            current_steps = int(anneal.get("num_steps", 200))
            reduced = max(20, int(round(current_steps / mult)))
            anneal["num_steps"] = reduced
            anneal["_num_steps_scaling_note"] = (
                f"divided template num_steps by efficiency multiplier {mult:.2f}"
                f" (DAPS nfe_to_converge {efficiency_info['daps_nfe_to_converge']:.0f},"
                f" P-DAPS nfe_to_converge {efficiency_info['pdaps_nfe_to_converge']:.0f})"
            )

    cfg["_extraction_provenance"] = {
        "alpha": alpha_info,
        "stability": stability_info,
        "efficiency": efficiency_info,
        "score_bias": bias_info,
        "adaptive_vs_fixed": adaptive_info or {"status": "missing", "warm_mode": warm_mode},
    }
    return cfg


def write_config(cfg, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    header = (
        "# AUTO-GENERATED by extract_mri_priors.py\n"
        "# Do not edit by hand; rerun the toy sweep if you want to refresh.\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Directory containing study1_*.csv etc. produced by toy_sweeps.py")
    p.add_argument("--template", default="configs/daps_config.yaml",
                   help="Base fastMRI config to overlay")
    p.add_argument("--out-file", default="configs/fastmri_auto_prior.yaml")
    p.add_argument("--tolerance", type=float, default=1.05,
                   help="Allowed fit-error slack vs DAPS baseline when picking alpha")
    p.add_argument("--safety-buffer", type=float, default=0.10,
                   help="Additive buffer on alpha for MRI transfer")
    p.add_argument("--warm-mode", choices=("fixed", "adaptive"), default="fixed",
                   help="MRI warm-start mode to write into the generated config")
    args = p.parse_args()

    study1 = _read_csv(os.path.join(args.results_dir, "study1_cond_alpha.csv"))
    study2 = _read_csv(os.path.join(args.results_dir, "study2_alpha_score_bias.csv"))
    study4 = _read_csv(os.path.join(args.results_dir, "study4_stability.csv"))
    daps_breakdown = _read_csv(os.path.join(args.results_dir, "daps_breakdown.csv"))
    adaptive_vs_fixed = _read_csv(os.path.join(args.results_dir, "adaptive_vs_fixed.csv"))

    alpha_info = find_optimal_alpha(study1, tolerance=args.tolerance,
                                    safety_buffer=args.safety_buffer)
    stability_info = extract_qualitative_stability(study4, daps_breakdown_df=daps_breakdown)
    efficiency_info = extract_efficiency_multiplier(study1)
    bias_info = extract_score_bias_knee(study2)
    adaptive_info = extract_adaptive_vs_fixed(
        adaptive_vs_fixed,
        tolerance=args.tolerance,
        warm_mode=args.warm_mode,
    )

    summary = OrderedDict([
        ("alpha", alpha_info),
        ("stability", stability_info),
        ("efficiency", efficiency_info),
        ("score_bias", bias_info),
        ("adaptive_vs_fixed", adaptive_info),
    ])
    print(json.dumps(summary, indent=2, default=str))

    if not os.path.exists(args.template):
        print(f"[extract_mri_priors] template {args.template} missing; writing summary only")
        with open(os.path.join(args.results_dir, "mri_priors_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return

    cfg = build_config(
        args.template,
        alpha_info,
        stability_info,
        efficiency_info,
        bias_info,
        adaptive_info=adaptive_info,
        warm_mode=args.warm_mode,
    )
    write_config(cfg, args.out_file)
    print(f"[extract_mri_priors] wrote {args.out_file}")


if __name__ == "__main__":
    main()
