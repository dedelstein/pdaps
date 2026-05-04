#!/usr/bin/env python3
"""Overlay dimensionless toy-study ratios onto a fastMRI algorithm config.

Scope (deliberately narrow): only quantities that are dimensionless or
ratio-based transfer cleanly from the 2D toy to multi-coil MRI. This script
extracts and applies exactly those.

Transferred (overlaid onto the MRI YAML):
  - warm-start fraction alpha — dimensionless mixing weight in
    x_init = alpha*x_prev + (1-alpha)*x0_hat. The toy's safe band for
    alpha is a defensible starting point for MRI.
  - efficiency multiplier = DAPS_nfe_to_converge / P-DAPS_nfe_to_converge.
    Method-vs-method ratio on the same problem; used to scale the
    template's `lgvd_config.num_steps` (inner Langevin step count).

NOT transferred (left at MRI-tuned values):
  - Absolute step sizes (gamma, lr): operator scale, noise level, and
    network conditioning differ between toy and MRI. The toy's
    gamma=0.5 working is not evidence MRI gamma=0.5 works.
  - Schedule extents (sigma_max, sigma_min): tied to the EDM training
    distribution and data scale.
  - inner_sigma_max gating threshold: the theoretical bound
    1/sqrt(N*gamma) predicts the right order of magnitude on both toy
    and MRI, but the empirical optima differ (~0.2 toy, ~5 MRI) because
    MRI's outer reverse-ODE has more contractive headroom. Set by MRI
    sweep, not by the toy.

Recorded in provenance only (not written into the YAML):
  - Stability finding: DAPS breakdown step (from daps_breakdown.csv if
    present, else study4_stability.csv); P-DAPS shows no divergence
    across the tested step range (study4_stability).
  - Score-bias sensitivity knee (study2_alpha_score_bias, if present).
  - Adaptive-vs-fixed warm-start gate finding.

These are qualitative thesis-paragraph fodder; they do NOT modify the
output YAML.

Inputs: VE-only toy_b formal-study CSVs produced by toy_sweeps.py.
Output: a fastMRI YAML with two fields overlaid (warm_fraction and
        lgvd_config.num_steps) plus a `_extraction_provenance` block.
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

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


def build_report(alpha_info, stability_info, efficiency_info, bias_info, adaptive_info):
    """Compose the toy-derived ratios + qualitative findings into a thesis-ready report."""
    alpha_status = alpha_info.get("status")
    alpha_value = None
    if alpha_status in ("ok", "baseline_unreached"):
        alpha_value = float(alpha_info["safe_mri_alpha"])

    eff_status = efficiency_info.get("status")
    eff_multiplier = None
    daps_nfe = pdaps_nfe = None
    if eff_status == "ok":
        eff_multiplier = float(efficiency_info["efficiency_multiplier"])
        daps_nfe = float(efficiency_info["daps_nfe_to_converge"])
        pdaps_nfe = float(efficiency_info["pdaps_nfe_to_converge"])

    suggested_inner_steps = None
    if eff_multiplier and eff_multiplier > 1.0:
        suggested_inner_steps = max(5, int(round(100 / eff_multiplier)))

    return OrderedDict([
        ("transferable", OrderedDict([
            ("warm_fraction", OrderedDict([
                ("value", alpha_value),
                ("source", "study1_cond_alpha.csv"),
                ("status", alpha_status),
                ("note", "Dimensionless mixing weight; safe band for MRI starting point."),
            ])),
            ("lgvd_num_steps", OrderedDict([
                ("efficiency_multiplier", eff_multiplier),
                ("daps_nfe_to_converge", daps_nfe),
                ("pdaps_nfe_to_converge", pdaps_nfe),
                ("suggested_value_at_template_100", suggested_inner_steps),
                ("source", "study1_cond_alpha.csv (NFE-to-converge)"),
                ("status", eff_status),
                ("note", "Method-vs-method ratio; divide MRI template's "
                         "lgvd_config.num_steps by this multiplier."),
            ])),
        ])),
        ("qualitative_findings", OrderedDict([
            ("stability", stability_info),
            ("score_bias", bias_info),
            ("adaptive_vs_fixed", adaptive_info),
        ])),
        ("not_transferred", [
            "Absolute step sizes (gamma, lr) — operator scale and EDM training distribution differ.",
            "Schedule extents (sigma_max, sigma_min) — tied to data scale.",
            "inner_sigma_max gating threshold — theory predicts the right order, "
            "empirical optima differ between toy and MRI; tune on MRI.",
        ]),
    ])


def render_markdown(report):
    """Render the report as a small markdown table for the thesis appendix."""
    lines = []
    lines.append("# Toy-derived ratios for fastMRI transfer\n")
    lines.append("Auto-generated by `apply_toy_ratios.py` from `toy_sweeps.py` outputs.\n")
    lines.append("This file is for citation; it does NOT drive `mri_validation.py`. ")
    lines.append("The MRI validation grid is hand-curated and centered on these values.\n\n")

    lines.append("## Transferable\n\n")
    lines.append("| Quantity | Toy-derived value | MRI use |\n")
    lines.append("|---|---|---|\n")
    wf = report["transferable"]["warm_fraction"]
    inner = report["transferable"]["lgvd_num_steps"]
    wf_val = "(unavailable)" if wf["value"] is None else f"{wf['value']:.3f}"
    mult = inner["efficiency_multiplier"]
    suggested = inner["suggested_value_at_template_100"]
    if mult is None:
        inner_cell = "(efficiency multiplier unavailable)"
    elif suggested is None:
        inner_cell = f"(multiplier {mult:.2f} ≤ 1.0; toy does not motivate reducing num_steps)"
    else:
        inner_cell = f"{suggested} (template 100 / multiplier {mult:.2f})"
    lines.append(f"| `warm_fraction` (alpha) | {wf_val} | "
                 f"Center of the validation sweep over warm_fractions. |\n")
    lines.append(f"| `lgvd_config.num_steps` | {inner_cell} | "
                 f"Inner Langevin step count in the validation grid. |\n\n")

    lines.append("## Qualitative findings (recorded only)\n\n")
    for name, info in report["qualitative_findings"].items():
        lines.append(f"- **{name}**: status `{info.get('status', 'unknown')}`\n")

    lines.append("\n## Deliberately not transferred\n\n")
    for reason in report["not_transferred"]:
        lines.append(f"- {reason}\n")

    return "".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Directory containing study1_*.csv etc. produced by toy_sweeps.py")
    p.add_argument("--out-file", default=None,
                   help="JSON output path. Defaults to <results-dir>/toy_ratios.json. "
                        "A companion .md file is always written alongside it.")
    # Legacy --template arg silently accepted to avoid breaking older invocations,
    # but no longer used (we no longer overlay onto a fastMRI YAML).
    p.add_argument("--template", default=None, help=argparse.SUPPRESS)
    p.add_argument("--tolerance", type=float, default=1.05,
                   help="Allowed fit-error slack vs DAPS baseline when picking alpha")
    p.add_argument("--safety-buffer", type=float, default=0.10,
                   help="Additive buffer on alpha for MRI transfer")
    p.add_argument("--warm-mode", choices=("fixed", "adaptive"), default="fixed",
                   help="Affects the adaptive-vs-fixed provenance only")
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

    report = build_report(alpha_info, stability_info, efficiency_info, bias_info, adaptive_info)
    print(json.dumps(report, indent=2, default=str))

    out_json = args.out_file or os.path.join(args.results_dir, "toy_ratios.json")
    out_md = os.path.splitext(out_json)[0] + ".md"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_md, "w") as f:
        f.write(render_markdown(report))
    print(f"[apply_toy_ratios] wrote {out_json}")
    print(f"[apply_toy_ratios] wrote {out_md}")


if __name__ == "__main__":
    main()
