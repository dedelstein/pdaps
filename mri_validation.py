import argparse
import contextlib
import csv
import collections
import gc
import io
import itertools
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "libs/inversebench"))


from dataloader import MultiCoilMRIDataset
from utilities import compute_metrics_dict, visualize_recon


MODEL_CONFIG = {
    "_target_": "models.precond.EDMPrecond",
    "model_type": "DhariwalUNet",
    "img_resolution": 320,
    "img_channels": 2,
    "label_dim": 0,
    "model_channels": 128,
    "channel_mult": [1, 1, 1, 2, 2],
    "attn_resolutions": [16],
    "num_blocks": 1,
    "dropout": 0.0,
}

ANNEALING = {
    "num_steps": 200,
    "sigma_max": 100,
    "sigma_min": 0.1,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

DAPS_TAU = 0.002028752174814177
WARN_DMO_RATIO = 1.5
WARN_DMO_NORMAL = 0.035
WARN_SSIM_FLOOR = 0.40
DEBUG_MEM_DUMP_GB = 1.0

REVERSE_ODE = {
    "num_steps": 5,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

DPS_SCHEDULER = {
    "num_steps": 1000,
    "schedule": "vp",
    "timestep": "vp",
    "scaling": "vp",
}

PULA_SCHEDULER = {
    "num_steps": 40,
    "sigma_max": 1,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "sqrt",
    "timestep": "log",
}


def grid(points):
    keys = list(points)
    for values in itertools.product(*(points[key] for key in keys)):
        yield dict(zip(keys, values))


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def method_grid(preset="tiny", log_level="INFO"):
    if preset == "final_comparison":
        return _final_comparison_grid(log_level=log_level)
    if preset == "pdaps_ablations":
        return _pdaps_ablations_grid(log_level=log_level)
    if preset == "pdaps_remediation":
        return _pdaps_remediation_grid(log_level=log_level)
    if preset == "pdaps_mechanism":
        return _pdaps_mechanism_grid(log_level=log_level)
    if preset == "pdaps_nullspace_focus":
        return _pdaps_nullspace_focus_grid(log_level=log_level)
    if preset == "pdaps_v2":
        return _pdaps_v2_grid(log_level=log_level)
    if preset == "pdaps_v3":
        return _pdaps_v3_grid(log_level=log_level)
    if preset == "pdaps_v4":
        return _pdaps_v4_grid(log_level=log_level)
    if preset == "pdaps_v5":
        return _pdaps_v5_grid(log_level=log_level)
    if preset == "pdaps_v6":
        return _pdaps_v6_grid(log_level=log_level)
    if preset == "pdaps_v7":
        return _pdaps_v7_grid(log_level=log_level)
    if preset == "pdaps_v8a":
        return _pdaps_v8a_grid(log_level=log_level)
    if preset == "pdaps_v8b":
        return _pdaps_v8b_grid(log_level=log_level)
    if preset == "pdaps_v8c":
        return _pdaps_v8c_grid(log_level=log_level)
    if preset == "pdaps_v8d":
        return _pdaps_v8d_grid(log_level=log_level)
    if preset == "pdaps_v8e":
        return _pdaps_v8e_grid(log_level=log_level)
    if preset == "pdaps_v8f":
        return _pdaps_v8f_grid(log_level=log_level)
    if preset == "pdaps_working":
        return _pdaps_working_grid(log_level=log_level)
    if preset == "pdaps_prelaunch_c0":
        return _pdaps_prelaunch_c0_grid(log_level=log_level)
    if preset == "pdaps_prelaunch_a_v8f":
        return _pdaps_prelaunch_a_grid(base="v8f", log_level=log_level)
    if preset == "pdaps_prelaunch_a_floor0":
        return _pdaps_prelaunch_a_grid(base="floor0", log_level=log_level)
    if preset == "pdaps_prelaunch_a_inf":
        return _pdaps_prelaunch_a_grid(base="inf", log_level=log_level)
    if preset == "pdaps_prelaunch_a_floor0_inf":
        return _pdaps_prelaunch_a_grid(base="floor0_inf", log_level=log_level)
    if preset == "pdaps_prelaunch_b_v8f_balfast":
        return _pdaps_prelaunch_b_grid(base="v8f", anchors="balfast", log_level=log_level)
    if preset == "pdaps_prelaunch_b_v8f_balanced":
        return _pdaps_prelaunch_b_grid(base="v8f", anchors="balanced", log_level=log_level)
    if preset == "pdaps_prelaunch_b_floor0_balfast":
        return _pdaps_prelaunch_b_grid(base="floor0", anchors="balfast", log_level=log_level)
    if preset == "pdaps_prelaunch_b_inf_balfast":
        return _pdaps_prelaunch_b_grid(base="inf", anchors="balfast", log_level=log_level)
    if preset == "pdaps_prelaunch_b_floor0_inf_balfast":
        return _pdaps_prelaunch_b_grid(base="floor0_inf", anchors="balfast", log_level=log_level)
    if preset == "pdaps_prelaunch_baselines":
        return _prelaunch_baseline_grid(log_level=log_level)
    if preset == "pdaps_prelaunch_lrmin":
        return _pdaps_prelaunch_lrmin_grid(log_level=log_level)
    if preset == "check_abandoned":
        return _pdaps_check_abandoned_grid(log_level=log_level)
    if preset == "pdaps_bugcheck":
        return _pdaps_bugcheck_grid(log_level=log_level)
    if preset == "pdaps_targeted":
        return _pdaps_targeted_grid(log_level=log_level)
    pdaps_num_steps_list = [25]   # default unless preset overrides
    if preset == "smoke":
        dps_scales = [1.0]
        daps_lrs = [1e-5]
        pula_gammas = [0.5]
        pdaps_gammas = [0.5]
        warm_fractions = [0.2]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
    elif preset == "pdaps_inner_sweep":
        # Single-axis ablation over the inner-correction gating threshold.
        # Theoretical bound: σ ≲ 1/√(N_inner·γ) ≈ 0.28 for N=25, γ=0.5.
        dps_scales = []
        daps_lrs = [1e-5]              # one DAPS reference point
        pula_gammas = []
        pdaps_gammas = [0.5]
        warm_fractions = [0.2]
        pdaps_inner_sigma_maxes = [0.3, 1.0, 5.0, 20.0, 1e9]
    elif preset == "pdaps_tight":
        # Tight MRI tuning around the toy-derived gate. The goal is to test
        # whether delaying the inner correction and keeping it short preserves
        # detail while retaining data consistency.
        dps_scales = []
        daps_lrs = [3e-6]              # one validated DAPS reference
        pula_gammas = []
        pdaps_gammas = [0.5]
        warm_fractions = [0.1, 0.2]
        pdaps_inner_sigma_maxes = [0.3, 1.0]
        pdaps_num_steps_list = [5, 10]
    elif preset == "iso_nfe":
        # Hold everything else fixed at validated defaults; sweep
        # lgvd_config.num_steps to test if P-DAPS at iso-NFE matches DAPS.
        dps_scales = []
        daps_lrs = [3e-6]              # selected DAPS lr from tiny
        pula_gammas = []
        pdaps_gammas = [0.5]
        warm_fractions = [0.2]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
        pdaps_num_steps_list = [25, 50, 100]
    elif preset == "pdaps_match_nfe":
        # Iso-inner-step comparison vs DAPS. DAPS runs 100 inner Langevin
        # steps per outer step (lgvd_config.num_steps=100); the inner-sweep
        # preset gave P-DAPS only 25, so we couldn't tell if P-DAPS just
        # needed more inner work. This preset runs P-DAPS at 100 inner steps
        # and probes the two productive σ-gating regimes seen in debug logs:
        # the "high gate" (inner fires from start, productive band σ≲7) and
        # the "low gate" (inner fires only at σ≲5, avoiding nullspace blowup).
        dps_scales = []
        daps_lrs = [1e-5]              # DAPS reference (matches inner_sweep)
        pula_gammas = []
        pdaps_gammas = [0.5]
        warm_fractions = [0.2]
        pdaps_inner_sigma_maxes = [5.0, 1e9]   # productive gate + ungated
        pdaps_num_steps_list = [100]
    elif preset == "warm_sweep":
        # Hold everything else at validated defaults; sweep warm_fraction.
        # Toy's safe band is ~[0.1, 0.3]; we span a wider range to verify
        # the band transfers (or doesn't) to multi-coil MRI.
        dps_scales = []
        daps_lrs = []
        pula_gammas = []
        pdaps_gammas = [0.5]
        warm_fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
    elif preset == "tiny":
        dps_scales = [0.5, 1.0, 2.0]
        daps_lrs = [3e-6, 1e-5, 3e-5]
        pula_gammas = [0.25, 0.5, 1.0]
        pdaps_gammas = [0.25, 0.5]
        warm_fractions = [0.1, 0.2]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
    elif preset == "probe":
        # Exploratory grid: wider than `tiny`, still tractable on an L40s.
        # Extends the *stiff* end (lower DAPS lr, lower pULA γ) so the R=8
        # acceleration regime is actually probed instead of hitting the wall
        # of the R=4-tuned grid. Warm-fraction range widened to match toy's
        # safe band [0.1, 0.3] plus 0.5 as a stress point.
        dps_scales = [0.5, 1.0, 2.0, 4.0]
        daps_lrs = [1e-6, 3e-6, 1e-5, 3e-5]
        pula_gammas = [0.1, 0.25, 0.5, 1.0]
        pdaps_gammas = [0.25, 0.5]
        warm_fractions = [0.1, 0.2, 0.3, 0.5]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
    elif preset == "full":
        dps_scales = [0.5, 1.0, 2.0]
        daps_lrs = [3e-6, 1e-5, 3e-5]
        pula_gammas = [0.25, 0.5, 1.0]
        pdaps_gammas = [0.25, 0.5, 1.0]
        warm_fractions = [0.1, 0.2, 0.4]
        pdaps_inner_sigma_maxes = [PDAPS_INNER_SIGMA_MAX]
    else:
        raise ValueError(f"Unknown grid preset: {preset}")

    methods = []
    for p in grid({"guidance_scale": dps_scales}):
        methods.append({
            "method": "DPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.dps.DPS",
                "diffusion_scheduler_config": DPS_SCHEDULER,
                "guidance_scale": p["guidance_scale"],
            },
        })

    for p in grid({"lr": daps_lrs}):
        methods.append({
            "method": "DAPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.daps.DAPS",
                "annealing_scheduler_config": ANNEALING,
                "diffusion_scheduler_config": REVERSE_ODE,
                "lgvd_config": {"num_steps": 100, "lr": p["lr"], "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
            },
        })

    for p in grid({"gamma": pula_gammas}):
        methods.append({
            "method": "pULA",
            "params": p,
            "algorithm": {
                "_target_": "algo.pula.pULA",
                "noise_scheduler_config": PULA_SCHEDULER,
                "K": 4,
                "gamma": p["gamma"],
                "cg_iter": 10,
                "log_level": log_level,
            },
        })

    for p in grid({"gamma": pdaps_gammas, "inner_sigma_max": pdaps_inner_sigma_maxes,
                   "lgvd_num_steps": pdaps_num_steps_list}):
        methods.append(pdaps_entry("P-DAPS", "none", p["gamma"], 0.0,
                                   p["inner_sigma_max"], p["lgvd_num_steps"], log_level))

    for p in grid({"gamma": pdaps_gammas, "warm_fraction": warm_fractions,
                   "inner_sigma_max": pdaps_inner_sigma_maxes,
                   "lgvd_num_steps": pdaps_num_steps_list}):
        methods.append(pdaps_entry("P-DAPS-fixed", "fixed", p["gamma"], p["warm_fraction"],
                                   p["inner_sigma_max"], p["lgvd_num_steps"], log_level))
    return methods


PDAPS_INNER_SIGMA_MAX = 0.3


def _baseline_entries(dps_scales, daps_lrs, pula_gammas, log_level="INFO"):
    methods = []
    for p in grid({"guidance_scale": dps_scales}):
        methods.append({
            "method": "DPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.dps.DPS",
                "diffusion_scheduler_config": DPS_SCHEDULER,
                "guidance_scale": p["guidance_scale"],
            },
        })

    for p in grid({"lr": daps_lrs}):
        methods.append({
            "method": "DAPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.daps.DAPS",
                "annealing_scheduler_config": ANNEALING,
                "diffusion_scheduler_config": REVERSE_ODE,
                "lgvd_config": {
                    "num_steps": 100,
                    "lr": p["lr"],
                    "tau": DAPS_TAU,
                    "lr_min_ratio": 0.01,
                },
            },
        })

    for p in grid({"gamma": pula_gammas}):
        methods.append({
            "method": "pULA",
            "params": p,
            "algorithm": {
                "_target_": "algo.pula.pULA",
                "noise_scheduler_config": PULA_SCHEDULER,
                "K": 4,
                "gamma": p["gamma"],
                "cg_iter": 10,
                "log_level": log_level,
            },
        })
    return methods


def pdaps_entry(method, warm_mode, gamma, warm_fraction,
                inner_sigma_max=PDAPS_INNER_SIGMA_MAX, lgvd_num_steps=25, log_level="INFO",
                lam_floor=0.0, target_lam_floor=None, solve_lam_floor=None,
                noise_lam_floor=None, noise_tau=1.0, noise_mode="full",
                gamma_schedule="constant", gamma_floor=0.0, gamma_ceiling=float("inf"),
                precond_mode="standard", noise_rhs_mode="standard",
                penalty_scale=1.0, penalty_schedule="lambda", penalty_eps=0.0,
                mask_split_eps=None,
                mid_inner_project_every=0, tweedie_reanchor_every=0,
                reanchor_blend_beta=1.0,
                edm_project_post=False, warm_init_strategy="previous",
                inner_gate_mode="sigma", residual_threshold=0.3,
                noise_gate_mode="none", noise_residual_threshold=None,
                noise_sigma_min=None, noise_residual_min=None,
                annealing_override=None,
                sigma_stop_truncate=None,
                tau=1.0,
                lr_min_ratio=0.01,
                label_suffix=""):
    inner_str = "inf" if inner_sigma_max >= 1e8 else f"{inner_sigma_max:g}"
    params = {
        "gamma": gamma, "warm_fraction": warm_fraction,
        "inner_sigma_max": inner_str, "lgvd_num_steps": int(lgvd_num_steps),
    }
    if tau != 1.0:
        params["tau"] = float(tau)
    if lr_min_ratio != 0.01:
        params["lr_min_ratio"] = float(lr_min_ratio)
    if lam_floor > 0.0:
        params["lam_floor"] = float(lam_floor)
    if target_lam_floor is not None:
        params["target_lam_floor"] = float(target_lam_floor)
    if solve_lam_floor is not None:
        params["solve_lam_floor"] = float(solve_lam_floor)
    if noise_lam_floor is not None:
        params["noise_lam_floor"] = float(noise_lam_floor)
    if noise_tau != 1.0:
        params["noise_tau"] = float(noise_tau)
    if noise_mode != "full":
        params["noise_mode"] = noise_mode
    if gamma_schedule != "constant":
        params["gamma_schedule"] = gamma_schedule
    if gamma_floor > 0.0:
        params["gamma_floor"] = float(gamma_floor)
    if math.isfinite(gamma_ceiling):
        params["gamma_ceiling"] = float(gamma_ceiling)
    if precond_mode != "standard":
        params["precond_mode"] = precond_mode
    if noise_rhs_mode != "standard":
        params["noise_rhs_mode"] = noise_rhs_mode
    if penalty_scale != 1.0:
        params["penalty_scale"] = float(penalty_scale)
    if penalty_schedule != "lambda":
        params["penalty_schedule"] = penalty_schedule
    if penalty_eps > 0.0:
        params["penalty_eps"] = float(penalty_eps)
    if mask_split_eps is not None:
        params["mask_split_eps"] = float(mask_split_eps)
    if mid_inner_project_every > 0:
        params["mid_inner_project_every"] = int(mid_inner_project_every)
    if tweedie_reanchor_every > 0:
        params["tweedie_reanchor_every"] = int(tweedie_reanchor_every)
    if reanchor_blend_beta != 1.0:
        params["reanchor_blend_beta"] = float(reanchor_blend_beta)
    if edm_project_post:
        params["edm_proj"] = True
    if warm_init_strategy != "previous":
        params["warm_init_strategy"] = warm_init_strategy
    if inner_gate_mode != "sigma":
        params["inner_gate_mode"] = inner_gate_mode
        params["residual_threshold"] = float(residual_threshold)
    if noise_gate_mode != "none":
        params["noise_gate_mode"] = noise_gate_mode
        if noise_gate_mode in {"residual", "compound"}:
            params["noise_residual_threshold"] = float(
                residual_threshold if noise_residual_threshold is None else noise_residual_threshold
            )
        if noise_gate_mode in {"sigma_early", "compound_early"}:
            params["noise_sigma_min"] = None if noise_sigma_min is None else float(noise_sigma_min)
        if noise_gate_mode in {"residual_early", "compound_early"}:
            params["noise_residual_min"] = None if noise_residual_min is None else float(noise_residual_min)
    method_label = method + (f"[{label_suffix}]" if label_suffix else "")
    lgvd_config = {
        "num_steps": int(lgvd_num_steps),
        "gamma": gamma,
        "cg_iter": 10,
        "lr_min_ratio": float(lr_min_ratio),
        "tau": float(tau),
        "lam_floor": float(lam_floor),
        "noise_tau": float(noise_tau),
        "noise_mode": noise_mode,
        "gamma_schedule": gamma_schedule,
        "precond_mode": precond_mode,
        "noise_rhs_mode": noise_rhs_mode,
        "penalty_scale": float(penalty_scale),
        "penalty_schedule": penalty_schedule,
        "penalty_eps": float(penalty_eps),
        "mid_inner_project_every": int(mid_inner_project_every),
        "tweedie_reanchor_every": int(tweedie_reanchor_every),
        "reanchor_blend_beta": float(reanchor_blend_beta),
    }
    if mask_split_eps is not None:
        lgvd_config["mask_split_eps"] = float(mask_split_eps)
    if gamma_floor > 0.0:
        lgvd_config["gamma_floor"] = float(gamma_floor)
    if math.isfinite(gamma_ceiling):
        lgvd_config["gamma_ceiling"] = float(gamma_ceiling)
    if target_lam_floor is not None:
        lgvd_config["target_lam_floor"] = float(target_lam_floor)
    if solve_lam_floor is not None:
        lgvd_config["solve_lam_floor"] = float(solve_lam_floor)
    if noise_lam_floor is not None:
        lgvd_config["noise_lam_floor"] = float(noise_lam_floor)
    annealing_scheduler_config = ANNEALING
    if annealing_override:
        annealing_scheduler_config = {**ANNEALING, **annealing_override}
        for key, value in annealing_override.items():
            params[f"annealing_{key}"] = value
    if sigma_stop_truncate is not None:
        params["sigma_stop_truncate"] = float(sigma_stop_truncate)

    return {
        "method": method_label,
        "params": params,
        "algorithm": {
            "_target_": "algo.pdaps.PDAPS",
            "annealing_scheduler_config": annealing_scheduler_config,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": lgvd_config,
            "warm_mode": warm_mode,
            "warm_fraction": warm_fraction,
            "inner_sigma_max": inner_sigma_max,
            "edm_project_post": bool(edm_project_post),
            "warm_init_strategy": warm_init_strategy,
            "inner_gate_mode": inner_gate_mode,
            "residual_threshold": float(residual_threshold),
            "noise_gate_mode": noise_gate_mode,
            "noise_residual_threshold": (
                None if noise_residual_threshold is None else float(noise_residual_threshold)
            ),
            "noise_sigma_min": None if noise_sigma_min is None else float(noise_sigma_min),
            "noise_residual_min": None if noise_residual_min is None else float(noise_residual_min),
            "sigma_stop_truncate": None if sigma_stop_truncate is None else float(sigma_stop_truncate),
            "log_level": log_level,
        },
    }


def pdaps_working_entry(label_suffix, lgvd_num_steps, sigma_stop_truncate, log_level="INFO"):
    """
    Narrow production-tuning surface for the current working P-DAPS core.

    Keep exploratory solver/noise/gate branches out of validation tuning unless
    a diagnostic preset intentionally reopens them.
    """
    return pdaps_core_entry(
        label_suffix=label_suffix,
        lgvd_num_steps=lgvd_num_steps,
        sigma_stop_truncate=sigma_stop_truncate,
        log_level=log_level,
    )


def pdaps_core_entry(
        label_suffix,
        lgvd_num_steps,
        sigma_stop_truncate,
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        solve_lam_floor=3.0,
        lr_min_ratio=0.01,
        cg_iter=10,
        tau=DAPS_TAU,
        log_level="INFO"):
    """
    Slim production P-DAPS entry.

    Deleted branches from algo.pdaps.PDAPS are intentionally absent here:
    inner noise, nonstandard preconditioners, target lambda floors, residual
    gates, reanchor/projection, and warm-init variants.
    """
    inner_str = "inf" if inner_sigma_max >= 1e8 else f"{inner_sigma_max:g}"
    params = {
        "gamma": float(gamma),
        "warm_fraction": float(warm_fraction),
        "inner_sigma_max": inner_str,
        "lgvd_num_steps": int(lgvd_num_steps),
        "tau": float(tau),
        "solve_lam_floor": float(solve_lam_floor),
        "lr_min_ratio": float(lr_min_ratio),
    }
    if sigma_stop_truncate is not None:
        params["sigma_stop_truncate"] = float(sigma_stop_truncate)
    if cg_iter != 10:
        params["cg_iter"] = int(cg_iter)
    return {
        "method": f"P-DAPS-core[{label_suffix}]",
        "params": params,
        "algorithm": {
            "_target_": "algo.pdaps_core.PDAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {
                "num_steps": int(lgvd_num_steps),
                "gamma": float(gamma),
                "tau": float(tau),
                "solve_lam_floor": float(solve_lam_floor),
                "lr_min_ratio": float(lr_min_ratio),
                "cg_iter": int(cg_iter),
            },
            "warm_fraction": float(warm_fraction),
            "inner_sigma_max": float(inner_sigma_max),
            "sigma_stop_truncate": None if sigma_stop_truncate is None else float(sigma_stop_truncate),
            "log_level": log_level,
        },
    }


def _final_comparison_grid(log_level="INFO"):
    """
    Pre-registered final head-to-head grid.

    Selection is by validation SSIM only; PSNR and data misfit are secondary
    tie-breakers. Test rows should be generated only after selected.json is
    frozen for the disjoint patient split.
    """
    methods = _baseline_entries(
        dps_scales=[0.5, 1.0, 2.0],
        daps_lrs=[1e-6, 3e-6, 1e-5],
        pula_gammas=[0.25, 0.5, 1.0],
        log_level=log_level,
    )
    for p in grid({
        "lgvd_num_steps": [25, 50],
        "sigma_stop_truncate": [0.17, 0.25, 0.38],
        "gamma": [0.5, 0.75],
    }):
        stop_label = f"{p['sigma_stop_truncate']:g}".replace(".", "p")
        gamma_label = f"{p['gamma']:g}".replace(".", "p")
        entry = pdaps_core_entry(
            label_suffix=(
                f"final_lgvd{p['lgvd_num_steps']}_"
                f"stop{stop_label}_gamma{gamma_label}"
            ),
            lgvd_num_steps=p["lgvd_num_steps"],
            sigma_stop_truncate=p["sigma_stop_truncate"],
            gamma=p["gamma"],
            warm_fraction=0.8,
            inner_sigma_max=5.0,
            log_level=log_level,
        )
        entry["params"]["candidate"] = entry["method"]
        entry["method"] = "P-DAPS-core"
        methods.append(entry)
    return methods


def pdaps_legacy_working_entry(label_suffix, lgvd_num_steps, sigma_stop_truncate, log_level="INFO"):
    entry = pdaps_entry(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=lgvd_num_steps,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
        sigma_stop_truncate=sigma_stop_truncate,
        label_suffix=label_suffix,
    )
    entry["params"]["noise_mode"] = "range_only"
    return entry


def _pdaps_ablations_grid(log_level="INFO"):
    """
    Big P-DAPS remediation grid. Drops already-characterized failed
    gamma-schedule and lambda-floor variants, then tests the surviving
    controls plus covariance-matched mask-split and Laplacian preconditioners.
    """
    methods = []
    # Reference: DAPS at its validated lr.
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(method="P-DAPS", warm_mode="none", gamma=0.5, warm_fraction=0.0,
                  inner_sigma_max=5.0, lgvd_num_steps=100, log_level=log_level)

    # (1) Baseline P-DAPS at iso-NFE (no ablation knob applied).
    methods.append(pdaps_entry(**common, label_suffix="baseline"))

    # Existing winners and controls from the first debug run.
    methods.append(pdaps_entry(**common, noise_tau=0.0, label_suffix="drift"))
    methods.append(pdaps_entry(**common, noise_mode="range_only", label_suffix="range_noise"))
    methods.append(pdaps_entry(**common, noise_mode="image_only", label_suffix="null_noise"))
    methods.append(pdaps_entry(**common, edm_project_post=True, label_suffix="edmproj"))

    # Fine-grained noise-temperature sweep.
    methods.append(pdaps_entry(**common, noise_tau=0.025, label_suffix="tau0p025"))
    methods.append(pdaps_entry(**common, noise_tau=0.05, label_suffix="tau0p05"))
    methods.append(pdaps_entry(**common, noise_tau=0.1, label_suffix="tau0p1"))
    methods.append(pdaps_entry(**common, noise_tau=0.2, label_suffix="tau0p2"))

    # Fourier-mask proxy for a TSVD subspace split.
    methods.append(pdaps_entry(**common, precond_mode="mask_split",
                               noise_rhs_mode="matched", label_suffix="split_matched"))
    methods.append(pdaps_entry(**common, precond_mode="mask_split",
                               noise_rhs_mode="matched", noise_tau=0.0,
                               label_suffix="split_drift"))
    methods.append(pdaps_entry(**common, precond_mode="mask_split",
                               noise_rhs_mode="matched", noise_tau=0.1,
                               label_suffix="split_tau0p1"))

    # General-form Tikhonov with covariance-matched Laplacian noise.
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", label_suffix="lap_matched"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", noise_tau=0.0,
                               label_suffix="lap_drift"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", noise_tau=0.1,
                               label_suffix="lap_tau0p1"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="heuristic", label_suffix="lap_heur"))

    # Laplacian strength controls: diagnose whether σ-coupled weighting is too weak.
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", penalty_scale=10.0,
                               label_suffix="lap10_matched"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", penalty_scale=100.0,
                               label_suffix="lap100_matched"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", penalty_scale=0.1,
                               penalty_schedule="constant",
                               label_suffix="lap_mu0p1_matched"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", penalty_scale=1.0,
                               penalty_schedule="constant",
                               label_suffix="lap_mu1_matched"))
    return methods


def _pdaps_remediation_grid(log_level="INFO"):
    """
    Post-patch ablations validating the lam_floor / matched-mode fixes.

    Compares standard, Laplacian, and mask-split preconditioners under the
    fixed pipeline, with focused floor and noise-temperature sweeps.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(method="P-DAPS", warm_mode="none", gamma=0.5, warm_fraction=0.0,
                  inner_sigma_max=5.0, lgvd_num_steps=100, log_level=log_level)

    # Standard-preconditioner floor sweep.
    methods.append(pdaps_entry(**common, label_suffix="baseline_lf0"))
    methods.append(pdaps_entry(**common, lam_floor=0.01, label_suffix="baseline_lf0p01"))
    methods.append(pdaps_entry(**common, lam_floor=0.1, label_suffix="baseline_lf0p1"))
    methods.append(pdaps_entry(**common, lam_floor=1.0, label_suffix="baseline_lf1"))
    methods.append(pdaps_entry(**common, lam_floor=10.0, label_suffix="baseline_lf10"))
    methods.append(pdaps_entry(**common, solve_lam_floor=1.0, label_suffix="baseline_solve_lf1"))
    methods.append(pdaps_entry(**common, noise_lam_floor=1.0, label_suffix="baseline_noise_lf1"))

    # Laplacian matched-mode validation after threading lam_solve into the solve.
    lap_common = dict(common, precond_mode="laplacian", noise_rhs_mode="matched")
    methods.append(pdaps_entry(**lap_common, label_suffix="lap_matched_lf0"))
    methods.append(pdaps_entry(**lap_common, lam_floor=0.1, label_suffix="lap_matched_lf0p1"))
    methods.append(pdaps_entry(**lap_common, lam_floor=1.0, label_suffix="lap_matched_lf1"))
    methods.append(pdaps_entry(**lap_common, lam_floor=10.0, label_suffix="lap_matched_lf10"))
    methods.append(pdaps_entry(**lap_common, lam_floor=1.0, penalty_scale=10.0,
                               label_suffix="lap_matched_pen10_lf1"))
    methods.append(pdaps_entry(**lap_common, lam_floor=1.0, penalty_scale=100.0,
                               label_suffix="lap_matched_pen100_lf1"))

    # Mask-split matched-mode validation plus a non-matched eps knob check.
    split_common = dict(common, precond_mode="mask_split", noise_rhs_mode="matched")
    methods.append(pdaps_entry(**split_common, label_suffix="split_matched_lf0"))
    methods.append(pdaps_entry(**split_common, lam_floor=1.0, label_suffix="split_matched_lf1"))
    methods.append(pdaps_entry(**common, precond_mode="mask_split",
                               noise_rhs_mode="heuristic", mask_split_eps=0.1,
                               lam_floor=1.0, label_suffix="split_heur_eps0p1_lf1"))

    # Noise-mode semantics under the standard preconditioner.
    methods.append(pdaps_entry(**common, lam_floor=1.0, noise_mode="image_only",
                               label_suffix="image_only_lf1"))
    methods.append(pdaps_entry(**common, lam_floor=1.0, noise_mode="null_only",
                               label_suffix="null_only_lf1"))
    methods.append(pdaps_entry(**common, lam_floor=1.0, noise_mode="range_only",
                               label_suffix="range_only_lf1"))

    # 2D sweep at the working Laplacian matched corner.
    floor_labels = [(0.1, "0p1"), (1.0, "1"), (10.0, "10")]
    tau_labels = [(0.5, "0p5"), (1.0, "1")]
    for lam_floor, lf_label in floor_labels:
        for noise_tau, tau_label in tau_labels:
            methods.append(pdaps_entry(
                **lap_common,
                lam_floor=lam_floor,
                noise_tau=noise_tau,
                label_suffix=f"lap_matched_lf{lf_label}_tau{tau_label}",
            ))

    return methods


def _pdaps_targeted_grid(log_level="INFO"):
    """
    Targeted follow-up to the broad P-DAPS remediation ablation.

    Keeps the competitive candidates and tests the two main mechanism
    conjectures from the DEBUG traces:
    (1) useful stochasticity is either very small or range-restricted;
    (2) delaying inner activation may prevent unrecoverable nullspace growth.
    Known complete failures are intentionally dropped, except baseline as a
    single failure reference.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(method="P-DAPS", warm_mode="none", gamma=0.5, warm_fraction=0.0,
                  inner_sigma_max=5.0, lgvd_num_steps=100, log_level=log_level)

    # One failure reference to confirm the known nullspace-injection mode.
    methods.append(pdaps_entry(**common, label_suffix="baseline"))

    # Main candidate set from the 2026-05-08 debug ablation.
    methods.append(pdaps_entry(**common, noise_tau=0.0, label_suffix="drift"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", noise_tau=0.0,
                               label_suffix="lap_drift"))
    methods.append(pdaps_entry(**common, noise_mode="range_only", label_suffix="range_noise"))
    methods.append(pdaps_entry(**common, noise_tau=0.025, label_suffix="tau0p025"))
    methods.append(pdaps_entry(**common, noise_tau=0.05, label_suffix="tau0p05"))

    # Fine tiny-noise probes around the apparent stable stochastic window.
    methods.append(pdaps_entry(**common, noise_tau=0.005, label_suffix="tau0p005"))
    methods.append(pdaps_entry(**common, noise_tau=0.01, label_suffix="tau0p01"))
    methods.append(pdaps_entry(**common, noise_tau=0.02, label_suffix="tau0p02"))

    # Laplacian + small stochasticity probes: does the good deterministic
    # Laplacian correction tolerate tiny matched noise?
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", noise_tau=0.025,
                               label_suffix="lap_tau0p025"))
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", noise_tau=0.05,
                               label_suffix="lap_tau0p05"))

    # Borderline strength control from the broad run: keep one scaled
    # Laplacian matched-noise point to see if it survives broader sampling.
    methods.append(pdaps_entry(**common, precond_mode="laplacian",
                               noise_rhs_mode="matched", penalty_scale=100.0,
                               label_suffix="lap100_matched"))

    # Delayed inner-activation probes for the best stochastic candidates.
    late_common = dict(common)
    late_common["inner_sigma_max"] = 3.0
    methods.append(pdaps_entry(**late_common, noise_tau=0.0, label_suffix="drift_s3"))
    methods.append(pdaps_entry(**late_common, noise_mode="range_only", label_suffix="range_s3"))
    methods.append(pdaps_entry(**late_common, noise_tau=0.025, label_suffix="tau0p025_s3"))

    return methods


def _pdaps_mechanism_grid(log_level="INFO"):
    """
    Follow-up mechanism grid for the 2026-05-09 P-DAPS remediation run.

    Keeps the directly comparable DAPS / range-only / Laplacian references,
    then adds the drift, truncation, warm-start, gamma-schedule, composition,
    EDM projection, mid-inner projection, re-anchor, and null-Laplacian cells.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(method="P-DAPS", warm_mode="none", gamma=0.5, warm_fraction=0.0,
                  inner_sigma_max=5.0, lgvd_num_steps=100, log_level=log_level)
    range_common = dict(common, lam_floor=1.0, noise_mode="range_only")
    lap_common = dict(common, lam_floor=1.0, precond_mode="laplacian",
                      noise_rhs_mode="matched", penalty_scale=100.0)
    lap_null_common = dict(common, lam_floor=1.0, precond_mode="laplacian_null",
                           noise_rhs_mode="matched", penalty_scale=100.0)

    def cell(base, label_suffix, **overrides):
        cfg = dict(base)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    # Direct references from the previous remediation grid.
    methods.append(cell(range_common, "range_only_lf1"))
    methods.append(cell(lap_common, "lap_matched_pen100_lf1"))

    # H1: deterministic drift toward the local Gaussian-MAP target.
    methods.append(cell(range_common, "range_drift", noise_tau=0.0))
    methods.append(cell(lap_common, "lap_pen100_drift", noise_tau=0.0))

    # H2: early stopping / inner truncation.
    methods.append(cell(range_common, "range_only_inner25", lgvd_num_steps=25))
    methods.append(cell(range_common, "range_only_inner50", lgvd_num_steps=50))

    # H3: warm-start regularization.
    methods.append(cell(range_common, "range_only_warm03", warm_mode="fixed", warm_fraction=0.3))
    methods.append(cell(range_common, "range_only_warm05", warm_mode="fixed", warm_fraction=0.5))
    methods.append(cell(lap_common, "lap_pen100_warm03", warm_mode="fixed", warm_fraction=0.3))

    # H4: sigma-dependent gamma schedule and floor.
    methods.append(cell(range_common, "range_only_lambdacap", gamma_schedule="lambda_cap"))
    methods.append(cell(range_common, "range_only_lambdacap_floor",
                        gamma_schedule="lambda_cap", gamma_floor=0.05))
    methods.append(cell(lap_common, "lap_pen100_lambdacap", gamma_schedule="lambda_cap"))

    # H5: range-restricted noise composed with Laplacian preconditioning.
    methods.append(cell(lap_common, "range_lap_pen100", noise_mode="range_only"))

    # H6: once-per-outer EDM projection.
    methods.append(cell(range_common, "range_only_edmproj", edm_project_post=True))
    methods.append(cell(lap_common, "lap_pen100_edmproj", edm_project_post=True))

    # B.1.1: mid-inner EDM projection.
    methods.append(cell(range_common, "range_only_midproj50", mid_inner_project_every=50))
    methods.append(cell(range_common, "range_only_midproj25", mid_inner_project_every=25))
    methods.append(cell(lap_common, "lap_pen100_midproj50", mid_inner_project_every=50))

    # B.1.2: Tweedie re-anchoring inside the inner loop.
    methods.append(cell(range_common, "range_only_reanchor50", tweedie_reanchor_every=50))
    methods.append(cell(range_common, "range_only_reanchor25", tweedie_reanchor_every=25))

    # B.1.3: range-restricted Laplacian penalty.
    methods.append(cell(lap_null_common, "lap_null_pen100"))
    methods.append(cell(lap_null_common, "range_lap_null_pen100", noise_mode="range_only"))

    return methods


def _pdaps_nullspace_focus_grid(log_level="INFO"):
    """
    Compact confirmation grid for the null-space noise mechanism.

    Intended for multi-slice / multi-seed reruns after the broad remediation
    screen: DAPS, the two strongest single fixes, their composition, one
    warm-start probe on the composed candidate, and a small lower-σ gate sweep
    on the composed candidate.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(method="P-DAPS", warm_mode="none", gamma=0.5, warm_fraction=0.0,
                  inner_sigma_max=5.0, lgvd_num_steps=100, log_level=log_level,
                  lam_floor=1.0)
    lap_common = dict(common, precond_mode="laplacian", noise_rhs_mode="matched")

    def cell(base, label_suffix, **overrides):
        cfg = dict(base)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    methods.append(cell(common, "range_only_lf1", noise_mode="range_only"))
    methods.append(cell(lap_common, "lap_matched_pen100_lf1", penalty_scale=100.0))
    methods.append(cell(lap_common, "lap_matched_lf1_tau0p5", noise_tau=0.5))
    methods.append(cell(lap_common, "range_lap_pen100", noise_mode="range_only",
                        penalty_scale=100.0))
    methods.append(cell(lap_common, "range_lap_pen100_warm03", noise_mode="range_only",
                        penalty_scale=100.0, warm_mode="fixed", warm_fraction=0.3))
    methods.append(cell(lap_common, "range_lap_pen100_s3", noise_mode="range_only",
                        penalty_scale=100.0, inner_sigma_max=3.0))
    methods.append(cell(lap_common, "range_lap_pen100_s1", noise_mode="range_only",
                        penalty_scale=100.0, inner_sigma_max=1.0))
    methods.append(cell(lap_common, "range_lap_pen100_warm03_s3", noise_mode="range_only",
                        penalty_scale=100.0, warm_mode="fixed", warm_fraction=0.3,
                        inner_sigma_max=3.0))

    return methods


def _pdaps_v2_grid(log_level="INFO"):
    """
    Single-slice follow-up ablation grid.

    This intentionally keeps all requested cells, including duplicate baseline
    controls under distinct labels, so failures and redundant points are visible
    in the output matrix.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.5,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=1.0,
        noise_mode="range_only",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    methods.append(cell("v2_01_range_only_warm05"))

    for label, lam_floor in (("0p1", 0.1), ("0p3", 0.3), ("1", 1.0), ("3", 3.0)):
        methods.append(cell(f"v2_lam_floor_{label}_warm05", lam_floor=lam_floor))

    methods.append(cell("v2_lam0p3_warm07", lam_floor=0.3, warm_fraction=0.7))
    methods.append(cell("v2_lam0p3_warm03", lam_floor=0.3, warm_fraction=0.3))

    for label, inner_sigma_max in (("3", 3.0), ("7", 7.0), ("10", 10.0)):
        methods.append(cell(f"v2_inner_sigma_{label}", inner_sigma_max=inner_sigma_max))

    methods.append(cell("v2_warm07", warm_fraction=0.7))
    methods.append(cell("v2_warm09", warm_fraction=0.9))
    methods.append(cell("v2_warm_adaptive05", warm_mode="adaptive", warm_fraction=0.5))

    methods.append(cell("v2_noise_tau0_drift", noise_tau=0.0))
    methods.append(cell("v2_noise_tau0p25", noise_tau=0.25))
    methods.append(cell("v2_noise_tau0_warm03", noise_tau=0.0, warm_fraction=0.3))

    methods.append(cell("v2_cgsense_warm_init", warm_init_strategy="cgsense"))
    methods.append(cell(
        "v2_residual_gate_thr0p3",
        inner_gate_mode="residual",
        residual_threshold=0.3,
    ))
    methods.append(cell(
        "v2_decoupled_lam_target0p3_solve1_noise1",
        lam_floor=0.0,
        target_lam_floor=0.3,
        solve_lam_floor=1.0,
        noise_lam_floor=1.0,
    ))
    methods.append(cell(
        "v2_light_reanchor50_beta0p1",
        tweedie_reanchor_every=50,
        reanchor_blend_beta=0.1,
    ))

    return methods


def _pdaps_v3_grid(log_level="INFO"):
    """
    v3 P-DAPS ablation grid centered on tau=0 MAP correction, with targeted
    Langevin-salvage cells for the thesis Bayesian framing.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.2,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=1.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    methods.append(cell("v3_tau1_anchor", noise_tau=1.0, warm_fraction=0.5))
    methods.append(cell("v3_baseline"))
    methods.append(cell("v3_warm0", warm_fraction=0.0))
    methods.append(cell("v3_warm0p1", warm_fraction=0.1))
    methods.append(cell("v3_warm0p3", warm_fraction=0.3))
    methods.append(cell("v3_lam3", lam_floor=3.0))
    methods.append(cell("v3_lam10", lam_floor=10.0))
    methods.append(cell("v3_gamma1", gamma=1.0))
    methods.append(cell(
        "v3_decoupled_target0p3_solve3",
        lam_floor=0.0,
        target_lam_floor=0.3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))
    methods.append(cell("v3_edm_project_post", edm_project_post=True))
    methods.append(cell(
        "v3_reanchor50_beta0p1",
        tweedie_reanchor_every=50,
        reanchor_blend_beta=0.1,
    ))
    methods.append(cell("v3_outer300_poly7", annealing_override={"num_steps": 300}))
    methods.append(cell("v3_tau1_lam10", noise_tau=1.0, lam_floor=10.0))
    methods.append(cell("v3_tau1_edm_project_post", noise_tau=1.0, edm_project_post=True))
    methods.append(cell(
        "v3_tau1_decoupled_target0p3_solve3",
        noise_tau=1.0,
        lam_floor=0.0,
        target_lam_floor=0.3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))

    return methods


def _pdaps_v4_grid(log_level="INFO"):
    """
    v4 P-DAPS grid: two targeted experiments motivated by v2/v3 DEBUG analysis.

    Experiment 1 — inner-step budget under noise_tau=0 (drift-only block):
      The drift-only inner solver converges by ~k=25 of 100 steps; the
      remaining ~75 steps are idle. This tests whether lgvd20/30/50 match
      the lgvd=100 baseline at no quality cost.

    Experiment 2 — objective-side λ_target sweep (CG-decoupled):
      The ~0.18 residual plateau at σ≈5 is a MAP fixed point, not solver
      non-convergence. Solver knobs (warm_fraction, γ, inner steps) are
      inert; the only lever is the objective's effective λ_target.
      solve_lam_floor=3.0 keeps CG well-conditioned throughout; λ_target
      is swept independently from 0.04 (natural 1/σ² at σ=5) to 1.0.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
        },
    })

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.2,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=1.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    # Anchor: identical to v3_baseline for direct cross-grid comparison.
    methods.append(cell("v4_baseline"))

    # Experiment 1: inner-step budget sweep (drift-only block, noise_tau=0).
    # lgvd=100 is already covered by v4_baseline above.
    methods.append(cell("v4_lgvd20", lgvd_num_steps=20))
    methods.append(cell("v4_lgvd30", lgvd_num_steps=30))
    methods.append(cell("v4_lgvd50", lgvd_num_steps=50))

    # Experiment 2: objective-side λ_target sweep, CG-decoupled.
    # lam_floor=0.0 removes the unified floor; solve_lam_floor=3.0 keeps
    # CG conditioning; noise_lam_floor=3.0 is moot under noise_tau=0.
    # target_lam_floor is the only active lever.
    #   0.04 ≈ 1/σ² at σ=5 → floor never binds in the active correction window.
    #   0.3  → in-grid continuity with v3_decoupled_target0p3_solve3.
    #   1.0  → should reproduce v4_baseline behaviour (λ_target floored at 1).
    methods.append(cell(
        "v4_target0p04_solve3",
        lam_floor=0.0,
        target_lam_floor=0.04,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))
    methods.append(cell(
        "v4_target0p1_solve3",
        lam_floor=0.0,
        target_lam_floor=0.1,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))
    methods.append(cell(
        "v4_target0p3_solve3",
        lam_floor=0.0,
        target_lam_floor=0.3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))
    methods.append(cell(
        "v4_target1_solve3",
        lam_floor=0.0,
        target_lam_floor=1.0,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
    ))

    return methods


def _pdaps_v5_grid(log_level="INFO"):
    """
    v5 P-DAPS ablation: validate the τ² objective fix.

    P-DAPS's inner Langevin targets the fixed point AᴴA(x−y) + λ_target·(x−x̂₀) = 0
    with λ_target = 1/σ².  DAPS uses λ_target = τ²/σ² (τ = DAPS_TAU ≈ 0.002),
    making P-DAPS's prior anchor ~243k× stronger.  The fix sets tau=DAPS_TAU so
    λ_target = τ²/σ², matching the DAPS objective.  This grid validates the fix
    in a clean A/B and characterises the knobs that interact with it.

    Block A — objective A/B (drift-only, γ constant)
    Block B — γ-schedule sweep on the fix
    Block C — inner-step budget, rebased on the fix
    Block D — Langevin-noise sweep on the fix
    Block E — floor insurance
    """
    methods = []

    # Standard DAPS reference.
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": DAPS_TAU, "lr_min_ratio": 0.01},
        },
    })

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.2,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=0.0,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    # ── Block A: objective A/B (drift-only, γ constant) ─────────────────────
    # v4_baseline anchor with unified lam_floor=1.0 for cross-grid comparison.
    methods.append(cell("v5a_v4baseline_anchor", tau=1.0, lam_floor=1.0,
                        target_lam_floor=None, solve_lam_floor=None,
                        noise_lam_floor=None))
    # The diagnosed bug in clean form: tau=1 → λ_target = 1/σ² (unfloored).
    methods.append(cell("v5a_tau1_unfloored", tau=1.0))
    # The fix: tau=DAPS_TAU → λ_target = τ²/σ².
    methods.append(cell("v5a_taufix", tau=DAPS_TAU))

    # ── Block B: γ-schedule sweep on the fix (drift-only) ──────────────────
    # (v5a_taufix is the constant-γ arm of this sweep.)
    methods.append(cell("v5b_taufix_glambda", tau=DAPS_TAU,
                        gamma_schedule="lambda"))
    methods.append(cell("v5b_taufix_glambdacap", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap"))
    methods.append(cell("v5b_taufix_gsqrtlambdacap", tau=DAPS_TAU,
                        gamma_schedule="sqrt_lambda_cap"))

    # ── Block C: inner-step budget, rebased on the fix ─────────────────────
    # Pinned to gamma_schedule="lambda_cap"; lgvd=100 is v5b_taufix_glambdacap.
    methods.append(cell("v5c_taufix_lgvd20", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", lgvd_num_steps=20))
    methods.append(cell("v5c_taufix_lgvd30", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", lgvd_num_steps=30))
    methods.append(cell("v5c_taufix_lgvd50", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", lgvd_num_steps=50))

    # ── Block D: Langevin-noise sweep on the fix ───────────────────────────
    # noise_tau here is P-DAPS's temperature knob, not the DAPS τ².
    methods.append(cell("v5d_taufix_noise0p025", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", noise_tau=0.025))
    methods.append(cell("v5d_taufix_noise0p1", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", noise_tau=0.1))
    methods.append(cell("v5d_taufix_noise1p0", tau=DAPS_TAU,
                        gamma_schedule="lambda_cap", noise_tau=1.0))

    # ── Block E: floor insurance ───────────────────────────────────────────
    methods.append(cell("v5e_taufix_floor1em3", tau=DAPS_TAU,
                        target_lam_floor=1e-3))

    return methods


def _pdaps_v6_grid(log_level="INFO"):
    """
    v6 P-DAPS ablation around the v5 winner.

    Common base is v5b_taufix_glambda:
      tau=DAPS_TAU, gamma_schedule="lambda", range-only drift/noise path,
      noise_tau=0, target_lam_floor=0, solve_lam_floor=3, warm_fraction=0.2.

    The grid keeps single-knob deltas for attribution.  Runtime-policy cells
    (outer/lgvd budget) should be analyzed separately from quality cells; Block B
    noise cells intentionally stay on the full outer=200 base.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": DAPS_TAU, "lr_min_ratio": 0.01},
        },
    })

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.2,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=0.0,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    # Block 0: anchors.
    methods.append(cell("v6_anchor_glambda"))
    methods.append(cell("v6_anchor_taufix_const", gamma_schedule="constant"))

    # Block A: schedule and inner budget.
    methods.append(cell("v6_glambda_outer100", annealing_override={"num_steps": 100}))
    methods.append(cell("v6_glambda_outer150", annealing_override={"num_steps": 150}))
    methods.append(cell("v6_glambda_outer300", annealing_override={"num_steps": 300}))
    methods.append(cell("v6_glambda_lgvd50", lgvd_num_steps=50))
    methods.append(cell("v6_glambda_lgvd200", lgvd_num_steps=200))

    # Block B: range-only Langevin noise on the full outer=200 base.
    methods.append(cell("v6_glambda_noise_temp0p005", noise_tau=0.005))
    methods.append(cell("v6_glambda_noise_temp0p025", noise_tau=0.025))
    methods.append(cell("v6_glambda_noise_temp0p1", noise_tau=0.1))
    methods.append(cell("v6_glambda_noise_temp1p0", noise_tau=1.0))

    # Block C: gamma-scale stability/overshoot edge.
    methods.append(cell("v6_glambda_gamma0p75", gamma=0.75))
    methods.append(cell("v6_glambda_gamma1p0", gamma=1.0))

    # Block D: initialization and warm anchoring.
    methods.append(cell("v6_glambda_warm0", warm_fraction=0.0))
    methods.append(cell("v6_glambda_warm0p5", warm_fraction=0.5))
    methods.append(cell("v6_glambda_warm0p8", warm_fraction=0.8))
    methods.append(cell("v6_glambda_warm_adaptive", warm_mode="adaptive", warm_fraction=0.5))
    methods.append(cell("v6_glambda_cgsense_init", warm_init_strategy="cgsense"))

    # Block E: periodic inner-loop Tweedie reanchor.
    methods.append(cell("v6_glambda_reanchor50_beta0p1",
                        tweedie_reanchor_every=50, reanchor_blend_beta=0.1))
    methods.append(cell("v6_glambda_reanchor25_beta0p1",
                        tweedie_reanchor_every=25, reanchor_blend_beta=0.1))

    # Block F: gate and solve-floor sweep.  The solve_floor=3 anchor is Block 0.
    methods.append(cell("v6_glambda_inner_sigma10", inner_sigma_max=10.0))
    methods.append(cell("v6_glambda_inner_sigma_nogate", inner_sigma_max=1e9))
    methods.append(cell("v6_glambda_solve_floor1", solve_lam_floor=1.0, noise_lam_floor=1.0))
    methods.append(cell("v6_glambda_solve_floor5", solve_lam_floor=5.0, noise_lam_floor=5.0))
    methods.append(cell("v6_glambda_solve_floor10", solve_lam_floor=10.0, noise_lam_floor=10.0))

    # Block G: conditional tiny prior-anchor regularization.  Keep in the grid
    # so it can be selected by method index only if image inspection warrants it.
    methods.append(cell("v6_glambda_target_floor_1em3", target_lam_floor=1e-3))

    return methods


def _pdaps_v7_grid(log_level="INFO"):
    """
    v7 P-DAPS ablation around the best v6 single-axis settings.

    Base combines gamma=1.0, warm_fraction=0.8, and target_lam_floor=1e-3.
    The grid keeps one-axis deltas around that base, with DAPS and the v6
    glambda anchor retained for cross-grid context.
    """
    methods = []
    methods.append({
        "method": "DAPS",
        "params": {"lr": 1e-5},
        "algorithm": {
            "_target_": "algo.daps.DAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 100, "lr": 1e-5,
                            "tau": DAPS_TAU, "lr_min_ratio": 0.01},
        },
    })

    v6_anchor = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.2,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=0.0,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )
    methods.append(pdaps_entry(**v6_anchor, label_suffix="v6_anchor_glambda"))

    common = dict(v6_anchor)
    common.update(
        gamma=1.0,
        warm_fraction=0.8,
        target_lam_floor=1e-3,
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        return pdaps_entry(**cfg, label_suffix=label_suffix)

    # Base and winner-combination sanity checks.
    methods.append(cell("v7_base"))
    methods.append(cell("v7_base_gamma0p5", gamma=0.5))
    methods.append(cell("v7_base_warm0p2", warm_fraction=0.2))

    # Anchor-weight sweep.
    methods.append(cell("v7_target_floor_0", target_lam_floor=0.0))
    methods.append(cell("v7_target_floor_1em2", target_lam_floor=1e-2))

    # Inner-gate sweep, including gates below the v6 default.
    methods.append(cell("v7_inner_sigma2", inner_sigma_max=2.0))
    methods.append(cell("v7_inner_sigma3", inner_sigma_max=3.0))
    methods.append(cell("v7_inner_sigma10", inner_sigma_max=10.0))

    # Mechanism cells now that the anchor term is no longer effectively zero.
    methods.append(cell("v7_reanchor25", tweedie_reanchor_every=25, reanchor_blend_beta=0.1))
    methods.append(cell("v7_reanchor50", tweedie_reanchor_every=50, reanchor_blend_beta=0.1))
    methods.append(cell("v7_cgsense_init", warm_init_strategy="cgsense"))

    # Efficiency frontier around the gamma=1.0 base.
    methods.append(cell("v7_outer100_lgvd50",
                        annealing_override={"num_steps": 100}, lgvd_num_steps=50))
    methods.append(cell("v7_outer150_lgvd75",
                        annealing_override={"num_steps": 150}, lgvd_num_steps=75))

    # Solve-floor sweep on the v7 base.
    methods.append(cell("v7_solve_floor1", solve_lam_floor=1.0, noise_lam_floor=1.0))
    methods.append(cell("v7_solve_floor5", solve_lam_floor=5.0, noise_lam_floor=5.0))

    # New axes not covered by v6.
    methods.append(cell("v7_edm_project_post", edm_project_post=True))
    methods.append(cell("v7_inner_gate_residual",
                        inner_gate_mode="residual", residual_threshold=0.3))
    methods.append(cell("v7_precond_laplacian",
                        precond_mode="laplacian", noise_rhs_mode="matched"))

    return methods


def _pdaps_v8a_grid(log_level="INFO"):
    """
    v8a P-DAPS regime map.

    Maps the gamma x Langevin-noise-temperature x noise-subspace surface around
    the v7 SSIM-best operating point.  This preset intentionally omits DAPS:
    v8a is a P-DAPS mechanism audit, not a broad algorithm comparison.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    # Drift-only anchors.  noise_mode is irrelevant here because the noise
    # solve is gated off at noise_tau=0.
    methods.append(cell("v8a_drift_g0p5", gamma=0.5, noise_tau=0.0))
    methods.append(cell("v8a_drift_g1p0", gamma=1.0, noise_tau=0.0))

    # Core regime map: turn Langevin noise on and test the measured-subspace
    # ablation against full image-space noise, including the null component.
    tau_labels = [(0.025, "0p025"), (0.05, "0p05"), (0.1, "0p1")]
    mode_labels = [("range_only", "range"), ("full", "full")]
    for gamma, gamma_label in ((0.5, "0p5"), (1.0, "1p0")):
        for noise_tau, tau_label in tau_labels:
            for noise_mode, mode_label in mode_labels:
                methods.append(cell(
                    f"v8a_g{gamma_label}_nt{tau_label}_{mode_label}",
                    gamma=gamma,
                    noise_tau=noise_tau,
                    noise_mode=noise_mode,
                ))

    # Upper-edge range-only probes.  Full noise at this temperature is left to
    # follow-up if the lower full-noise surface warrants it.
    methods.append(cell("v8a_g0p5_nt0p2_range", gamma=0.5,
                        noise_tau=0.2, noise_mode="range_only"))
    methods.append(cell("v8a_g1p0_nt0p2_range", gamma=1.0,
                        noise_tau=0.2, noise_mode="range_only"))

    # Current-code bridge to the v6 inherited deterministic base.
    methods.append(cell("v6_anchor_glambda", gamma=0.5, warm_fraction=0.2,
                        target_lam_floor=0.0, noise_tau=0.0,
                        noise_mode="range_only"))

    return methods


def _pdaps_v8b_grid(log_level="INFO"):
    """
    v8b P-DAPS mechanism-audit grid.

    Pinned to the v8a operating point, this preset audits whether individual
    noise, gate, reanchor, floor, preconditioner, and projection mechanisms are
    functional. It intentionally excludes the v8a drift reference; compare
    against the published v8a_drift_g0p5 metrics instead of re-running it.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    # Block A: rescue full image-space noise by lowering the noise lambda floor.
    tau_labels = [(0.005, "0p005"), (0.01, "0p01"),
                  (0.025, "0p025"), (0.05, "0p05")]
    floor_labels = [(3.0, "3p0"), (1.0, "1p0"),
                    (0.3, "0p3"), (0.1, "0p1")]
    for noise_tau, tau_label in tau_labels:
        for noise_lam_floor, floor_label in floor_labels:
            methods.append(cell(
                f"v8b_A_full_nt{tau_label}_lnf{floor_label}",
                noise_tau=noise_tau,
                noise_mode="full",
                noise_lam_floor=noise_lam_floor,
            ))

    # Block B: compound inner gate residual-threshold sweep under range noise.
    for residual_threshold, rt_label in (
        (0.50, "0p50"),
        (0.30, "0p30"),
        (0.20, "0p20"),
        (0.10, "0p10"),
        (0.08, "0p08"),
        (0.05, "0p05"),
    ):
        methods.append(cell(
            f"v8b_B_compound_rt{rt_label}_nt0p05_range",
            noise_tau=0.05,
            noise_mode="range_only",
            inner_gate_mode="compound",
            residual_threshold=residual_threshold,
        ))

    # Block C: Tweedie reanchor beta sweep plus drift and rescued-full controls.
    for beta, beta_label in (
        (0.10, "0p10"),
        (0.25, "0p25"),
        (0.50, "0p50"),
        (1.00, "1p0"),
    ):
        methods.append(cell(
            f"v8b_C_reanchor25_b{beta_label}_nt0p05_range",
            noise_tau=0.05,
            noise_mode="range_only",
            tweedie_reanchor_every=25,
            reanchor_blend_beta=beta,
        ))
    methods.append(cell(
        "v8b_C_reanchor25_b0p50_drift",
        noise_tau=0.0,
        noise_mode="range_only",
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.50,
    ))
    methods.append(cell(
        "v8b_C_reanchor25_b0p50_nt0p025_full_lnf0p3",
        noise_tau=0.025,
        noise_mode="full",
        noise_lam_floor=0.3,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.50,
    ))

    # Block D: lambda-floor interactions under range noise.
    methods.append(cell(
        "v8b_D_tf1em4_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        target_lam_floor=1e-4,
    ))
    methods.append(cell(
        "v8b_D_tf1em2_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        target_lam_floor=1e-2,
    ))
    methods.append(cell(
        "v8b_D_sf1p0_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        solve_lam_floor=1.0,
    ))

    # Block E: non-standard preconditioners under range and rescued-full noise.
    for label_prefix, precond_mode in (
        ("lap", "laplacian"),
        ("lapnull", "laplacian_null"),
        ("mask", "mask_split"),
    ):
        methods.append(cell(
            f"v8b_E_{label_prefix}_nt0p05_range",
            noise_tau=0.05,
            noise_mode="range_only",
            precond_mode=precond_mode,
            noise_rhs_mode="matched",
        ))
    for label_prefix, precond_mode in (
        ("lap", "laplacian"),
        ("lapnull", "laplacian_null"),
        ("mask", "mask_split"),
    ):
        methods.append(cell(
            f"v8b_E_{label_prefix}_nt0p025_full_lnf0p3",
            noise_tau=0.025,
            noise_mode="full",
            noise_lam_floor=0.3,
            precond_mode=precond_mode,
            noise_rhs_mode="matched",
        ))

    # Block F: under-exercised warm, mid-project, and EDM post-project knobs.
    methods.append(cell(
        "v8b_F_adaptive_warm_nt0p05_range",
        warm_mode="adaptive",
        noise_tau=0.05,
        noise_mode="range_only",
    ))
    methods.append(cell(
        "v8b_F_midproj25_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        mid_inner_project_every=25,
    ))
    methods.append(cell(
        "v8b_F_edmpost_drift_g0p5",
        noise_tau=0.0,
        noise_mode="range_only",
        edm_project_post=True,
    ))
    methods.append(cell(
        "v8b_F_edmpost_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        edm_project_post=True,
    ))
    methods.append(cell(
        "v8b_F_adaptive_warm_nt0p025_full_lnf0p3",
        warm_mode="adaptive",
        noise_tau=0.025,
        noise_mode="full",
        noise_lam_floor=0.3,
    ))

    # Block G: pure null-space and pure image-space completeness probes.
    for mode_label, noise_mode in (("nullonly", "null_only"),
                                  ("imageonly", "image_only")):
        for noise_tau, tau_label, noise_lam_floor, floor_label in (
            (0.05, "0p05", 3.0, "3p0"),
            (0.05, "0p05", 0.3, "0p3"),
            (0.05, "0p05", 0.1, "0p1"),
            (0.01, "0p01", 3.0, "3p0"),
        ):
            methods.append(cell(
                f"v8b_G_{mode_label}_nt{tau_label}_lnf{floor_label}",
                noise_tau=noise_tau,
                noise_mode=noise_mode,
                noise_lam_floor=noise_lam_floor,
            ))

    return methods


def _pdaps_v8c_grid(log_level="INFO"):
    """
    v8c P-DAPS rescue-surface grid.

    Pinned to the v8a operating point, this preset maps early-gated full/null
    noise rescue while carrying forward the v8b lower-but-live controls under
    the fixed mask_split and gate-diagnostic code.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    # Block A: rescue surface for previously catastrophic full/null noise.
    full_tau_labels = (
        (0.0005, "0p0005"),
        (0.001, "0p001"),
        (0.0025, "0p0025"),
        (0.005, "0p005"),
        (0.01, "0p01"),
    )
    sigma_min_labels = (
        (0.5, "0p5"),
        (1.0, "1p0"),
        (2.0, "2p0"),
        (3.0, "3p0"),
    )
    for noise_tau, tau_label in full_tau_labels:
        for noise_sigma_min, sigma_label in sigma_min_labels:
            methods.append(cell(
                f"v8c_A_full_nt{tau_label}_smin{sigma_label}",
                noise_tau=noise_tau,
                noise_mode="full",
                noise_gate_mode="sigma_early",
                noise_sigma_min=noise_sigma_min,
            ))

    for noise_tau, tau_label in ((0.001, "0p001"), (0.005, "0p005")):
        for noise_sigma_min, sigma_label in sigma_min_labels[:3]:
            methods.append(cell(
                f"v8c_A_null_nt{tau_label}_smin{sigma_label}",
                noise_tau=noise_tau,
                noise_mode="null_only",
                noise_gate_mode="sigma_early",
                noise_sigma_min=noise_sigma_min,
            ))

    # Block B: range-only tau extension under the safe noise family.
    for noise_tau, tau_label in (
        (0.005, "0p005"),
        (0.025, "0p025"),
        (0.05, "0p05"),
        (0.10, "0p10"),
    ):
        methods.append(cell(
            f"v8c_B_range_nt{tau_label}",
            noise_tau=noise_tau,
            noise_mode="range_only",
        ))

    # Block C: high-beta reanchor checks on range-only noise.
    for beta, beta_label in ((0.50, "0p5"), (1.00, "1p0")):
        methods.append(cell(
            f"v8c_C_reanchor25_b{beta_label}_nt0p05_range",
            noise_tau=0.05,
            noise_mode="range_only",
            tweedie_reanchor_every=25,
            reanchor_blend_beta=beta,
        ))

    # Block D: lower-but-live knobs and fixed-bug controls.
    methods.append(cell(
        "v8c_D_lapnull_nt0p05_range",
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_mode="range_only",
    ))
    methods.append(cell(
        "v8c_D_adaptive_warm_nt0p05_range",
        warm_mode="adaptive",
        noise_tau=0.05,
        noise_mode="range_only",
    ))
    methods.append(cell(
        "v8c_D_mask_split_nt0p05_range",
        precond_mode="mask_split",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_mode="range_only",
    ))
    methods.append(cell(
        "v8c_D_late_resid0p10_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "v8c_D_lapnull_late_resid0p10_nt0p05_range",
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_mode="range_only",
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "v8c_D_lnf0p1_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        noise_lam_floor=0.1,
    ))
    methods.append(cell(
        "v8c_D_lnf1p0_nt0p05_range",
        noise_tau=0.05,
        noise_mode="range_only",
        noise_lam_floor=1.0,
    ))

    # Block E: residual-based early gates.
    methods.append(cell(
        "v8c_E_full_nt0p005_residearly_rmin0p20",
        noise_tau=0.005,
        noise_mode="full",
        noise_gate_mode="residual_early",
        noise_residual_min=0.20,
    ))
    methods.append(cell(
        "v8c_E_full_nt0p005_compoundearly_smin1p0_rmin0p20",
        noise_tau=0.005,
        noise_mode="full",
        noise_gate_mode="compound_early",
        noise_sigma_min=1.0,
        noise_residual_min=0.20,
    ))
    methods.append(cell(
        "v8c_E_null_nt0p005_residearly_rmin0p20",
        noise_tau=0.005,
        noise_mode="null_only",
        noise_gate_mode="residual_early",
        noise_residual_min=0.20,
    ))

    return methods


def _pdaps_v8d_grid(log_level="INFO"):
    """
    v8d P-DAPS single-slice surface map.

    Covers early-termination, range/residual/sigma-early noise surfaces,
    warm-init x LGVD budget, live gamma/reanchor corners, and distinct
    mask_split rescue attempts. Pinned to the v8a/v8c operating point.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    # Anchors: keep the v8a/v8c operating points visible in this run.
    methods.append(cell("v8d_anchor_outer200_range_nt0p05", noise_tau=0.05))
    methods.append(cell("v8d_anchor_outer200_drift"))
    methods.append(cell(
        "v8d_anchor_outer150_range_nt0p05",
        annealing_override={"num_steps": 150},
        noise_tau=0.05,
    ))
    methods.append(cell(
        "v8d_anchor_outer150_drift",
        annealing_override={"num_steps": 150},
    ))

    # Block A: early-termination cliff. Outer-150 points are the anchors above.
    for outer, outer_label in ((175, "175"), (125, "125"), (100, "100"), (75, "75")):
        methods.append(cell(
            f"v8d_A_outer{outer_label}_range_nt0p05",
            annealing_override={"num_steps": outer},
            noise_tau=0.05,
        ))
    for outer, outer_label in ((125, "125"), (100, "100")):
        methods.append(cell(
            f"v8d_A_outer{outer_label}_drift",
            annealing_override={"num_steps": outer},
        ))

    # Block B: range-only tau sweep at the outer-150 budget.
    for noise_tau, tau_label in ((0.005, "0p005"), (0.025, "0p025"), (0.10, "0p10")):
        methods.append(cell(
            f"v8d_B_outer150_range_nt{tau_label}",
            annealing_override={"num_steps": 150},
            noise_tau=noise_tau,
        ))

    # Block C: residual-gated range noise.
    for noise_tau, tau_label, threshold, thresh_label in (
        (0.05, "0p05", 0.05, "0p05"),
        (0.05, "0p05", 0.10, "0p10"),
        (0.05, "0p05", 0.20, "0p20"),
        (0.10, "0p10", 0.10, "0p10"),
        (0.10, "0p10", 0.20, "0p20"),
    ):
        methods.append(cell(
            f"v8d_C_resid{thresh_label}_nt{tau_label}_range",
            noise_tau=noise_tau,
            noise_gate_mode="residual",
            noise_residual_threshold=threshold,
        ))

    # Block D: compact sigma_early rescue surface.
    for noise_tau, tau_label, noise_sigma_min, sigma_label in (
        (0.0005, "0p0005", 0.5, "0p5"),
        (0.001, "0p001", 0.5, "0p5"),
        (0.0025, "0p0025", 0.5, "0p5"),
        (0.001, "0p001", 1.0, "1p0"),
    ):
        methods.append(cell(
            f"v8d_D_full_nt{tau_label}_smin{sigma_label}",
            noise_tau=noise_tau,
            noise_mode="full",
            noise_gate_mode="sigma_early",
            noise_sigma_min=noise_sigma_min,
        ))
    for noise_tau, tau_label, noise_sigma_min, sigma_label in (
        (0.001, "0p001", 0.5, "0p5"),
        (0.005, "0p005", 0.5, "0p5"),
        (0.001, "0p001", 1.0, "1p0"),
    ):
        methods.append(cell(
            f"v8d_D_null_nt{tau_label}_smin{sigma_label}",
            noise_tau=noise_tau,
            noise_mode="null_only",
            noise_gate_mode="sigma_early",
            noise_sigma_min=noise_sigma_min,
        ))

    # Block E: warm init x LGVD budget, drift only.
    for lgvd_num_steps in (100, 50, 25):
        for init_label, warm_init_strategy in (("cgsense", "cgsense"), ("zerofilled", "zero_filled")):
            methods.append(cell(
                f"v8d_E_lgvd{lgvd_num_steps}_{init_label}_drift",
                lgvd_num_steps=lgvd_num_steps,
                warm_init_strategy=warm_init_strategy,
            ))

    # Block F: gamma x warm live corners. warm_fraction=0.8 is inherited.
    methods.append(cell("v8d_F_g0p75_warm0p8_drift", gamma=0.75))
    methods.append(cell("v8d_F_g0p75_warm0p8_nt0p05_range", gamma=0.75, noise_tau=0.05))
    methods.append(cell("v8d_F_g0p75_warm0p8_nt0p10_range", gamma=0.75, noise_tau=0.10))
    methods.append(cell("v8d_F_g1p0_warm0p8_drift", gamma=1.0))
    methods.append(cell("v8d_F_g1p0_warm0p8_nt0p05_range", gamma=1.0, noise_tau=0.05))

    # Block G: reanchor alive-axis probes outside the range nt=0.05 factorial.
    methods.append(cell(
        "v8d_G_reanchor10_b0p5_drift",
        tweedie_reanchor_every=10,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "v8d_G_reanchor50_b0p5_drift",
        tweedie_reanchor_every=50,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "v8d_G_reanchor25_b0p5_nt0p10_range",
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
        noise_tau=0.10,
    ))

    # Block H: mask_split rescue attempts available without a step cap.
    methods.append(cell(
        "v8d_H_mask_split_drift",
        precond_mode="mask_split",
        noise_rhs_mode="matched",
    ))
    methods.append(cell(
        "v8d_H_mask_split_nt0p05_range_resid0p10",
        precond_mode="mask_split",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "v8d_H_mask_split_nt0p05_range_resid0p20",
        precond_mode="mask_split",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.20,
    ))
    methods.append(cell(
        "v8d_H_mask_split_nt0p05_range_sigmaearly_smin1p0",
        precond_mode="mask_split",
        noise_rhs_mode="matched",
        noise_tau=0.05,
        noise_gate_mode="sigma_early",
        noise_sigma_min=1.0,
    ))

    # Block I: reanchor cadence x effective time-constant surface.
    for label_suffix, every, beta in (
        ("v8d_I_sanity_K25_b0p5_nt0p05_range", 25, 0.5),
        ("v8d_I_tau100_K5_b0p05_nt0p05_range", 5, 0.05),
        ("v8d_I_tau100_K10_b0p1_nt0p05_range", 10, 0.1),
        ("v8d_I_tau100_K25_b0p25_nt0p05_range", 25, 0.25),
        ("v8d_I_tau50_K10_b0p2_nt0p05_range", 10, 0.2),
        ("v8d_I_tau25_K5_b0p2_nt0p05_range", 5, 0.2),
    ):
        methods.append(cell(
            label_suffix,
            noise_tau=0.05,
            tweedie_reanchor_every=every,
            reanchor_blend_beta=beta,
        ))

    return methods


def _pdaps_v8e_grid(log_level="INFO"):
    """
    v8e terminal-sigma ablation.

    Clean cells use sigma_stop_truncate to slice the reference v8d
    200-step schedule. Lite cells use annealing sigma_min overrides to test
    denser re-spacing over the kept sigma range.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    clean_cells = (
        ("v8e_clean_lgvd100_stop0p10", 100, 0.10),
        ("v8e_clean_lgvd100_stop0p115", 100, 0.115),
        ("v8e_clean_lgvd100_stop0p14", 100, 0.14),
        ("v8e_clean_lgvd100_stop0p17", 100, 0.17),
        ("v8e_clean_lgvd100_stop0p25", 100, 0.25),
        ("v8e_clean_lgvd50_stop0p10", 50, 0.10),
        ("v8e_clean_lgvd50_stop0p115", 50, 0.115),
        ("v8e_clean_lgvd50_stop0p14", 50, 0.14),
        ("v8e_clean_lgvd50_stop0p17", 50, 0.17),
        ("v8e_clean_lgvd50_stop0p25", 50, 0.25),
        ("v8e_clean_lgvd25_stop0p38", 25, 0.38),
    )
    for label_suffix, lgvd_num_steps, sigma_stop in clean_cells:
        methods.append(cell(
            label_suffix,
            lgvd_num_steps=lgvd_num_steps,
            sigma_stop_truncate=sigma_stop,
        ))

    lite_cells = (
        ("v8e_lite_lgvd100_smin0p14", 100, 0.14),
        ("v8e_lite_lgvd100_smin0p17", 100, 0.17),
        ("v8e_lite_lgvd50_smin0p17", 50, 0.17),
        ("v8e_lite_lgvd50_smin0p25", 50, 0.25),
        ("v8e_lite_lgvd25_smin0p38", 25, 0.38),
    )
    for label_suffix, lgvd_num_steps, sigma_min in lite_cells:
        methods.append(cell(
            label_suffix,
            lgvd_num_steps=lgvd_num_steps,
            annealing_override={"sigma_min": sigma_min},
        ))

    return methods


def _pdaps_v8f_grid(log_level="INFO"):
    """
    v8f post-fix terminal-sigma ablation.

    Re-runs only the clean sigma_stop_truncate cells invalidated by the v8e
    return-value bug. Lite cells are intentionally omitted; v8e lite rows did
    not hit the truncation branch and remain valid comparison anchors.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    clean_cells = (
        ("v8f_clean_lgvd100_stop0p10", 100, 0.10),
        ("v8f_clean_lgvd100_stop0p115", 100, 0.115),
        ("v8f_clean_lgvd100_stop0p14", 100, 0.14),
        ("v8f_clean_lgvd100_stop0p17", 100, 0.17),
        ("v8f_clean_lgvd100_stop0p25", 100, 0.25),
        ("v8f_clean_lgvd50_stop0p10", 50, 0.10),
        ("v8f_clean_lgvd50_stop0p14", 50, 0.14),
        ("v8f_clean_lgvd50_stop0p17", 50, 0.17),
        ("v8f_clean_lgvd50_stop0p25", 50, 0.25),
        ("v8f_clean_lgvd25_stop0p38", 25, 0.38),
    )
    for label_suffix, lgvd_num_steps, sigma_stop in clean_cells:
        methods.append(cell(
            label_suffix,
            lgvd_num_steps=lgvd_num_steps,
            sigma_stop_truncate=sigma_stop,
        ))

    return methods


def _pdaps_working_grid(log_level="INFO"):
    """
    Small validation-tuning grid for the working P-DAPS core.

    This intentionally exposes only terminal sigma and inner-step budget.
    Use older v8* presets for diagnostics, not for routine tuning.
    """
    return [
        pdaps_working_entry("working_quality_full_lgvd100_stop0p10", 100, 0.10, log_level=log_level),
        pdaps_working_entry("working_quality_cut_lgvd100_stop0p14", 100, 0.14, log_level=log_level),
        pdaps_working_entry("working_balanced_lgvd50_stop0p17", 50, 0.17, log_level=log_level),
        pdaps_working_entry("working_balanced_lgvd50_stop0p25", 50, 0.25, log_level=log_level),
        pdaps_working_entry("working_fast_lgvd25_stop0p38", 25, 0.38, log_level=log_level),
    ]


def _pdaps_prelaunch_base_kwargs(base="v8f"):
    """
    Shared P-DAPS pre-launch operating points.

    `v8f` is the current-safe base. Other bases are promoted only if C0
    supports the corresponding lock change.
    """
    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=50,
        log_level="INFO",
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
        sigma_stop_truncate=0.17,
    )
    if base == "v8f":
        return common
    if base == "floor0":
        common["target_lam_floor"] = 0.0
        return common
    if base == "inf":
        common["inner_sigma_max"] = 1e9
        return common
    if base == "floor0_inf":
        common["target_lam_floor"] = 0.0
        common["inner_sigma_max"] = 1e9
        return common
    raise ValueError(f"Unknown prelaunch base: {base}")


def _pdaps_prelaunch_cell(label_suffix, log_level="INFO", base="v8f", **overrides):
    cfg = _pdaps_prelaunch_base_kwargs(base=base)
    cfg["log_level"] = log_level
    cfg.update(overrides)
    entry = pdaps_entry(**cfg, label_suffix=label_suffix)
    entry["params"]["noise_mode"] = cfg["noise_mode"]
    return entry


def _pdaps_prelaunch_core_cell(label_suffix, log_level="INFO", base="v8f", **overrides):
    cfg = _pdaps_prelaunch_base_kwargs(base=base)
    cfg["log_level"] = log_level
    cfg.update(overrides)
    return pdaps_core_entry(
        label_suffix=label_suffix,
        lgvd_num_steps=cfg["lgvd_num_steps"],
        sigma_stop_truncate=cfg["sigma_stop_truncate"],
        gamma=cfg["gamma"],
        warm_fraction=cfg["warm_fraction"],
        inner_sigma_max=cfg["inner_sigma_max"],
        solve_lam_floor=cfg["solve_lam_floor"],
        lr_min_ratio=cfg.get("lr_min_ratio", 0.01),
        tau=cfg["tau"],
        log_level=log_level,
    )


def _pdaps_prelaunch_c0_grid(log_level="INFO"):
    """
    Frozen-choice challenge around the current v8f-safe balanced anchor.

    Gate on paired quality, stability, and runtime. If a challenge wins, use the
    corresponding promoted A preset before spending the larger A/B budget.
    """
    return [
        _pdaps_prelaunch_cell("c0_anchor_v8f_lgvd50_stop0p17", log_level=log_level),
        _pdaps_prelaunch_cell(
            "c0_gamma_constant",
            log_level=log_level,
            gamma_schedule="constant",
        ),
        _pdaps_prelaunch_cell(
            "c0_target_floor0",
            log_level=log_level,
            target_lam_floor=0.0,
        ),
        _pdaps_prelaunch_cell(
            "c0_inner_sigma_inf",
            log_level=log_level,
            inner_sigma_max=1e9,
        ),
        _pdaps_prelaunch_cell(
            "c0_noise_tau0p025_range",
            log_level=log_level,
            noise_tau=0.025,
            noise_mode="range_only",
        ),
        _pdaps_prelaunch_cell(
            "c0_solve_floor1",
            log_level=log_level,
            solve_lam_floor=1.0,
        ),
        _pdaps_prelaunch_cell(
            "c0_solve_floor5",
            log_level=log_level,
            solve_lam_floor=5.0,
        ),
    ]


def _pdaps_prelaunch_a_grid(base="v8f", log_level="INFO"):
    """
    Runtime surface: inner budget x terminal stop on the C0-selected base.
    """
    methods = []
    stops = (
        ("full", None),
        ("stop0p10", 0.10),
        ("stop0p14", 0.14),
        ("stop0p17", 0.17),
        ("stop0p25", 0.25),
        ("stop0p38", 0.38),
    )
    for lgvd_num_steps in (25, 50, 100):
        for stop_label, sigma_stop in stops:
            methods.append(_pdaps_prelaunch_core_cell(
                f"a_{base}_lgvd{lgvd_num_steps}_{stop_label}",
                base=base,
                log_level=log_level,
                lgvd_num_steps=lgvd_num_steps,
                sigma_stop_truncate=sigma_stop,
            ))
    return methods


def _pdaps_prelaunch_anchor_defs(anchors):
    if anchors == "balanced":
        return (("balanced_lgvd50_stop0p25", 50, 0.25),)
    if anchors == "balfast":
        return (
            ("balanced_lgvd50_stop0p25", 50, 0.25),
            ("fast_lgvd25_stop0p38", 25, 0.38),
        )
    raise ValueError(f"Unknown prelaunch anchors: {anchors}")


def _pdaps_prelaunch_b_grid(base="v8f", anchors="balfast", log_level="INFO"):
    """
    One-axis tunable-knob audit around one or two A-selected anchors.

    Each anchor contributes six cells: anchor plus warm/gamma one-axis
    alternatives. Target lambda flooring was deleted from the core after C0/B.
    """
    methods = []
    for anchor_label, lgvd_num_steps, sigma_stop in _pdaps_prelaunch_anchor_defs(anchors):
        prefix = f"b_{base}_{anchor_label}"
        anchor = dict(
            base=base,
            log_level=log_level,
            lgvd_num_steps=lgvd_num_steps,
            sigma_stop_truncate=sigma_stop,
        )
        methods.append(_pdaps_prelaunch_core_cell(f"{prefix}_anchor", **anchor))
        for warm_fraction in (0.0, 0.2, 0.5):
            methods.append(_pdaps_prelaunch_core_cell(
                f"{prefix}_warm{str(warm_fraction).replace('.', 'p')}",
                warm_fraction=warm_fraction,
                **anchor,
            ))
        for gamma in (0.75, 1.0):
            methods.append(_pdaps_prelaunch_core_cell(
                f"{prefix}_gamma{str(gamma).replace('.', 'p')}",
                gamma=gamma,
                **anchor,
            ))
    return methods


def _prelaunch_baseline_grid(log_level="INFO"):
    """
    Paired validation calibration baselines. These are not final test claims.
    """
    return [
        {
            "method": "DPS[prelaunch_calib_g1]",
            "params": {"guidance_scale": 1.0},
            "algorithm": {
                "_target_": "algo.dps.DPS",
                "diffusion_scheduler_config": DPS_SCHEDULER,
                "guidance_scale": 1.0,
            },
        },
        {
            "method": "DAPS[prelaunch_calib_lr3em6]",
            "params": {"lr": 3e-6, "tau": DAPS_TAU, "lr_min_ratio": 0.01},
            "algorithm": {
                "_target_": "algo.daps.DAPS",
                "annealing_scheduler_config": ANNEALING,
                "diffusion_scheduler_config": REVERSE_ODE,
                "lgvd_config": {
                    "num_steps": 100,
                    "lr": 3e-6,
                    "tau": DAPS_TAU,
                    "lr_min_ratio": 0.01,
                },
            },
        },
        {
            "method": "pULA[prelaunch_calib_g0p5]",
            "params": {"gamma": 0.5},
            "algorithm": {
                "_target_": "algo.pula.pULA",
                "noise_scheduler_config": PULA_SCHEDULER,
                "K": 4,
                "gamma": 0.5,
                "cg_iter": 10,
                "log_level": log_level,
            },
        },
    ]


def _pdaps_prelaunch_lrmin_grid(log_level="INFO"):
    """
    Tiny check for the inherited DAPS terminal gamma decay.

    Runs the selected balanced production point with lr_min_ratio=0.01 versus
    no terminal decay. Intended scale: one file, one slice, one seed, accel 4/8.
    """
    return [
        _pdaps_prelaunch_core_cell(
            "lrmin_balanced_lgvd50_stop0p25_ratio0p01",
            log_level=log_level,
            lgvd_num_steps=50,
            sigma_stop_truncate=0.25,
            gamma=0.5,
            lr_min_ratio=0.01,
        ),
        _pdaps_prelaunch_core_cell(
            "lrmin_balanced_lgvd50_stop0p25_ratio1p0",
            log_level=log_level,
            lgvd_num_steps=50,
            sigma_stop_truncate=0.25,
            gamma=0.5,
            lr_min_ratio=1.0,
        ),
    ]


def _pdaps_check_abandoned_grid(log_level="INFO"):
    """
    Post-v8b audit grid for live knobs and crosses not covered by v8c.

    Pinned to the v8a operating point.  The v8a drift reference is cited, not
    re-run; compare drift cells against v8a_drift_g0p5.
    """
    methods = []

    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.0,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    # Block H: gamma shape knobs never landed in prior grids.
    methods.append(cell("H_gfloor0p05", gamma_floor=0.05))
    methods.append(cell("H_gceil0p5_g0p5", gamma_ceiling=0.5))
    methods.append(cell("H_gceil0p5_g1p0", gamma=1.0, gamma_ceiling=0.5))
    methods.append(cell("H_gceil1p0_g1p0", gamma=1.0, gamma_ceiling=1.0))
    methods.append(cell("H_sqrtlambda_g0p5", gamma_schedule="sqrt_lambda"))
    methods.append(cell("H_sqrtlambda_g1p0", gamma=1.0, gamma_schedule="sqrt_lambda"))

    # Block I: warm-init alternatives.
    methods.append(cell("I_zerofilled_warm0p8_drift", warm_init_strategy="zero_filled"))
    methods.append(cell(
        "I_zerofilled_warm0p5_drift",
        warm_fraction=0.5,
        warm_init_strategy="zero_filled",
    ))
    methods.append(cell(
        "I_zerofilled_warm0p8_nt0p05_range",
        warm_init_strategy="zero_filled",
        noise_tau=0.05,
    ))
    methods.append(cell("I_cgsense_diag", warm_init_strategy="cgsense"))

    # Block J: late-only noise gates with noise-only semantics.
    methods.append(cell(
        "J_noisegate_residual_0p20_nt0p05_range",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.20,
    ))
    methods.append(cell(
        "J_noisegate_residual_0p10_nt0p05_range",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "J_noisegate_residual_0p05_nt0p05_range",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.05,
    ))
    methods.append(cell(
        "J_noisegate_compound_resid0p10_nt0p05_range",
        noise_tau=0.05,
        noise_gate_mode="compound",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "J_noisegate_residual_0p10_nt0p1_range",
        noise_tau=0.10,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))

    # Block K: mid-inner project at unexplored strides / under drift.
    methods.append(cell("K_midproj50_drift", mid_inner_project_every=50))
    methods.append(cell("K_midproj10_drift", mid_inner_project_every=10))
    methods.append(cell("K_midproj25_drift", mid_inner_project_every=25))
    methods.append(cell(
        "K_midproj50_nt0p05_range",
        noise_tau=0.05,
        mid_inner_project_every=50,
    ))

    # Block L: Tweedie reanchor at unexplored conditions.
    methods.append(cell(
        "L_reanchor50_b0p5_drift",
        tweedie_reanchor_every=50,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "L_reanchor10_b0p5_drift",
        tweedie_reanchor_every=10,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "L_reanchor25_b0p5_nt0p1_range",
        noise_tau=0.10,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "L_reanchor25_b0p5_adaptive_warm_nt0p05_range",
        warm_mode="adaptive",
        noise_tau=0.05,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))

    # Block M: gamma x warm cross corners.
    methods.append(cell("M_g0p5_warm0p5", warm_fraction=0.5))
    methods.append(cell("M_g1p0_warm0p5", gamma=1.0, warm_fraction=0.5))
    methods.append(cell("M_g0p75_warm0p8", gamma=0.75))
    methods.append(cell("M_g0p75_warm0p5", gamma=0.75, warm_fraction=0.5))

    # Block N: outer x LGVD frontier at v8a gamma=0.5.
    methods.append(cell(
        "N_outer100_lgvd50_g0p5",
        lgvd_num_steps=50,
        annealing_override={"num_steps": 100},
    ))
    methods.append(cell(
        "N_outer150_lgvd75_g0p5",
        lgvd_num_steps=75,
        annealing_override={"num_steps": 150},
    ))
    methods.append(cell(
        "N_outer150_lgvd100_g0p5",
        lgvd_num_steps=100,
        annealing_override={"num_steps": 150},
    ))

    # Block O: opened inner gate with capped per-step magnitude.
    methods.append(cell(
        "O_inner_sigma10_gceil0p5",
        inner_sigma_max=10.0,
        gamma_ceiling=0.5,
    ))
    methods.append(cell(
        "O_inner_sigma_nogate_gceil0p5",
        inner_sigma_max=1e9,
        gamma_ceiling=0.5,
    ))

    # Block P: Laplacian / Laplacian-null under unexplored knobs.
    methods.append(cell(
        "P_lapnull_drift",
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
    ))
    methods.append(cell(
        "P_lapnull_nt0p05_range_warm0p2",
        warm_fraction=0.2,
        noise_tau=0.05,
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
    ))
    methods.append(cell(
        "P_lapnull_nt0p05_range_gceil0p5",
        noise_tau=0.05,
        gamma_ceiling=0.5,
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
    ))
    methods.append(cell(
        "P_lapnull_pen100_matched_nt0p05_range",
        noise_tau=0.05,
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        penalty_scale=100.0,
    ))
    methods.append(cell(
        "P_lap_pen100_matched_drift",
        precond_mode="laplacian",
        noise_rhs_mode="matched",
        penalty_scale=100.0,
    ))

    # Block Q: high-information two-knob crosses.
    methods.append(cell(
        "Q_gceil0p5_g1p0_nt0p05_range",
        gamma=1.0,
        gamma_ceiling=0.5,
        noise_tau=0.05,
    ))
    methods.append(cell(
        "Q_noisegate_residual_0p10_reanchor25_b0p5_nt0p05_range",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "Q_adaptive_warm_noisegate_residual_0p10_nt0p05_range",
        warm_mode="adaptive",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "Q_gceil0p5_g1p0_noisegate_residual_0p10_nt0p05_range",
        gamma=1.0,
        gamma_ceiling=0.5,
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "Q_reanchor25_b0p5_gceil0p5_g1p0_drift",
        gamma=1.0,
        gamma_ceiling=0.5,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "Q_midproj25_gceil0p5_g1p0_drift",
        gamma=1.0,
        gamma_ceiling=0.5,
        mid_inner_project_every=25,
    ))
    methods.append(cell(
        "Q_adaptive_warm_reanchor25_b0p5_nt0p05_range",
        warm_mode="adaptive",
        noise_tau=0.05,
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "Q_lapnull_noisegate_residual_0p10_nt0p05_range",
        noise_tau=0.05,
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "Q_lapnull_reanchor25_b0p5_nt0p05_range",
        noise_tau=0.05,
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        tweedie_reanchor_every=25,
        reanchor_blend_beta=0.5,
    ))
    methods.append(cell(
        "Q_sqrtlambda_g0p5_nt0p05_range",
        noise_tau=0.05,
        gamma_schedule="sqrt_lambda",
    ))
    methods.append(cell(
        "Q_zerofilled_warm0p8_noisegate_residual_0p10_nt0p05_range",
        warm_init_strategy="zero_filled",
        noise_tau=0.05,
        noise_gate_mode="residual",
        noise_residual_threshold=0.10,
    ))
    methods.append(cell(
        "Q_gfloor0p05_inner_sigma10_drift",
        gamma_floor=0.05,
        inner_sigma_max=10.0,
    ))

    # Block R: penalty schedule / epsilon variants.
    methods.append(cell(
        "R_lapnull_pen_constant_mu1p0_matched_drift",
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        penalty_schedule="constant",
        penalty_scale=1.0,
    ))
    methods.append(cell(
        "R_lapnull_penalty_eps0p01_matched_drift",
        precond_mode="laplacian_null",
        noise_rhs_mode="matched",
        penalty_eps=0.01,
    ))
    methods.append(cell(
        "R_lap_pen100_matched_nt0p05_range",
        noise_tau=0.05,
        precond_mode="laplacian",
        noise_rhs_mode="matched",
        penalty_scale=100.0,
    ))

    return methods


def _pdaps_bugcheck_grid(log_level="INFO"):
    """Small post-v8b correctness checks for targeted single-slice reruns."""
    common = dict(
        method="P-DAPS",
        warm_mode="fixed",
        gamma=0.5,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        lgvd_num_steps=100,
        log_level=log_level,
        lam_floor=0.0,
        target_lam_floor=1e-3,
        solve_lam_floor=3.0,
        noise_lam_floor=3.0,
        noise_tau=0.05,
        noise_mode="range_only",
        precond_mode="standard",
        noise_rhs_mode="standard",
        tau=DAPS_TAU,
        gamma_schedule="lambda",
    )

    def cell(label_suffix, **overrides):
        cfg = dict(common)
        cfg.update(overrides)
        entry = pdaps_entry(**cfg, label_suffix=label_suffix)
        entry["params"]["noise_mode"] = cfg["noise_mode"]
        return entry

    return [
        cell(
            "bug_E_mask_fixed_nt0p05_range",
            precond_mode="mask_split",
            noise_rhs_mode="matched",
        ),
        cell(
            "bug_noisegate_resid0p10_nt0p05_range",
            noise_gate_mode="residual",
            noise_residual_threshold=0.10,
        ),
        cell(
            "bug_full_nt0p005_sigmaearly1p0",
            noise_tau=0.005,
            noise_mode="full",
            noise_gate_mode="sigma_early",
            noise_sigma_min=1.0,
        ),
    ]


def _cuda_mem_probe(tag, dump_top=0):
    """Leak probe: log live/reserved CUDA memory. If dump_top>0, also list the
    largest live CUDA tensors (shape/dtype) so we can name what is being held."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[MEMPROBE] {tag}: allocated={alloc:.3f}GB reserved={reserved:.3f}GB", flush=True)
    if dump_top > 0:
        tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensors.append((obj.element_size() * obj.nelement(), tuple(obj.shape), str(obj.dtype)))
            except Exception:
                continue
        tensors.sort(reverse=True)
        for nbytes, shape, dtype in tensors[:dump_top]:
            print(f"[MEMPROBE]   live {nbytes/1e6:.1f}MB {shape} {dtype}", flush=True)
        # Histogram by (shape, dtype): for a per-cell leak the *count* of duplicated
        # tensors is the smoking gun — it grows by a fixed amount each cell.
        hist = collections.Counter((shape, dtype) for _, shape, dtype in tensors)
        bytes_by_key = collections.defaultdict(float)
        for nbytes, shape, dtype in tensors:
            bytes_by_key[(shape, dtype)] += nbytes
        for (shape, dtype), count in sorted(hist.items(), key=lambda kv: -bytes_by_key[kv[0]])[:dump_top]:
            print(f"[MEMPROBE]   x{count:<4d} {bytes_by_key[(shape, dtype)]/1e6:8.1f}MB total {shape} {dtype}", flush=True)
        print(f"[MEMPROBE]   total live cuda tensors: {len(tensors)}", flush=True)


def load_model(args, device):
    net = hydra.utils.instantiate(OmegaConf.create(MODEL_CONFIG))
    ckpt = torch.load(Path(args.models_dir) / args.ckpt_name, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["ema"])
    return net.to(device).eval()


def make_forward_op(args, device):
    cfg = {
        "_target_": "inverse_problems.multi_coil_mri.MultiCoilMRI",
        "total_lines": args.image_size[1],
        "acceleration_ratio": args.acceleration,
        "pattern": args.pattern,
        "mask_seed": args.mask_seed,
        "device": str(device),
    }
    return hydra.utils.instantiate(OmegaConf.create(cfg))


def move_to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def run_one(entry, sample, sample_idx, split, net, args, out_dir,
            save_image=False, filename=None):
    device = next(net.parameters()).device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    row = {
        "split": split,
        "sample_idx": sample_idx,
        "filename": filename,
        "seed": args.seed,
        "acceleration": args.acceleration,
        "method": entry["method"],
        "params_json": json.dumps(entry["params"], sort_keys=True),
        "failed": False,
    }

    sanitized_method = entry["method"].replace("/", "_").replace("[", "_").replace("]", "_")
    log_dir = out_dir / "logs" / f"accel_{args.acceleration}"
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = getattr(args, "seeds", None)
    seed_suffix = f"_seed{args.seed}" if seeds and len(seeds) > 1 else ""
    log_path = log_dir / f"{sanitized_method}_{split}_{sample_idx}{seed_suffix}.log"
    trace_path = None
    if entry["algorithm"]["_target_"] in {"algo.pdaps.PDAPS", "algo.pdaps_core.PDAPS"} and args.log_level == "DEBUG":
        trace_dir = out_dir / "trajectories" / f"accel_{args.acceleration}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{sanitized_method}_{split}_{sample_idx}{seed_suffix}.npz"

    with open(log_path, "w") as log_fh:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_fh)), contextlib.redirect_stderr(_Tee(sys.stderr, log_fh)):
            start = time.perf_counter()
            try:
                forward_op = algo = data = observation = target = recon = None
                forward_op = make_forward_op(args, device)
                algo = hydra.utils.instantiate(OmegaConf.create(entry["algorithm"]), forward_op=forward_op, net=net)
                data = move_to_device(sample, device)
                data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                observation = forward_op(data)
                target = data["target"]

                if trace_path is not None:
                    recon = algo.inference(
                        observation,
                        num_samples=1,
                        verbose=args.verbose,
                        target=target,
                        trace_path=str(trace_path),
                    )
                    row["trajectory_npz"] = str(trace_path)
                else:
                    recon = algo.inference(observation, num_samples=1, verbose=args.verbose)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                metrics = compute_metrics_dict(forward_op, recon, target, observation)
                row.update(metrics)
                metric_names = ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed")
                if not metrics.get("finite", False) or not all(math.isfinite(float(metrics[name])) for name in metric_names):
                    row["failed"] = True
                    row["error"] = "nonfinite_reconstruction_or_metrics"
                row["catastrophe_warn"] = bool(
                    metrics.get("ssim", 1.0) < WARN_SSIM_FLOOR
                    or metrics.get("data_misfit_per_observed", 0.0) > WARN_DMO_RATIO * WARN_DMO_NORMAL
                )
                row["runtime_s"] = time.perf_counter() - start
                row["gate_stats_json"] = json.dumps(getattr(algo, "last_gate_stats", []))

                if save_image and not row["failed"]:
                    cfg = OmegaConf.create({
                        "algorithm": {"_target_": entry["algorithm"]["_target_"]},
                        "forward_op": {"acceleration_ratio": args.acceleration},
                    })
                    image_dir = out_dir / "figures" / f"accel_{args.acceleration}" / entry["method"].replace("/", "_")
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"{split}_{sample_idx}{seed_suffix}.png"
                    old_cwd = os.getcwd()
                    os.chdir(image_dir)
                    try:
                        visualize_recon(
                            forward_op,
                            forward_op.unnormalize(recon).cpu(),
                            forward_op.unnormalize(target).cpu(),
                            sample_idx,
                            cfg,
                            save_path=image_path.name,
                        )
                        row["figure_path"] = str(image_path)
                    finally:
                        os.chdir(old_cwd)
            except Exception as exc:
                row["failed"] = True
                row["catastrophe_warn"] = False
                row["error"] = repr(exc)
                row["traceback"] = traceback.format_exc()
                row["runtime_s"] = time.perf_counter() - start
            finally:
                del recon, target, observation, data, algo, forward_op
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    with contextlib.suppress(Exception):
                        torch.cuda.ipc_collect()

            status_str = f"[FAILED: {row.get('error', 'unknown')}]" if row["failed"] else f"PSNR={row.get('psnr', 0.0):.2f} SSIM={row.get('ssim', 0.0):.4f}"
            print(f"[{entry['method']}] {split.capitalize()} Image {sample_idx} ({filename}): "
                  f"{status_str} DataMisfit={row.get('data_misfit', 0.0):.3e} Time={row['runtime_s']:.1f}s")

    return row


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["params_json"])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, params_json), group in groups.items():
        ok = [row for row in group if not row.get("failed")]
        summary = {
            "method": method,
            "params_json": params_json,
            "n": len(group),
            "n_ok": len(ok),
            "failure_rate": 1.0 - len(ok) / max(1, len(group)),
            "catastrophe_warn_rate": (
                sum(1 for row in group if str(row.get("catastrophe_warn", False)).lower() == "true")
                / max(1, len(group))
            ),
        }
        for metric in ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed", "runtime_s"):
            vals = [float(row[metric]) for row in ok if metric in row]
            if vals:
                summary[f"{metric}_mean"] = sum(vals) / len(vals)
                summary[f"{metric}_std"] = torch.tensor(vals).std(unbiased=False).item()
        out.append(summary)
    return sorted(out, key=lambda row: (row["method"], -row.get("psnr_mean", -1e9)))


def expand_params_rows(rows):
    expanded = []
    for row in rows:
        out = dict(row)
        try:
            params = json.loads(row.get("params_json", "{}"))
        except json.JSONDecodeError:
            params = {}
        for key, value in params.items():
            out[f"params.{key}"] = value
        expanded.append(out)
    return expanded


def write_ablation_artifacts(out_dir, val_rows, test_rows):
    combined = []
    for row in val_rows:
        combined.append(dict(row))
    for row in test_rows:
        combined.append(dict(row))
    expanded = expand_params_rows(combined)
    write_csv(out_dir / "ablation_table.csv", expanded)

    failures = [
        row for row in expanded
        if row.get("failed") or str(row.get("catastrophe_warn", False)).lower() == "true"
    ]
    if failures:
        write_csv(out_dir / "failure_table.csv", failures)
    else:
        fields = sorted({key for row in expanded for key in row}) or ["failed"]
        with open(out_dir / "failure_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    test_summary = summarize(test_rows)
    write_csv(out_dir / "ablation_test_summary.csv", test_summary)
    print("Ablation test summary:")
    for row in test_summary:
        print(
            f"  {row['method']}: n_ok={row['n_ok']}/{row['n']} "
            f"PSNR={row.get('psnr_mean', float('nan')):.3f} "
            f"SSIM={row.get('ssim_mean', float('nan')):.4f} "
            f"NMSE={row.get('nmse_mean', float('nan')):.5f} "
            f"fail={row['failure_rate']:.3f}"
        )


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            if "failed" in row:
                row["failed"] = str(row["failed"]).lower() == "true"
            rows.append(row)
        return rows


def write_combined_acceleration_artifacts(out_dir, accelerations):
    val_rows = []
    test_rows = []
    for accel in accelerations:
        sub_dir = out_dir / f"accel_{accel}"
        val_rows.extend(read_csv_rows(sub_dir / "validation_raw.csv"))
        test_rows.extend(read_csv_rows(sub_dir / "test_raw.csv"))
    if val_rows or test_rows:
        write_ablation_artifacts(out_dir, val_rows, test_rows)


def select_best(validation_summary):
    """
    Pre-registered selection rule: maximize validation SSIM, then use PSNR and
    lower data misfit only as tie-breakers.
    """
    best = {}
    for row in validation_summary:
        if row["n_ok"] == 0:
            continue
        current = best.get(row["method"])
        candidate = (row.get("ssim_mean", -1e9), row.get("psnr_mean", -1e9), -row.get("data_misfit_mean", 1e9))
        if current is None or candidate > current[0]:
            best[row["method"]] = (candidate, row["params_json"])
    return {method: params_json for method, (_, params_json) in best.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny multi-coil knee MRI validation for DPS/DAPS/pULA/P-DAPS.")
    parser.add_argument("--models-dir", default="/dtu/blackhole/1d/214141/Thesis/models")
    parser.add_argument("--ckpt-name", default="MRI-knee.pt")
    parser.add_argument("--kspace-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val")
    parser.add_argument("--maps-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit")
    parser.add_argument("--filename", default="file1000196.h5")
    parser.add_argument("--filenames", nargs="+", default=None,
                        help="Multiple filenames for multi-patient runs. If given, "
                             "--val-slices and --test-slices are interpreted *per file*. "
                             "Overrides --filename.")
    parser.add_argument("--test-filenames", nargs="+", default=None,
                        help="Patient files reserved for test. When set, --filenames are "
                             "validation patients and must be disjoint from --test-filenames. "
                             "Without this, the legacy within-file slice split is used.")
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--accelerations", nargs="+", type=int, default=None,
                        help="If given, sweep these accelerations (e.g. --accelerations 4 8). "
                             "Each runs the full val→select→test pipeline in its own subdir.")
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="If given, run each method/sample for every seed and pool the rows.")
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--val-slices", type=int, default=2)
    parser.add_argument("--test-slices", type=int, default=3)
    parser.add_argument("--test-same-as-val", action="store_true",
                        help="Reuse validation samples as the test split. Useful for single-slice ablations.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing raw CSV rows in --out-dir and reuse selected.json if present.")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated list of exact methods to run (e.g. pULA,P-DAPS)")
    parser.add_argument("--methods-include", default=None,
                        help="Comma-separated method-name substrings to include after preset construction.")
    parser.add_argument("--method-indices", default=None,
                        help="Comma-separated grid indices/ranges to run after preset construction "
                             "and --methods filtering, e.g. 0,2,5-8.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "VAL"], default="VAL")
    parser.add_argument("--evaluate-all", action="store_true",
                        help="Run every grid entry on the test split instead of validation-selected entries.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Only run the validation pass. Useful for evaluate-all single-slice ablations "
                             "where validation and test would be the same samples.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save reconstruction figures for validation rows as well as test rows.")
    parser.add_argument("--debug-mem", action="store_true",
                        help="Log live/reserved CUDA memory after each eval cell and dump the largest "
                             "live tensors past a threshold (leak diagnosis; off by default).")
    parser.add_argument("--grid-preset",
                        choices=("smoke", "tiny", "probe", "full", "final_comparison", "pdaps_inner_sweep",
                                 "pdaps_tight", "iso_nfe", "pdaps_match_nfe",
                                 "pdaps_ablations", "pdaps_remediation",
                                 "pdaps_targeted", "pdaps_mechanism",
                                 "pdaps_nullspace_focus", "pdaps_v2", "pdaps_v3",
                                 "pdaps_v4", "pdaps_v5", "pdaps_v6", "pdaps_v7",
                                 "pdaps_v8a", "pdaps_v8b", "pdaps_v8c", "pdaps_v8d", "pdaps_v8e", "pdaps_v8f",
                                 "pdaps_working",
                                 "pdaps_prelaunch_c0",
                                 "pdaps_prelaunch_a_v8f", "pdaps_prelaunch_a_floor0",
                                 "pdaps_prelaunch_a_inf", "pdaps_prelaunch_a_floor0_inf",
                                 "pdaps_prelaunch_b_v8f_balfast", "pdaps_prelaunch_b_v8f_balanced",
                                 "pdaps_prelaunch_b_floor0_balfast", "pdaps_prelaunch_b_inf_balfast",
                                 "pdaps_prelaunch_b_floor0_inf_balfast",
                                 "pdaps_prelaunch_baselines", "pdaps_prelaunch_lrmin",
                                 "check_abandoned",
                                 "pdaps_bugcheck", "warm_sweep"),
                        default="tiny")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-grid", action="store_true")
    return parser.parse_args()


def parse_index_selection(raw, n_entries):
    selected = []
    seen = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, stop_s = part.split("-", 1)
            start, stop = int(start_s), int(stop_s)
            if stop < start:
                raise ValueError(f"Invalid descending method index range: {part}")
            values = range(start, stop + 1)
        else:
            values = [int(part)]
        for idx in values:
            if idx < 0 or idx >= n_entries:
                raise ValueError(f"Method index {idx} out of range for {n_entries} entries")
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    return selected


def _normalize_filenames(filenames):
    out = []
    for filename in filenames:
        name = Path(filename).name
        if name in out:
            raise ValueError(f"Duplicate filename in split: {name}")
        out.append(name)
    return out


def _samples_by_file(dataset):
    by_file = {}
    for global_idx, (kspace_path, _maps_path, _slice) in enumerate(dataset.samples):
        by_file.setdefault(kspace_path.name, []).append(global_idx)
    return by_file


def _take_file_samples(dataset, by_file, filenames, n_slices, slice_offset, split_name):
    selected = []
    missing = [filename for filename in filenames if filename not in by_file]
    if missing:
        raise ValueError(f"{split_name} filenames have no indexed slices: {missing}")
    for filename in filenames:
        global_indices = by_file[filename]
        wanted = global_indices[slice_offset:slice_offset + n_slices]
        if len(wanted) < n_slices:
            print(f"Warning: {filename} has only {len(wanted)} usable {split_name} slices at offset "
                  f"{slice_offset} (wanted {n_slices}), using what's available")
        for global_idx in wanted:
            selected.append((global_idx, filename))
    return selected


def _select_samples(dataset, slice_offset, val_slices, test_slices):
    """
    Pick `val_slices + test_slices` slices from each file in the dataset,
    starting at `slice_offset` within that file. Returns
    (val_samples, test_samples) where each is a list of
    (global_idx, filename). val_samples is the first `val_slices` from each
    file; test_samples is the remaining `test_slices` from each file.
    """
    by_file = _samples_by_file(dataset)

    val_samples, test_samples = [], []
    for filename, global_indices in by_file.items():
        wanted = global_indices[slice_offset:slice_offset + val_slices + test_slices]
        if len(wanted) < val_slices + test_slices:
            print(f"Warning: {filename} has only {len(wanted)} usable slices at offset "
                  f"{slice_offset} (wanted {val_slices + test_slices}), using what's available")
        for i, global_idx in enumerate(wanted):
            entry = (global_idx, filename)
            (val_samples if i < val_slices else test_samples).append(entry)
    return val_samples, test_samples


def _select_patient_disjoint_samples(dataset, val_filenames, test_filenames,
                                     slice_offset, val_slices, test_slices):
    """
    Pick validation and test samples from disjoint patient files. Each HDF5
    file is treated as one patient; slice selection starts after the dataset's
    central-slice crop and then applies `slice_offset`.
    """
    val_filenames = _normalize_filenames(val_filenames)
    test_filenames = _normalize_filenames(test_filenames)
    overlap = sorted(set(val_filenames) & set(test_filenames))
    if overlap:
        raise ValueError(
            "Patient leakage guard: --filenames and --test-filenames overlap: "
            + ", ".join(overlap)
        )

    by_file = _samples_by_file(dataset)
    val_samples = _take_file_samples(dataset, by_file, val_filenames, val_slices,
                                     slice_offset, "validation")
    test_samples = _take_file_samples(dataset, by_file, test_filenames, test_slices,
                                      slice_offset, "test")
    print("Patient-disjoint partition:")
    print(f"  validation patients ({len(val_filenames)}): {', '.join(val_filenames)}")
    print(f"  test patients ({len(test_filenames)}): {', '.join(test_filenames)}")
    print(f"  selected slices: {len(val_samples)} validation + {len(test_samples)} test")
    return val_samples, test_samples


def _row_key_from_values(method, params_json, split, sample_idx, filename, seed, acceleration):
    return (
        str(method),
        str(params_json),
        str(split),
        str(sample_idx),
        str(filename),
        str(seed),
        str(acceleration),
    )


def _row_key(row):
    return _row_key_from_values(
        row.get("method"),
        row.get("params_json"),
        row.get("split"),
        row.get("sample_idx"),
        row.get("filename"),
        row.get("seed"),
        row.get("acceleration"),
    )


def _successful_row_keys(rows):
    return {_row_key(row) for row in rows if not row.get("failed")}


def _expected_row_key(entry, split, sample_idx, filename, seed, acceleration):
    return _row_key_from_values(
        entry["method"],
        json.dumps(entry["params"], sort_keys=True),
        split,
        sample_idx,
        filename,
        seed,
        acceleration,
    )


def _load_resume_rows(path, enabled):
    if not enabled:
        return []
    rows = read_csv_rows(path)
    if rows:
        print(f"Resuming with {len(rows)} existing rows from {path}")
    return rows


def _run_one_acceleration(args, entries, net, dataset, val_samples, test_samples, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(entries, f, indent=2)

    seed_values = list(args.seeds) if getattr(args, "seeds", None) else [args.seed]
    selected_path = out_dir / "selected.json"
    selected = None
    if args.resume and selected_path.exists() and not args.evaluate_all:
        with open(selected_path) as f:
            candidate_selected = json.load(f)
        expected_methods = {entry["method"] for entry in entries}
        if set(candidate_selected) == expected_methods:
            selected = candidate_selected
            print(f"Resuming with frozen selection from {selected_path}")
        else:
            print(
                f"Ignoring incomplete selection in {selected_path}: "
                f"found {sorted(candidate_selected)}, expected {sorted(expected_methods)}"
            )

    val_rows = _load_resume_rows(out_dir / "validation_raw.csv", args.resume)
    if selected is None:
        if args.debug_mem:
            _cuda_mem_probe("baseline before VAL")
        val_done = _successful_row_keys(val_rows)
        for entry in entries:
            for seed in seed_values:
                args.seed = int(seed)
                for idx, filename in val_samples:
                    key = _expected_row_key(entry, "validation", idx, filename, args.seed, args.acceleration)
                    if key in val_done:
                        continue
                    sample = dataset[idx]
                    row = run_one(entry, sample, idx, "validation", net, args, out_dir,
                                  save_image=(args.evaluate_all or args.save_images),
                                  filename=filename)
                    del sample
                    val_rows.append(row)
                    val_done.add(_row_key(row))
                    write_csv(out_dir / "validation_raw.csv", val_rows)
                    if args.debug_mem:
                        _cuda_mem_probe(
                            f"after VAL {entry['method']} idx={idx}",
                            dump_top=8 if (
                                torch.cuda.is_available()
                                and torch.cuda.memory_allocated() > DEBUG_MEM_DUMP_GB * 1e9
                            ) else 0,
                        )
    elif val_rows:
        print("Skipping validation execution because selected.json is already frozen.")

    val_summary = summarize(val_rows)
    write_csv(out_dir / "validation_summary.csv", val_summary)

    if args.evaluate_all:
        selected_entries = entries
        with open(out_dir / "evaluation_plan.json", "w") as f:
            json.dump({"mode": "evaluate_all", "num_entries": len(entries)}, f, indent=2)
    else:
        if selected is None:
            selected = select_best(val_summary)
            with open(selected_path, "w") as f:
                json.dump(selected, f, indent=2)
        selected_entries = [
            entry for entry in entries
            if selected.get(entry["method"]) == json.dumps(entry["params"], sort_keys=True)
        ]

    test_rows = _load_resume_rows(out_dir / "test_raw.csv", args.resume)
    if args.skip_test:
        with open(out_dir / "evaluation_plan.json", "w") as f:
            json.dump({
                "mode": "validation_only",
                "num_entries": len(entries),
                "reason": "skip_test",
            }, f, indent=2)
    else:
        if args.debug_mem:
            _cuda_mem_probe("baseline before TEST")
        test_done = _successful_row_keys(test_rows)
        for entry in selected_entries:
            for seed in seed_values:
                args.seed = int(seed)
                for idx, filename in test_samples:
                    key = _expected_row_key(entry, "test", idx, filename, args.seed, args.acceleration)
                    if key in test_done:
                        continue
                    sample = dataset[idx]
                    row = run_one(entry, sample, idx, "test", net, args, out_dir,
                                  save_image=True, filename=filename)
                    del sample
                    test_rows.append(row)
                    test_done.add(_row_key(row))
                    write_csv(out_dir / "test_raw.csv", test_rows)
                    if args.debug_mem:
                        _cuda_mem_probe(
                            f"after TEST {entry['method']} idx={idx}",
                            dump_top=8 if (
                                torch.cuda.is_available()
                                and torch.cuda.memory_allocated() > DEBUG_MEM_DUMP_GB * 1e9
                            ) else 0,
                        )
        write_csv(out_dir / "test_summary.csv", summarize(test_rows))
    if args.evaluate_all or args.skip_test:
        write_ablation_artifacts(out_dir, val_rows, test_rows)
    print(f"Wrote {out_dir}")


def run_validation(args):
    entries = method_grid(args.grid_preset, log_level=args.log_level)
    if args.grid_preset == "pdaps_v2":
        args.evaluate_all = True
    exact_methods_filter = getattr(args, "methods", None)
    if exact_methods_filter:
        wanted = {m.strip() for m in exact_methods_filter.split(",") if m.strip()}
        entries = [e for e in entries if e["method"] in wanted]
    include_methods_filter = getattr(args, "methods_include", None)
    if include_methods_filter:
        wanted = [m.strip() for m in include_methods_filter.split(",") if m.strip()]
        entries = [e for e in entries if any(token in e["method"] for token in wanted)]
    method_indices = getattr(args, "method_indices", None)
    if method_indices:
        indices = parse_index_selection(method_indices, len(entries))
        entries = [entries[idx] for idx in indices]
    if args.list_grid:
        print(json.dumps(entries, indent=2))
        return None

    out_dir = Path(args.out_dir or f"results/mri_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not out_dir.is_absolute():
        try:
            out_dir = Path(get_original_cwd()) / out_dir
        except ValueError:
            out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args, device)

    val_files = _normalize_filenames(args.filenames if args.filenames else [args.filename])
    test_files = _normalize_filenames(args.test_filenames) if getattr(args, "test_filenames", None) else None
    if test_files:
        if getattr(args, "test_same_as_val", False):
            raise ValueError("--test-same-as-val cannot be combined with patient-disjoint --test-filenames")
        files = val_files + [filename for filename in test_files if filename not in set(val_files)]
    else:
        files = val_files
    dataset = MultiCoilMRIDataset(args.kspace_dir, args.maps_dir, args.image_size, filenames=files)
    if test_files:
        val_samples, test_samples = _select_patient_disjoint_samples(
            dataset, val_files, test_files, args.slice_offset, args.val_slices, args.test_slices)
    else:
        val_samples, test_samples = _select_samples(dataset, args.slice_offset,
                                                    args.val_slices, args.test_slices)
        if getattr(args, "test_same_as_val", False):
            test_samples = list(val_samples)
        print(f"Selected {len(val_samples)} val + {len(test_samples)} test slices "
              f"across {len(files)} file(s) "
              f"({args.val_slices} val + {args.test_slices} test per file).")

    accelerations = args.accelerations if args.accelerations else [args.acceleration]
    if len(accelerations) == 1:
        args.acceleration = accelerations[0]
        _run_one_acceleration(args, entries, net, dataset, val_samples, test_samples, out_dir)
    else:
        for accel in accelerations:
            args.acceleration = accel
            sub_dir = out_dir / f"accel_{accel}"
            print(f"=== acceleration {accel}x → {sub_dir} ===")
            _run_one_acceleration(args, entries, net, dataset, val_samples, test_samples, sub_dir)
        if args.evaluate_all:
            write_combined_acceleration_artifacts(out_dir, accelerations)
    return out_dir


def run_from_hydra(cfg):
    validation = cfg.get("validation", {})
    args = argparse.Namespace(
        models_dir=cfg.paths.models_dir,
        ckpt_name=cfg.pretrain.ckpt_name,
        kspace_dir=cfg.dataset.kspace_dir,
        maps_dir=cfg.dataset.maps_dir,
        filename=validation.get("filename", cfg.dataset.get("filenames", ["file1000196.h5"])[0]),
        filenames=list(validation.get("filenames", cfg.dataset.get("filenames", []))) or None,
        test_filenames=list(validation.get("test_filenames", [])) or None,
        image_size=list(cfg.dataset.image_size),
        acceleration=int(cfg.forward_op.acceleration_ratio),
        accelerations=[int(a) for a in validation.get("accelerations", [])] or None,
        pattern=cfg.forward_op.get("pattern", "random"),
        mask_seed=int(cfg.forward_op.get("mask_seed", 0)),
        seed=int(validation.get("seed", 123)),
        seeds=[int(s) for s in validation.get("seeds", [])] or None,
        slice_offset=int(validation.get("slice_offset", 0)),
        val_slices=int(validation.get("val_slices", 2)),
        test_slices=int(validation.get("test_slices", 3)),
        test_same_as_val=bool(validation.get("test_same_as_val", False)),
        out_dir=validation.get("out_dir", None),
        resume=bool(validation.get("resume", False)),
        verbose=bool(validation.get("verbose", False)),
        list_grid=bool(validation.get("list_grid", False)),
        grid_preset=validation.get("grid_preset", "tiny"),
        methods=validation.get("methods", None),
        methods_include=validation.get("methods_include", None),
        log_level=validation.get("log_level", "VAL"),
        evaluate_all=bool(validation.get("evaluate_all", False)),
        save_images=bool(validation.get("save_images", False)),
        method_indices=validation.get("method_indices", None),
        skip_test=bool(validation.get("skip_test", False)),
    )
    return run_validation(args)


def main():
    run_validation(parse_args())


if __name__ == "__main__":
    main()
