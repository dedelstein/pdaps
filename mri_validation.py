import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
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
    "num_steps": 200,
    "sigma_max": 10,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}


def grid(points):
    keys = list(points)
    for values in itertools.product(*(points[key] for key in keys)):
        yield dict(zip(keys, values))


def method_grid(preset="tiny", log_level="INFO"):
    if preset == "pdaps_ablations":
        return _pdaps_ablations_grid(log_level=log_level)
    if preset == "pdaps_remediation":
        return _pdaps_remediation_grid(log_level=log_level)
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


def pdaps_entry(method, warm_mode, gamma, warm_fraction,
                inner_sigma_max=PDAPS_INNER_SIGMA_MAX, lgvd_num_steps=25, log_level="INFO",
                lam_floor=0.0, target_lam_floor=None, solve_lam_floor=None,
                noise_lam_floor=None, noise_tau=1.0, noise_mode="full",
                gamma_schedule="constant", gamma_floor=0.0, gamma_ceiling=float("inf"),
                precond_mode="standard", noise_rhs_mode="standard",
                penalty_scale=1.0, penalty_schedule="lambda", penalty_eps=0.0,
                mask_split_eps=1.0,
                edm_project_post=False, label_suffix=""):
    inner_str = "inf" if inner_sigma_max >= 1e8 else f"{inner_sigma_max:g}"
    params = {
        "gamma": gamma, "warm_fraction": warm_fraction,
        "inner_sigma_max": inner_str, "lgvd_num_steps": int(lgvd_num_steps),
    }
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
    if mask_split_eps != 1.0:
        params["mask_split_eps"] = float(mask_split_eps)
    if edm_project_post:
        params["edm_proj"] = True
    method_label = method + (f"[{label_suffix}]" if label_suffix else "")
    lgvd_config = {
        "num_steps": int(lgvd_num_steps),
        "gamma": gamma,
        "cg_iter": 10,
        "lr_min_ratio": 0.01,
        "lam_floor": float(lam_floor),
        "noise_tau": float(noise_tau),
        "noise_mode": noise_mode,
        "gamma_schedule": gamma_schedule,
        "precond_mode": precond_mode,
        "noise_rhs_mode": noise_rhs_mode,
        "penalty_scale": float(penalty_scale),
        "penalty_schedule": penalty_schedule,
        "penalty_eps": float(penalty_eps),
        "mask_split_eps": float(mask_split_eps),
    }
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
    return {
        "method": method_label,
        "params": params,
        "algorithm": {
            "_target_": "algo.pdaps.PDAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": lgvd_config,
            "warm_mode": warm_mode,
            "warm_fraction": warm_fraction,
            "inner_sigma_max": inner_sigma_max,
            "edm_project_post": bool(edm_project_post),
            "log_level": log_level,
        },
    }


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

    forward_op = make_forward_op(args, device)
    algo = hydra.utils.instantiate(OmegaConf.create(entry["algorithm"]), forward_op=forward_op, net=net)
    data = move_to_device(sample, device)
    data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    observation = forward_op(data)
    target = data["target"]

    row = {
        "split": split,
        "sample_idx": sample_idx,
        "filename": filename,
        "acceleration": args.acceleration,
        "method": entry["method"],
        "params_json": json.dumps(entry["params"], sort_keys=True),
        "failed": False,
    }

    start = time.perf_counter()
    try:
        recon = algo.inference(observation, num_samples=1, verbose=args.verbose)
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics = compute_metrics_dict(forward_op, recon, target, observation)
        row.update(metrics)
        metric_names = ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed")
        if not metrics.get("finite", False) or not all(math.isfinite(float(metrics[name])) for name in metric_names):
            row["failed"] = True
            row["error"] = "nonfinite_reconstruction_or_metrics"
        row["runtime_s"] = time.perf_counter() - start
        row["gate_stats_json"] = json.dumps(getattr(algo, "last_gate_stats", []))
        
        if save_image and not row["failed"]:
            cfg = OmegaConf.create({
                "algorithm": {"_target_": entry["algorithm"]["_target_"]},
                "forward_op": {"acceleration_ratio": args.acceleration},
            })
            image_dir = out_dir / "figures" / entry["method"].replace("/", "_")
            image_dir.mkdir(parents=True, exist_ok=True)
            old_cwd = os.getcwd()
            os.chdir(image_dir)
            try:
                visualize_recon(forward_op, forward_op.unnormalize(recon).cpu(), forward_op.unnormalize(target).cpu(), sample_idx, cfg)
            finally:
                os.chdir(old_cwd)
    except Exception as exc:
        row["failed"] = True
        row["error"] = repr(exc)
        row["runtime_s"] = time.perf_counter() - start
        
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
        }
        for metric in ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed", "runtime_s"):
            vals = [float(row[metric]) for row in ok if metric in row]
            if vals:
                summary[f"{metric}_mean"] = sum(vals) / len(vals)
                summary[f"{metric}_std"] = torch.tensor(vals).std(unbiased=False).item()
        out.append(summary)
    return sorted(out, key=lambda row: (row["method"], -row.get("psnr_mean", -1e9)))


def select_best(validation_summary):
    best = {}
    for row in validation_summary:
        if row["n_ok"] == 0:
            continue
        current = best.get(row["method"])
        candidate = (row.get("psnr_mean", -1e9), row.get("ssim_mean", -1e9), -row.get("data_misfit_mean", 1e9))
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
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--accelerations", nargs="+", type=int, default=None,
                        help="If given, sweep these accelerations (e.g. --accelerations 4 8). "
                             "Each runs the full val→select→test pipeline in its own subdir.")
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--val-slices", type=int, default=2)
    parser.add_argument("--test-slices", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated list of methods to run (e.g. pULA,P-DAPS)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "VAL"], default="VAL")
    parser.add_argument("--grid-preset",
                        choices=("smoke", "tiny", "probe", "full", "pdaps_inner_sweep",
                                 "pdaps_tight", "iso_nfe", "pdaps_match_nfe",
                                 "pdaps_ablations", "pdaps_remediation",
                                 "pdaps_targeted", "warm_sweep"),
                        default="tiny")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-grid", action="store_true")
    return parser.parse_args()


def _select_samples(dataset, slice_offset, val_slices, test_slices):
    """
    Pick `val_slices + test_slices` slices from each file in the dataset,
    starting at `slice_offset` within that file. Returns
    (val_samples, test_samples) where each is a list of
    (global_idx, filename, sample_dict). val_samples is the first
    `val_slices` from each file; test_samples is the remaining
    `test_slices` from each file.
    """
    by_file = {}
    for global_idx, (kspace_path, _maps_path, _slice) in enumerate(dataset.samples):
        by_file.setdefault(kspace_path.name, []).append(global_idx)

    val_samples, test_samples = [], []
    for filename, global_indices in by_file.items():
        wanted = global_indices[slice_offset:slice_offset + val_slices + test_slices]
        if len(wanted) < val_slices + test_slices:
            print(f"Warning: {filename} has only {len(wanted)} usable slices at offset "
                  f"{slice_offset} (wanted {val_slices + test_slices}), using what's available")
        for i, global_idx in enumerate(wanted):
            entry = (global_idx, filename, dataset[global_idx])
            (val_samples if i < val_slices else test_samples).append(entry)
    return val_samples, test_samples


def _run_one_acceleration(args, entries, net, val_samples, test_samples, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(entries, f, indent=2)

    val_rows = []
    for entry in entries:
        for idx, filename, sample in val_samples:
            val_rows.append(run_one(entry, sample, idx, "validation", net, args, out_dir,
                                    filename=filename))
            write_csv(out_dir / "validation_raw.csv", val_rows)
    val_summary = summarize(val_rows)
    write_csv(out_dir / "validation_summary.csv", val_summary)

    selected = select_best(val_summary)
    selected_entries = [entry for entry in entries if selected.get(entry["method"]) == json.dumps(entry["params"], sort_keys=True)]
    with open(out_dir / "selected.json", "w") as f:
        json.dump(selected, f, indent=2)

    test_rows = []
    for entry in selected_entries:
        for idx, filename, sample in test_samples:
            test_rows.append(run_one(entry, sample, idx, "test", net, args, out_dir,
                                     save_image=True, filename=filename))
            write_csv(out_dir / "test_raw.csv", test_rows)
    write_csv(out_dir / "test_summary.csv", summarize(test_rows))
    print(f"Wrote {out_dir}")


def run_validation(args):
    entries = method_grid(args.grid_preset, log_level=args.log_level)
    methods_filter = getattr(args, "methods", None)
    if methods_filter:
        wanted = {m.strip() for m in methods_filter.split(",") if m.strip()}
        entries = [e for e in entries if e["method"] in wanted]
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

    files = args.filenames if args.filenames else [args.filename]
    dataset = MultiCoilMRIDataset(args.kspace_dir, args.maps_dir, args.image_size, filenames=files)
    val_samples, test_samples = _select_samples(dataset, args.slice_offset,
                                                args.val_slices, args.test_slices)
    print(f"Selected {len(val_samples)} val + {len(test_samples)} test slices "
          f"across {len(files)} file(s) "
          f"({args.val_slices} val + {args.test_slices} test per file).")

    accelerations = args.accelerations if args.accelerations else [args.acceleration]
    if len(accelerations) == 1:
        args.acceleration = accelerations[0]
        _run_one_acceleration(args, entries, net, val_samples, test_samples, out_dir)
    else:
        for accel in accelerations:
            args.acceleration = accel
            sub_dir = out_dir / f"accel_{accel}"
            print(f"=== acceleration {accel}x → {sub_dir} ===")
            _run_one_acceleration(args, entries, net, val_samples, test_samples, sub_dir)
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
        image_size=list(cfg.dataset.image_size),
        acceleration=int(cfg.forward_op.acceleration_ratio),
        accelerations=[int(a) for a in validation.get("accelerations", [])] or None,
        pattern=cfg.forward_op.get("pattern", "random"),
        mask_seed=int(cfg.forward_op.get("mask_seed", 0)),
        seed=int(validation.get("seed", 123)),
        slice_offset=int(validation.get("slice_offset", 0)),
        val_slices=int(validation.get("val_slices", 2)),
        test_slices=int(validation.get("test_slices", 3)),
        out_dir=validation.get("out_dir", None),
        verbose=bool(validation.get("verbose", False)),
        list_grid=bool(validation.get("list_grid", False)),
        grid_preset=validation.get("grid_preset", "tiny"),
        methods=validation.get("methods", None),
        log_level=validation.get("log_level", "VAL"),
    )
    return run_validation(args)


def main():
    run_validation(parse_args())


if __name__ == "__main__":
    main()
