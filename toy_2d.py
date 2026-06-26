#!/usr/bin/env python3
"""Synthetic toy suite for comparing DPS, DAPS, pULA, and P-DAPS."""

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import trange

torch.manual_seed(42)
DEVICE = torch.device("cpu")
DTYPE = torch.float64

DIVERGENCE_ABS_MAX = 1e4
SDE = "VE"
VP_BETA_D = 19.9
VP_BETA_MIN = 0.1


def daps_tau_toy(sigma_noise):
    """DAPS-matched tau for the toy, derived from the toy whitening convention.

    inversebench DAPS weights the data term by 1/(2*tau**2) on the
    unnormalized loss ||Ax-y||^2, so tau is the effective noise scale and the
    MRI value near 0.002 is a tuned trust scale. The toy DAPS uses the physical
    weight 1/sigma_noise**2. Toy P-DAPS whitens the operator with
    A_s = A/sigma_noise, folding that scale into A_s; in the whitened frame the
    prior anchor weight is lam_target = tau**2/sigma**2. Matching the untuned toy
    DAPS prior 1/sigma**2 therefore gives tau = 1.
    """
    _ = sigma_noise
    return 1.0


@dataclass(frozen=True)
class NoiseSchedule:
    sigmas: torch.Tensor
    alphas: torch.Tensor
    time_steps: Optional[np.ndarray]
    factor_steps: np.ndarray
    scaling_factor: np.ndarray


class DivergenceError(RuntimeError):
    """Raised by an inner Langevin step when the iterate blows up."""

    def __init__(self, step_size, max_abs, where=""):
        self.step_size = float(step_size)
        self.max_abs = float(max_abs)
        self.where = where
        super().__init__(
            f"DIVERGENCE_DETECTED: step_size={self.step_size:.3e}, "
            f"max_abs={self.max_abs:.3e} ({where})"
        )


def _check_divergence(x, step_size, where):
    if not torch.isfinite(x).all():
        raise DivergenceError(step_size, float("inf"), where)
    mx = x.abs().max().item()
    if mx > DIVERGENCE_ABS_MAX:
        raise DivergenceError(step_size, mx, where)


def _diverged_result(x_last, progress, exc, step_idx):
    final = x_last.detach().cpu().numpy() if isinstance(x_last, torch.Tensor) else np.asarray(x_last)
    return {
        "final": final,
        "progress": progress,
        "diverged_at": int(step_idx),
        "diverged_step_size": exc.step_size,
        "diverged_max_abs": exc.max_abs,
        "diverged_where": exc.where,
    }


SCENARIOS = {
    "toy_a_mode_recovery": {
        "name": "Toy A: Mode Recovery under Partial Observation",
        "dim": 2,
        "summary": "Sharper modes, weaker x2 observation, fewer outer steps, and mild high-noise score bias. Intended to make error recovery harder for coupled solvers.",
        "cluster_centers": [[0.0, 1.00], [1.05, 0.0], [0.0, -1.00], [-1.05, 0.0]],
        "cluster_std": 0.065,
        "a_diag": [1.0, 0.012],
        "sigma_noise": 0.05,
        "x_true": [0.0, 1.00],
        "lim": 1.6,
        "obs_seed": 0,
        "run_params": {
            "N": 80,
            "sigma_max": 1.6,
            "sigma_min": 0.005,
            "dps_guidance_scale": 2.2,
            "daps_ode_steps": 5,
            "daps_langevin_steps": 120,
            "daps_langevin_lr": 2.5e-5,
            "pula_step_size": 0.5,
            "pula_nb_langevin": 6,
            "pdaps_ode_steps": 5,
            "pdaps_langevin_steps": 18,
            "pdaps_langevin_step_size": 0.5,
            "pdaps_warm_fraction": 0.10,
        },
        "score_bias": {"bias": [0.24, 0.0], "sigma_gate": 0.24, "sharpness": 16.0},
    },
    "toy_b_stiffness": {
        "name": "Toy B: Stiffness under Ill-Conditioning",
        "dim": 2,
        "summary": "Harsher conditioning with larger starting noise and fewer outer steps so the decoupled outer denoise/re-noise structure matters more.",
        "cluster_centers": [[0.0, 0.70], [0.90, 0.0], [0.0, -0.70], [-0.90, 0.0]],
        "cluster_std": 0.11,
        "a_diag": [1.0, 0.0008],
        "sigma_noise": 0.05,
        "x_true": [0.0, 0.70],
        "lim": 1.6,
        "obs_seed": 0,
        "run_params": {
            "N": 55,
            "sigma_max": 2.0,
            "sigma_min": 0.005,
            "dps_guidance_scale": 2.6,
            "daps_ode_steps": 7,
            "daps_langevin_steps": 70,
            "daps_langevin_lr": 1.2e-5,
            "pula_step_size": 0.5,
            "pula_nb_langevin": 5,
            "pdaps_ode_steps": 7,
            "pdaps_langevin_steps": 16,
            "pdaps_langevin_step_size": 0.5,
            "pdaps_warm_fraction": 0.25,
        },
        "score_bias": {"bias": [0.0, 0.0], "sigma_gate": 0.25, "sharpness": 12.0},
    },
    "toy_c_score_bias": {
        "name": "Toy C: Error Propagation under Score Bias",
        "dim": 2,
        "summary": "Adds mild high-noise model bias so coupled methods must carry upstream score errors deeper into the schedule.",
        "cluster_centers": [[0.0, 0.75], [0.90, 0.0], [0.0, -0.75], [-0.90, 0.0]],
        "cluster_std": 0.09,
        "a_diag": [1.0, 0.03],
        "sigma_noise": 0.05,
        "x_true": [0.0, 0.75],
        "lim": 1.5,
        "obs_seed": 0,
        "run_params": {
            "N": 300,
            "sigma_max": 1.5,
            "sigma_min": 0.005,
            "dps_guidance_scale": 2.0,
            "daps_ode_steps": 8,
            "daps_langevin_steps": 100,
            "daps_langevin_lr": 1.5e-5,
            "pula_step_size": 0.5,
            "pula_nb_langevin": 10,
            "pdaps_ode_steps": 8,
            "pdaps_langevin_steps": 20,
            "pdaps_langevin_step_size": 0.5,
            "pdaps_warm_fraction": 0.25,
        },
        "score_bias": {"bias": [0.18, 0.0], "sigma_gate": 0.22, "sharpness": 14.0},
    },
    "toy_d_high_d_stiffness": {
        "name": "Toy D: High-d Stiff Linear-Gaussian",
        "dim": 64,
        "summary": "A 64-dimensional linear-Gaussian-GMM toy with a controlled singular-value spectrum. Modes live in the first two coordinates; stiffness lives in the full operator.",
        "cluster_centers_2d": [[0.0, 1.00], [1.05, 0.0], [0.0, -1.00], [-1.05, 0.0]],
        "cluster_std": 0.065,
        "kappa": 100,
        "sigma_noise": 0.05,
        "x_true_2d": [0.0, 1.00],
        "lim": 1.6,
        "obs_seed": 0,
        "u_seed": 0,
        "v_seed": 1,
        "posterior": "analytic",
        "run_params": {
            "N": 35,
            "sigma_max": 1.6,
            "sigma_min": 0.005,
            "dps_guidance_scale": 2.2,
            "daps_ode_steps": 5,
            "daps_langevin_steps": 40,
            "daps_langevin_lr": 2.5e-5,
            "pula_step_size": 0.5,
            "pula_nb_langevin": 4,
            "pdaps_ode_steps": 5,
            "pdaps_langevin_steps": 8,
            "pdaps_langevin_step_size": 0.5,
            "pdaps_warm_fraction": 0.25,
        },
        "score_bias": {"bias": [0.0, 0.0], "sigma_gate": 0.24, "sharpness": 16.0},
    },
}
CORE_SCENARIO_KEYS = ("toy_a_mode_recovery", "toy_b_stiffness")
OPTIONAL_SCENARIO_KEYS = ("toy_c_score_bias", "toy_d_high_d_stiffness")


CURRENT_SCENARIO = None
CURRENT_SCENARIO_KEY = None
DIM = None
LIM = None
CLUSTER_STD = None
CLUSTER_CENTERS = None
LOG_W = None
MU = None
PRIOR_VAR = None
SIGMA_NOISE = None
A = None
X_TRUE = None
Y_OBS = None

SNAP_SIGMAS = [0.5, 0.1, 0.01]
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "white_to_fire",
    ["#ffffff", "#f5d76e", "#f28f3b", "#c8553d", "#4b1d3f"],
)
WARM_SWEEP_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHOD_COLORS = {
    "DPS": "#c8553d",
    "DAPS": "#7f3c8d",
    "pULA": "#2c7fb8",
    "P-DAPS": "#1b9e77",
    "P-DAPS-warm": "#8c6d1f",
}


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_torch_scalar(value):
    return torch.tensor(value, dtype=DTYPE, device=DEVICE)


def alpha_of(sigma):
    if SDE == "VE":
        return torch.ones_like(sigma) if torch.is_tensor(sigma) else 1.0
    if torch.is_tensor(sigma):
        return 1.0 / torch.sqrt(1.0 + sigma ** 2)
    return 1.0 / float(np.sqrt(1.0 + sigma ** 2))


def marginal_std(sigma):
    return alpha_of(sigma) * sigma


def _vp_sigma_fn(t):
    return np.sqrt(np.exp(VP_BETA_D * t ** 2 / 2 + VP_BETA_MIN * t) - 1)


def _vp_sigma_derivative_fn(t):
    exp_term = np.exp(VP_BETA_D * t ** 2 / 2 + VP_BETA_MIN * t)
    sigma = _vp_sigma_fn(t)
    return (VP_BETA_D * t + VP_BETA_MIN) * exp_term / 2 / np.maximum(sigma, 1e-300)


def _vp_sigma_inv_fn(sigma):
    sigma = np.asarray(sigma, dtype=float)
    return np.sqrt(VP_BETA_MIN ** 2 + 2 * VP_BETA_D * np.log(sigma ** 2 + 1)) / VP_BETA_D - VP_BETA_MIN / VP_BETA_D


def _vp_scaling_derivative_over_scaling(t):
    return -(VP_BETA_D * t + VP_BETA_MIN) / 2


def make_schedule(num_steps, sigma_max, sigma_min):
    """Build the toy noise schedule.

    VE preserves the existing geometric-in-sigma discretization. VP follows
    inversebench's separated sigma/scaling convention with linear time steps
    between the clean-noise endpoints implied by sigma_max and sigma_min.
    """
    if SDE == "VE":
        sigmas_np = np.geomspace(sigma_max, sigma_min, num_steps + 1)
        sigmas = torch.tensor(sigmas_np, dtype=DTYPE, device=DEVICE)
        return NoiseSchedule(
            sigmas=sigmas,
            alphas=torch.ones_like(sigmas),
            time_steps=None,
            factor_steps=2.0 * sigmas_np[:-1] * (sigmas_np[:-1] - sigmas_np[1:]),
            scaling_factor=np.ones(num_steps, dtype=float),
        )

    t_max = float(_vp_sigma_inv_fn(sigma_max))
    t_min = float(_vp_sigma_inv_fn(sigma_min))
    time_steps = np.linspace(t_max, t_min, num_steps + 1)
    sigmas_np = _vp_sigma_fn(time_steps)
    alphas_np = 1.0 / np.sqrt(1.0 + sigmas_np ** 2)
    dt = time_steps[:-1] - time_steps[1:]
    factor_steps = 2.0 * alphas_np[:-1] ** 2 * sigmas_np[:-1] * _vp_sigma_derivative_fn(time_steps[:-1]) * dt
    scaling_factor = 1.0 - _vp_scaling_derivative_over_scaling(time_steps[:-1]) * dt
    return NoiseSchedule(
        sigmas=torch.tensor(sigmas_np, dtype=DTYPE, device=DEVICE),
        alphas=torch.tensor(alphas_np, dtype=DTYPE, device=DEVICE),
        time_steps=time_steps,
        factor_steps=np.maximum(factor_steps, 0.0),
        scaling_factor=scaling_factor,
    )


def make_langevin_sigmas(num_values, sigma_max, sigma_min):
    if SDE == "VE":
        return torch.tensor(np.geomspace(sigma_max, sigma_min, num_values), dtype=DTYPE, device=DEVICE)
    if num_values == 1:
        return torch.tensor([sigma_max], dtype=DTYPE, device=DEVICE)
    return make_schedule(num_values - 1, sigma_max, sigma_min).sigmas


def _renoise_clean(x0, sigma_next, alpha_next):
    if SDE == "VE":
        return x0 + torch.randn_like(x0) * sigma_next
    return alpha_next * (x0 + torch.randn_like(x0) * sigma_next)


def _clean_from_noised(x, alpha):
    if SDE == "VE":
        return x
    return x / alpha


def _scenario_centers(cfg):
    dim = int(cfg.get("dim", 2))
    if "cluster_centers" in cfg:
        return cfg["cluster_centers"]
    centers = np.zeros((len(cfg["cluster_centers_2d"]), dim), dtype=float)
    centers[:, :2] = np.asarray(cfg["cluster_centers_2d"], dtype=float)
    return centers.tolist()


def _scenario_x_true(cfg):
    dim = int(cfg.get("dim", 2))
    if "x_true" in cfg:
        return cfg["x_true"]
    x_true = np.zeros(dim, dtype=float)
    x_true[:2] = np.asarray(cfg["x_true_2d"], dtype=float)
    return x_true.tolist()


def _random_orthogonal(dim, seed):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    q, r = torch.linalg.qr(torch.randn(dim, dim, dtype=DTYPE, device=DEVICE, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


def _scenario_matrix(cfg):
    dim = int(cfg.get("dim", 2))
    if "A" in cfg:
        return torch.tensor(cfg["A"], dtype=DTYPE, device=DEVICE)
    if "a_diag" in cfg:
        return torch.diag(torch.tensor(cfg["a_diag"], dtype=DTYPE, device=DEVICE))
    kappa = float(cfg.get("kappa", 1.0))
    exponents = torch.linspace(0.0, 1.0, dim, dtype=DTYPE, device=DEVICE)
    singular_values = kappa ** (-exponents)
    u = _random_orthogonal(dim, cfg.get("u_seed", 0))
    v = _random_orthogonal(dim, cfg.get("v_seed", 1))
    return u @ torch.diag(singular_values) @ v.T


def configure_scenario(key, quiet=False, a_diag_override=None, sigma_noise_override=None,
                       score_bias_override=None, kappa_override=None):
    global CURRENT_SCENARIO, CURRENT_SCENARIO_KEY
    global DIM, LIM, CLUSTER_STD, CLUSTER_CENTERS, LOG_W, MU, PRIOR_VAR
    global SIGMA_NOISE, A, X_TRUE, Y_OBS

    cfg = copy.deepcopy(SCENARIOS[key])
    if a_diag_override is not None:
        cfg["a_diag"] = list(a_diag_override)
    if kappa_override is not None:
        cfg["kappa"] = float(kappa_override)
    if sigma_noise_override is not None:
        cfg["sigma_noise"] = float(sigma_noise_override)
    if score_bias_override is not None:
        cfg["score_bias"] = copy.deepcopy(score_bias_override)
    CURRENT_SCENARIO = cfg
    CURRENT_SCENARIO_KEY = key

    DIM = int(cfg.get("dim", 2))
    LIM = cfg["lim"]
    CLUSTER_STD = cfg["cluster_std"]
    cfg["cluster_centers"] = _scenario_centers(cfg)
    cfg["x_true"] = _scenario_x_true(cfg)
    CLUSTER_CENTERS = torch.tensor(cfg["cluster_centers"], dtype=DTYPE, device=DEVICE)
    LOG_W = torch.log(torch.full((CLUSTER_CENTERS.shape[0],), 1.0 / CLUSTER_CENTERS.shape[0], dtype=DTYPE, device=DEVICE))
    MU = CLUSTER_CENTERS
    PRIOR_VAR = CLUSTER_STD ** 2

    SIGMA_NOISE = cfg["sigma_noise"]
    A = _scenario_matrix(cfg)
    X_TRUE = torch.tensor(cfg["x_true"], dtype=DTYPE, device=DEVICE)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(cfg["obs_seed"])
    Y_OBS = A @ X_TRUE + SIGMA_NOISE * torch.randn(A.shape[0], dtype=DTYPE, device=DEVICE, generator=generator)

    if not quiet:
        print(f"\nScenario: {cfg['name']}")
        print(f"x_true = {X_TRUE.cpu().numpy()}")
        print(f"y_obs  = {Y_OBS.cpu().numpy()}")


configure_scenario("toy_a_mode_recovery", quiet=True)


def _clean_gmm_log_prob(x, sigma):
    var = PRIOR_VAR + sigma ** 2
    diff = x.unsqueeze(1) - MU.unsqueeze(0)
    exponents = -0.5 * (diff ** 2).sum(-1) / var
    log_norm = -0.5 * DIM * torch.log(torch.tensor(2 * np.pi * var, dtype=DTYPE, device=DEVICE))
    return torch.logsumexp(LOG_W + log_norm + exponents, dim=1)


def _gmm_log_prob(x, sigma):
    alpha = alpha_of(sigma)
    var = alpha ** 2 * (PRIOR_VAR + sigma ** 2)
    mu = alpha * MU
    diff = x.unsqueeze(1) - mu.unsqueeze(0)
    exponents = -0.5 * (diff ** 2).sum(-1) / var
    log_norm = -0.5 * DIM * torch.log(torch.tensor(2 * np.pi, dtype=DTYPE, device=DEVICE) * var)
    return torch.logsumexp(LOG_W + log_norm + exponents, dim=1)


def _score_bias(sigma):
    cfg = CURRENT_SCENARIO.get("score_bias", None)
    if cfg is None:
        return torch.zeros(DIM, dtype=DTYPE, device=DEVICE)
    bias_values = list(cfg["bias"])
    if len(bias_values) < DIM:
        bias_values = bias_values + [0.0] * (DIM - len(bias_values))
    bias = torch.tensor(bias_values[:DIM], dtype=DTYPE, device=DEVICE)
    if torch.allclose(bias, torch.zeros_like(bias)):
        return bias
    weight = 1.0 / (1.0 + np.exp(-cfg["sharpness"] * (sigma - cfg["sigma_gate"])))
    return bias * weight


def score_fn(x, sigma):
    x_ = x if x.requires_grad else x.detach().requires_grad_(True)
    lp = _gmm_log_prob(x_, sigma).sum()
    (s,) = torch.autograd.grad(lp, x_)
    return s.detach() + _score_bias(sigma)


def clean_score_fn(x, sigma):
    x_ = x if x.requires_grad else x.detach().requires_grad_(True)
    lp = _clean_gmm_log_prob(x_, sigma).sum()
    (s,) = torch.autograd.grad(lp, x_)
    return s.detach() + _score_bias(sigma)


def tweedie(x, sigma, *, create_graph=False):
    x_ = x if x.requires_grad else x.detach().requires_grad_(True)
    lp = _gmm_log_prob(x_, sigma).sum()
    (s,) = torch.autograd.grad(lp, x_, create_graph=create_graph)
    alpha = alpha_of(sigma)
    return (x_ + (alpha * sigma) ** 2 * (s + _score_bias(sigma))) / alpha


def sample_ground_truth(n_samples=200_000):
    if CURRENT_SCENARIO.get("posterior") == "analytic":
        return sample_ground_truth_analytic(n_samples=n_samples)
    indices = torch.multinomial(torch.exp(LOG_W), n_samples, replacement=True)
    samples = MU[indices] + CLUSTER_STD * torch.randn(n_samples, DIM, dtype=DTYPE)
    residuals = Y_OBS.unsqueeze(0) - (samples @ A.T)
    log_lik = -0.5 * (residuals ** 2).sum(-1) / SIGMA_NOISE ** 2
    log_w = log_lik - log_lik.max()
    w = torch.exp(log_w)
    w = w / w.sum()
    idx = torch.multinomial(w, n_samples, replacement=True)
    return samples[idx].numpy()


def sample_ground_truth_analytic(n_samples=200_000):
    prior_precision = torch.eye(DIM, dtype=DTYPE, device=DEVICE) / PRIOR_VAR
    noise_precision = 1.0 / SIGMA_NOISE ** 2
    post_cov = torch.inverse(prior_precision + noise_precision * (A.T @ A))
    rhs_y = noise_precision * (A.T @ Y_OBS)
    post_means = torch.stack([post_cov @ (prior_precision @ mu + rhs_y) for mu in MU])

    obs_cov = PRIOR_VAR * (A @ A.T) + SIGMA_NOISE ** 2 * torch.eye(A.shape[0], dtype=DTYPE, device=DEVICE)
    chol_obs = torch.linalg.cholesky(obs_cov)
    log_det_obs = 2.0 * torch.log(torch.diag(chol_obs)).sum()
    residuals = Y_OBS.unsqueeze(0) - (MU @ A.T)
    solved = torch.cholesky_solve(residuals.T, chol_obs).T
    log_lik = -0.5 * ((residuals * solved).sum(-1) + log_det_obs + A.shape[0] * np.log(2 * np.pi))
    log_post_w = LOG_W + log_lik
    probs = torch.softmax(log_post_w, dim=0)

    indices = torch.multinomial(probs, n_samples, replacement=True)
    chol_post = torch.linalg.cholesky(post_cov)
    noise = torch.randn(n_samples, DIM, dtype=DTYPE, device=DEVICE)
    samples = post_means[indices] + noise @ chol_post.T
    return samples.cpu().numpy()


def sample_prior(n_samples=80_000):
    indices = torch.multinomial(torch.exp(LOG_W), n_samples, replacement=True)
    samples = MU[indices] + CLUSTER_STD * torch.randn(n_samples, DIM, dtype=DTYPE)
    return samples.numpy()


def posterior_log_prob(x):
    prior_lp = _gmm_log_prob(x, sigma=0.0)
    residual = Y_OBS.unsqueeze(0) - (x @ A.T)
    log_lik = -0.5 * (residual ** 2).sum(-1) / SIGMA_NOISE ** 2
    return prior_lp + log_lik


def make_grid(grid_size=250):
    xs = np.linspace(-LIM, LIM, grid_size)
    ys = np.linspace(-LIM, LIM, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    points = torch.tensor(np.stack([xx, yy], axis=-1).reshape(-1, 2), dtype=DTYPE, device=DEVICE)
    return xx, yy, points


def density_on_grid(log_prob_fn, grid_size=250):
    xx, yy, points = make_grid(grid_size)
    logp = log_prob_fn(points).detach().cpu().numpy().reshape(xx.shape)
    logp = logp - logp.max()
    density = np.exp(logp)
    density = density / np.maximum(density.sum(), 1e-12)
    return xx, yy, density


def _record_progress(progress, budget_steps, step_idx, samples, t0):
    if (step_idx + 1) in budget_steps:
        progress[step_idx + 1] = {
            "samples": samples.detach().clone().cpu().numpy(),
            "seconds": time.perf_counter() - t0,
        }


def _initial_samples(nb, sigma_max, adversarial_init=False):
    alpha = alpha_of(sigma_max)
    if not adversarial_init:
        return torch.randn(nb, DIM, dtype=DTYPE, device=DEVICE) * marginal_std(sigma_max)

    distances = torch.linalg.norm(MU - X_TRUE.unsqueeze(0), dim=1)
    center = MU[int(torch.argmax(distances).item())]
    x_clean = center.unsqueeze(0) + CLUSTER_STD * torch.randn(nb, DIM, dtype=DTYPE, device=DEVICE)
    return alpha * x_clean


def run_dps(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005, guidance_scale=2.0,
            budget_steps=None, adversarial_init=False):
    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])

    for i in trange(N, desc="DPS"):
        sig = sigmas[i].item()
        alpha = alphas[i].item()
        factor = schedule.factor_steps[i]
        scaling_factor = schedule.scaling_factor[i]

        x_cur = x.detach().requires_grad_(True)
        x0_hat = tweedie(x_cur, sig, create_graph=True)
        residual = (x0_hat @ A.T) - Y_OBS
        loss = (residual ** 2).sum()
        (ll_grad,) = torch.autograd.grad(loss, x_cur)
        ll_grad = ll_grad * 0.5 / (loss.detach().sqrt() + 1e-12)

        sc = (alpha * x0_hat.detach() - x_cur.detach()) / (alpha * sig) ** 2
        x = (
            scaling_factor * x_cur.detach()
            + factor * sc
            + np.sqrt(factor) * torch.randn_like(x)
            - guidance_scale * ll_grad.detach()
        )
        _record_progress(progress, budget_steps, i, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def _reverse_ode(x, sigma_start, num_steps=8):
    schedule = make_schedule(num_steps, sigma_start, 1e-3)
    sigs = schedule.sigmas
    alphas = schedule.alphas
    for k in range(num_steps):
        s = sigs[k].item()
        alpha = alphas[k].item()
        x0 = tweedie(x, s)
        sc = (alpha * x0 - x) / (alpha * s) ** 2
        x = schedule.scaling_factor[k] * x + schedule.factor_steps[k] * sc * 0.5
    return x.detach()


def _langevin_inner(x0_hat, y, sigma, num_steps=200, lr=5e-4):
    x = x0_hat.clone().detach()
    for _ in range(num_steps):
        x.requires_grad_(True)
        res = (x @ A.T) - y
        data_grad = (res @ A) / SIGMA_NOISE ** 2
        prior_grad = (x - x0_hat.detach()) / sigma ** 2
        grad = data_grad + prior_grad
        x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
    _check_divergence(x, lr, "daps_langevin")
    return x.detach()


def run_daps(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005, ode_steps=8,
             lgvd_steps=200, lgvd_lr=5e-4, budget_steps=None, adversarial_init=False):
    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])

    for step in trange(N, desc="DAPS"):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        alpha_next = alphas[step + 1].item()
        x0_hat = _reverse_ode(x, sigma, num_steps=ode_steps)
        try:
            x0y = _langevin_inner(x0_hat, Y_OBS, sigma, num_steps=lgvd_steps, lr=lgvd_lr)
        except DivergenceError as exc:
            _record_progress(progress, budget_steps, step, x0_hat, t0)
            return _diverged_result(x0_hat, progress, exc, step)
        x = _renoise_clean(x0y, sigma_next, alpha_next)
        _record_progress(progress, budget_steps, step, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def run_pula(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005, step_size=0.5,
             nb_langevin=10, budget_steps=None, adversarial_init=False):
    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    AtA = A_s.T @ A_s
    variances = make_langevin_sigmas(N, sigma_max, sigma_min) ** 2

    M0 = torch.inverse(AtA + torch.eye(DIM, dtype=DTYPE) / variances[0])
    mean = M0 @ (A_s.T @ y_s.unsqueeze(-1))
    if adversarial_init:
        x = _clean_from_noised(_initial_samples(nb, sigma_max, adversarial_init=True), alpha_of(sigma_max))
    else:
        x = mean.squeeze(-1) + torch.randn(nb, DIM, dtype=DTYPE) @ torch.linalg.cholesky(M0).T

    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])

    for i in trange(N, desc="pULA"):
        var = variances[i].item()
        sigma = np.sqrt(var)
        M = torch.inverse(AtA + torch.eye(DIM, dtype=DTYPE) / var)

        for _ in range(nb_langevin):
            score_prior = clean_score_fn(x, sigma)
            residual = y_s - x @ A_s.T
            lik_score = residual @ A_s
            grad = score_prior + lik_score
            conv_grad = (M @ grad.T).T

            n1 = torch.randn(A_s.shape[0], nb, dtype=DTYPE)
            n2 = torch.randn(DIM, nb, dtype=DTYPE)
            z = (M @ (A_s.T @ n1 + n2 / np.sqrt(var))).T
            x = x.detach() + (step_size / 2) * conv_grad + np.sqrt(step_size) * z

        try:
            _check_divergence(x, step_size, "pula")
        except DivergenceError as exc:
            _record_progress(progress, budget_steps, i, x, t0)
            return _diverged_result(x, progress, exc, i)
        _record_progress(progress, budget_steps, i, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def _pdaps_inner(
    x_init,
    x0_hat,
    y,
    sigma,
    ratio,
    *,
    num_steps=20,
    gamma=0.5,
    tau=None,
    lr_min_ratio=1.0,
    solve_lam_floor=0.0,
    gamma_schedule="constant",
    noise_tau=1.0,
    where="pdaps_langevin",
):
    tau = daps_tau_toy(SIGMA_NOISE) if tau is None else float(tau)
    lam_raw = 1.0 / sigma ** 2
    lam_target = lam_raw * tau ** 2
    lam_solve = max(lam_raw, float(solve_lam_floor))

    if gamma_schedule == "constant":
        schedule_scale = 1.0
    elif gamma_schedule == "lambda":
        schedule_scale = lam_raw
    else:
        raise ValueError(f"unknown gamma_schedule: {gamma_schedule!r}")

    gamma_eff = float(gamma) * (1.0 + float(ratio) * (float(lr_min_ratio) - 1.0)) * schedule_scale
    A_s = A / SIGMA_NOISE
    y_s = y / SIGMA_NOISE
    AtA = A_s.T @ A_s
    M = torch.inverse(AtA + lam_solve * torch.eye(DIM, dtype=DTYPE, device=DEVICE))
    x = x_init.clone().detach()
    x0_hat = x0_hat.detach()

    for _ in range(num_steps):
        residual = y_s - x @ A_s.T
        lik_score = residual @ A_s
        prior_score = -(x - x0_hat) * lam_target
        grad = lik_score + prior_score
        conv_grad = (M @ grad.T).T

        x = x.detach() + 0.5 * gamma_eff * conv_grad
        if noise_tau > 0:
            nb = x.shape[0]
            n1 = torch.randn(A_s.shape[0], nb, dtype=DTYPE, device=DEVICE)
            n2 = torch.randn(DIM, nb, dtype=DTYPE, device=DEVICE)
            z = (M @ (A_s.T @ n1 + np.sqrt(lam_solve) * n2)).T
            x = x + np.sqrt(gamma_eff * float(noise_tau)) * z

    _check_divergence(x, gamma_eff, where)
    return x.detach()


def run_pdaps(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005, ode_steps=8,
              lgvd_steps=20, lgvd_step_size=0.5, budget_steps=None, adversarial_init=False,
              inner_sigma_max=float("inf"), *, tau=None, gamma_schedule="constant",
              lr_min_ratio=1.0, solve_lam_floor=0.0, noise_tau=1.0):
    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])

    for step in trange(N, desc="P-DAPS"):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        alpha_next = alphas[step + 1].item()
        x0_hat = _reverse_ode(x, sigma, num_steps=ode_steps)
        if sigma > inner_sigma_max:
            x0y = x0_hat
        else:
            try:
                x0y = _pdaps_inner(
                    x0_hat,
                    x0_hat,
                    Y_OBS,
                    sigma,
                    step / max(1, N),
                    num_steps=lgvd_steps,
                    gamma=lgvd_step_size,
                    tau=tau,
                    gamma_schedule=gamma_schedule,
                    lr_min_ratio=lr_min_ratio,
                    solve_lam_floor=solve_lam_floor,
                    noise_tau=noise_tau,
                    where="pdaps_langevin",
                )
            except DivergenceError as exc:
                _record_progress(progress, budget_steps, step, x0_hat, t0)
                return _diverged_result(x0_hat, progress, exc, step)
        x = _renoise_clean(x0y, sigma_next, alpha_next)
        _record_progress(progress, budget_steps, step, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def run_pdaps_warm(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005,
                   ode_steps=8, lgvd_steps=20, lgvd_step_size=0.5,
                   warm_fraction=0.5, budget_steps=None, adversarial_init=False,
                   inner_sigma_max=float("inf"), *, tau=None, gamma_schedule="constant",
                   lr_min_ratio=1.0, solve_lam_floor=0.0, noise_tau=1.0):
    warm_fraction = float(np.clip(warm_fraction, 0.0, 1.0))
    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])
    x0y_prev = None

    for step in trange(N, desc=f"P-DAPS-warm (alpha={warm_fraction:.2f})"):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        alpha_next = alphas[step + 1].item()
        x0_hat = _reverse_ode(x.detach(), sigma, num_steps=ode_steps)
        if sigma > inner_sigma_max:
            x0y = x0_hat
        else:
            if x0y_prev is None:
                x_init = x0_hat
            else:
                x_init = warm_fraction * x0y_prev + (1.0 - warm_fraction) * x0_hat
            try:
                x0y = _pdaps_inner(
                    x_init,
                    x0_hat,
                    Y_OBS,
                    sigma,
                    step / max(1, N),
                    num_steps=lgvd_steps,
                    gamma=lgvd_step_size,
                    tau=tau,
                    gamma_schedule=gamma_schedule,
                    lr_min_ratio=lr_min_ratio,
                    solve_lam_floor=solve_lam_floor,
                    noise_tau=noise_tau,
                    where="pdaps_langevin_warm",
                )
            except DivergenceError as exc:
                _record_progress(progress, budget_steps, step, x0_hat, t0)
                return _diverged_result(x0_hat, progress, exc, step)
            x0y_prev = x0y.detach()
        x = _renoise_clean(x0y, sigma_next, alpha_next)
        _record_progress(progress, budget_steps, step, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def run_pdaps_adaptive(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005,
                       ode_steps=8, lgvd_steps=20, lgvd_step_size=0.5,
                       warm_fraction=0.5, budget_steps=None, adversarial_init=False,
                       eps=1e-8, *, tau=None, gamma_schedule="constant",
                       lr_min_ratio=1.0, solve_lam_floor=0.0, noise_tau=1.0):
    alpha_max = float(np.clip(warm_fraction, 0.0, 1.0))
    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    x_prev = None
    progress = {}
    gate_stats = []
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])
    m = max(1, A.shape[0])

    def residual_norm(u):
        return torch.linalg.vector_norm((u @ A.T) - Y_OBS, dim=1) / np.sqrt(m)

    for step in trange(N, desc=f"P-DAPS-adaptive (alpha_max={alpha_max:.2f})"):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        alpha_next = alphas[step + 1].item()
        x0_hat = _reverse_ode(x.detach(), sigma, num_steps=ode_steps)

        if x_prev is None:
            x_init = x0_hat
            gate_stats.append({
                "step": int(step),
                "alpha_mean": 0.0,
                "alpha_min": 0.0,
                "alpha_max": 0.0,
                "drift_mean": 0.0,
                "r_hat_mean": float(residual_norm(x0_hat).mean().detach().cpu()),
                "r_prev_mean": float("nan"),
            })
        else:
            r_hat = residual_norm(x0_hat)
            r_prev = residual_norm(x_prev)
            drift = torch.linalg.vector_norm(x_prev - x0_hat, dim=1)
            drift = drift / torch.linalg.vector_norm(x0_hat, dim=1).clamp_min(eps)
            gate = (r_hat / (r_hat + r_prev + eps)) / (1.0 + drift)
            alpha_t = (alpha_max * gate).clamp(0.0, alpha_max)
            x_init = alpha_t.view(-1, 1) * x_prev + (1.0 - alpha_t.view(-1, 1)) * x0_hat
            gate_stats.append({
                "step": int(step),
                "alpha_mean": float(alpha_t.mean().detach().cpu()),
                "alpha_min": float(alpha_t.min().detach().cpu()),
                "alpha_max": float(alpha_t.max().detach().cpu()),
                "drift_mean": float(drift.mean().detach().cpu()),
                "r_hat_mean": float(r_hat.mean().detach().cpu()),
                "r_prev_mean": float(r_prev.mean().detach().cpu()),
            })

        try:
            x0y = _pdaps_inner(
                x_init,
                x0_hat,
                Y_OBS,
                sigma,
                step / max(1, N),
                num_steps=lgvd_steps,
                gamma=lgvd_step_size,
                tau=tau,
                gamma_schedule=gamma_schedule,
                lr_min_ratio=lr_min_ratio,
                solve_lam_floor=solve_lam_floor,
                noise_tau=noise_tau,
                where="pdaps_langevin_adaptive",
            )
        except DivergenceError as exc:
            _record_progress(progress, budget_steps, step, x0_hat, t0)
            result = _diverged_result(x0_hat, progress, exc, step)
            result["gate_stats"] = gate_stats
            return result
        x_prev = x0y.detach()
        x = _renoise_clean(x0y, sigma_next, alpha_next)
        _record_progress(progress, budget_steps, step, x, t0)

    return {
        "final": x.detach().cpu().numpy(),
        "progress": progress,
        "diverged_at": None,
        "gate_stats": gate_stats,
    }


def run_pdaps_v3(nb=3000, N=300, sigma_max=1.5, sigma_min=0.005, ode_steps=8,
                 *, null_blend=1.0, inner_sigma_max=float("inf"), null_rcond=0.05,
                 tau=None, sigma_stop_truncate=None,
                 budget_steps=None, adversarial_init=False):
    """Toy port of ``algo/pdaps_v3.py::PDAPSv3`` -- exact-split decoupled annealing.

    Mirrors the production sampler the rescue ablations converged on: the iterative
    inner Langevin block is replaced by the *exact* solution of its Gaussian target,
    split into a data-informed subspace (solved exactly) and a data-null subspace
    (filled by a convex blend of the denoiser anchor and the carried iterate). The
    only difference from the MRI version is the operator: here the whitened forward
    op ``A_s = A / sigma_noise`` is dense and small, so the per-direction solve is
    done in its SVD basis instead of by CG over the k-space mask.

    Per outer level ``sigma``:
      1. denoise ``x_t -> x0hat`` with the reverse ODE (the prior anchor).
      2. for ``sigma <= inner_sigma_max``, in the right-singular basis ``V`` of ``A_s``:
           - DATA directions (singular value ``s_i > null_rcond * s_max``) get the exact
             inner-target solution
                 ``c_i = (s_i (U^T y_s)_i + lam d_i) / (s_i^2 + lam)``,
             with ``lam = tau^2 / sigma^2`` the prior:anchor weight (toy ``tau = 1``;
             see ``daps_tau_toy``). This is the deterministic limit of the inner
             Langevin block.
           - NULL directions (``s_i <= null_rcond * s_max``, where the data target is
             flat) get ``null_blend * x0hat + (1 - null_blend) * x_prev``.
      3. re-noise the clean estimate to the next level -- the stochastic OUTER step
         that carries posterior diversity (the inner block is deterministic by design).

    ``null_blend`` is the single operating-point dial (the diversity-distortion
    frontier): ``1.0`` = pure denoiser anchor (identical to the un-split exact MAP
    solve, max diversity), ``0.0`` = carry the previous iterate in the null space
    (collapses toward the point estimate). ``A`` is square in every toy scenario, so
    ``U, V`` are both ``DIM x DIM``.
    """
    tau = daps_tau_toy(SIGMA_NOISE) if tau is None else float(tau)
    null_blend = float(np.clip(null_blend, 0.0, 1.0))

    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    U, s, Vh = torch.linalg.svd(A_s)
    V = Vh.transpose(-1, -2)
    Uty = U.transpose(-1, -2) @ y_s
    null_mask = (s < null_rcond * s.max()).unsqueeze(0)

    schedule = make_schedule(N, sigma_max, sigma_min)
    sigmas = schedule.sigmas
    alphas = schedule.alphas
    x = _initial_samples(nb, sigma_max, adversarial_init=adversarial_init)
    x_prev = None
    progress = {}
    t0 = time.perf_counter()
    budget_steps = set(budget_steps or [])

    for step in trange(N, desc=f"P-DAPS-v3 (nb={null_blend:.2f})"):
        sigma = sigmas[step].item()
        if sigma_stop_truncate is not None and sigma < sigma_stop_truncate and x_prev is not None:
            x = x_prev
            break
        sigma_next = sigmas[step + 1].item()
        alpha_next = alphas[step + 1].item()
        x0_hat = _reverse_ode(x, sigma, num_steps=ode_steps)

        if sigma <= inner_sigma_max:
            x_carry = x0_hat if x_prev is None else x_prev
            lam = tau ** 2 / sigma ** 2
            d = x0_hat @ V                       # anchor coords in V-basis (nb, DIM)
            p = x_carry @ V                      # carried-iterate coords
            data_c = (s.unsqueeze(0) * Uty.unsqueeze(0) + lam * d) / (s.unsqueeze(0) ** 2 + lam)
            null_c = null_blend * d + (1.0 - null_blend) * p
            c = torch.where(null_mask, null_c, data_c)
            x0y = (c @ V.transpose(-1, -2)).detach()
        else:
            x0y = x0_hat

        try:
            _check_divergence(x0y, 0.0, "pdaps_v3")
        except DivergenceError as exc:
            _record_progress(progress, budget_steps, step, x0_hat, t0)
            return _diverged_result(x0_hat, progress, exc, step)

        x_prev = x0y
        x = _renoise_clean(x0y, sigma_next, alpha_next)
        _record_progress(progress, budget_steps, step, x, t0)

    return {"final": x.detach().cpu().numpy(), "progress": progress, "diverged_at": None}


def summarize_samples(samples):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    top_mass = float(np.mean(samples[:, 1] > 0))
    mode_masses = mode_masses_2d(samples)
    return {"mean": mean, "std": std, "top_mass": top_mass, "mode_masses": mode_masses}


def mode_masses_2d(samples):
    centers = MU[:, :2].detach().cpu().numpy()
    distances = ((samples[:, None, :2] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = distances.argmin(axis=1)
    return np.bincount(labels, minlength=len(centers)).astype(float) / max(1, len(labels))


def samples_in_view(samples, lim=None):
    lim = LIM if lim is None else lim
    finite = np.isfinite(samples).all(axis=1)
    bounded = (np.abs(samples[:, 0]) <= lim) & (np.abs(samples[:, 1]) <= lim)
    mask = finite & bounded
    return samples[mask], float(mask.mean()) if len(mask) else 0.0


def compare_to_truth(samples, gt_summary):
    in_view, frac = samples_in_view(samples)
    if len(in_view) < 40 or frac < 0.10:
        return {
            "fit_error": np.nan,
            "upper_mode_error": np.nan,
            "std_x2_error": np.nan,
            "mean_rmse": np.nan,
            "mode_mass_l1": np.nan,
        }
    summary = summarize_samples(in_view)
    mean_rmse = float(np.sqrt(np.mean((summary["mean"] - gt_summary["mean"]) ** 2)))
    mode_mass_l1 = float(np.abs(summary["mode_masses"] - gt_summary["mode_masses"]).sum())
    if DIM > 2:
        return {
            "fit_error": mean_rmse + mode_mass_l1,
            "upper_mode_error": mode_mass_l1,
            "std_x2_error": mean_rmse,
            "mean_rmse": mean_rmse,
            "mode_mass_l1": mode_mass_l1,
        }
    upper_err = abs(summary["top_mass"] - gt_summary["top_mass"])
    std_err = abs(summary["std"][1] - gt_summary["std"][1])
    return {
        "fit_error": upper_err + std_err,
        "upper_mode_error": upper_err,
        "std_x2_error": std_err,
        "mean_rmse": mean_rmse,
        "mode_mass_l1": mode_mass_l1,
    }


def _operator_space_metrics(samples_np):
    """Operator-space discrepancies between sampler-pushforward A*x and the
    observation y_obs. Two quantities returned:

      data_chi2_mean : chi-squared residual of the sampler *mean*,
                       ||y - A*mean(x)||^2 / sigma_noise^2
                       (close to 0 when sampler mean lies on the manifold)
      neg_log_evidence : -log mean_i exp(-||y - A*x_i||^2 / (2 sigma^2)),
                         a log-evidence-like score (lower = better sampler
                         coverage of explanations of y_obs)

    Both metrics are computed in operator space, not posterior space.
    """
    if A is None or Y_OBS is None or SIGMA_NOISE is None:
        return {"data_chi2_mean": np.nan, "neg_log_evidence": np.nan}
    finite_mask = np.isfinite(samples_np).all(axis=1)
    if not finite_mask.any():
        return {"data_chi2_mean": np.nan, "neg_log_evidence": np.nan}
    x = torch.as_tensor(samples_np[finite_mask], dtype=DTYPE)
    y = Y_OBS
    sig2 = float(SIGMA_NOISE) ** 2
    mean_x = x.mean(dim=0)
    resid_mean = y - A @ mean_x
    chi2_mean = float((resid_mean ** 2).sum().item() / sig2)
    pred_y = x @ A.T
    sq = ((y[None, :] - pred_y) ** 2).sum(dim=1) / (2.0 * sig2)
    log_n = float(np.log(x.shape[0]))
    nle = float((torch.logsumexp(-sq, dim=0) - log_n).item())
    return {"data_chi2_mean": chi2_mean, "neg_log_evidence": -nle}


def _summarize_method_result(method, result, gt_summary):
    final = result["final"]
    in_view, frac = samples_in_view(final)
    fit = compare_to_truth(final, gt_summary)
    summary = summarize_samples(in_view) if len(in_view) >= 40 and frac >= 0.10 else None
    finite = np.isfinite(final).all(axis=1)
    gt_var_x2 = gt_summary["std"][1] ** 2
    method_var_x2 = summary["std"][1] ** 2 if summary else np.nan
    diverged_at = result.get("diverged_at")
    diverged = diverged_at is not None
    op = _operator_space_metrics(final)
    mr = fit["mean_rmse"]
    sx = fit["std_x2_error"]
    bures2 = float(np.sqrt(mr ** 2 + sx ** 2)) if np.isfinite(mr) and np.isfinite(sx) else np.nan
    return {
        "method": method,
        "fit_error": np.nan if diverged else fit["fit_error"],
        "upper_mode_error": np.nan if diverged else fit["upper_mode_error"],
        "upper_mode_mass_error": np.nan if diverged else fit["upper_mode_error"],
        "std_x2_error": np.nan if diverged else fit["std_x2_error"],
        "mean_rmse": np.nan if diverged else fit["mean_rmse"],
        "mode_mass_l1": np.nan if diverged else fit["mode_mass_l1"],
        "runtime_s": result["progress"][max(result["progress"])]["seconds"] if result["progress"] else np.nan,
        "upper_frac": np.nan if diverged or summary is None else summary["top_mass"],
        "std_x2": np.nan if diverged or summary is None else summary["std"][1],
        "gt_var_x2": gt_var_x2,
        "gt_top_mass": gt_summary["top_mass"],
        "gt_std_x2": gt_summary["std"][1],
        "method_var_x2": np.nan if diverged else method_var_x2,
        "posterior_var_x2_error": np.nan if diverged or not np.isfinite(method_var_x2) else abs(method_var_x2 - gt_var_x2),
        "in_view": np.nan if diverged else frac,
        "nan_count": int((~finite).sum()),
        "diverged": 1 if diverged else 0,
        "diverged_at": diverged_at if diverged else "",
        "diverged_step_size": result.get("diverged_step_size", ""),
        "bures2_x2": np.nan if diverged else bures2,
        "data_chi2_mean": np.nan if diverged else op["data_chi2_mean"],
        "neg_log_evidence": np.nan if diverged else op["neg_log_evidence"],
    }


def _warm_method_name(warm_fraction):
    return f"P-DAPS-warm a={warm_fraction:.2f}"


def _filtered_warm_fractions(warm_fractions, *, drop_zero=True):
    filtered = []
    for warm_fraction in warm_fractions:
        warm_fraction = float(warm_fraction)
        if drop_zero and np.isclose(warm_fraction, 0.0):
            continue
        filtered.append(warm_fraction)
    return tuple(filtered)


def _run_baseline_methods(nb, rp, budget_steps, adversarial_init=False):
    N = rp["N"]
    return {
        "DPS": run_dps(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            guidance_scale=rp["dps_guidance_scale"], budget_steps=budget_steps,
            adversarial_init=adversarial_init,
        ),
        "DAPS": run_daps(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["daps_ode_steps"], lgvd_steps=rp["daps_langevin_steps"],
            lgvd_lr=rp["daps_langevin_lr"], budget_steps=budget_steps,
            adversarial_init=adversarial_init,
        ),
        "pULA": run_pula(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            step_size=rp["pula_step_size"], nb_langevin=rp["pula_nb_langevin"],
            budget_steps=budget_steps, adversarial_init=adversarial_init,
        ),
        "P-DAPS": run_pdaps(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=rp["pdaps_langevin_step_size"], budget_steps=budget_steps,
            adversarial_init=adversarial_init,
        ),
    }


def _measurement_legend_handles():
    return [
        Patch(facecolor="#6baed6", edgecolor="none", alpha=0.12, label="data-consistent region"),
        Line2D([0], [0], color="#3182bd", lw=1.0, label="$y_1$-implied center in $x_1$"),
        Line2D([0], [0], marker="+", color="#4d4d4d", lw=0, markersize=8, label="prior centers"),
        Line2D([0], [0], marker="x", color="#111111", lw=0, markersize=7, label="$x_{\\mathrm{true}}$"),
    ]


def _observability_note():
    note = (
        "Black x: latent true state used to generate data. "
        "Blue line/band: the region in x-space favored by the observed data y = Ax + epsilon. "
        "Methods never observe x_true directly; they only see the prior and the data term. "
        "Here y_2 is weakly informative because A_22 is small."
    )
    bias = CURRENT_SCENARIO["score_bias"]["bias"]
    if any(abs(v) > 0 for v in bias):
        note += " This toy also injects mild high-noise score bias."
    return note


def _scenario_detail_text():
    x_true_vals = ", ".join(f"{v:.3f}" for v in X_TRUE.detach().cpu().numpy())
    y_obs_vals = ", ".join(f"{v:.3f}" for v in Y_OBS.detach().cpu().numpy())
    x1_center = float(Y_OBS[0].item()) / CURRENT_SCENARIO["a_diag"][0]
    a11, a22 = CURRENT_SCENARIO["a_diag"]
    text = (
        f"x_true = ({x_true_vals})    "
        f"y_obs = ({y_obs_vals})    "
        f"y_1 / A_11 = {x1_center:.3f}    "
        f"A = diag({a11}, {a22})    "
        f"sigma_n = {SIGMA_NOISE}"
    )
    bias = CURRENT_SCENARIO["score_bias"]["bias"]
    if any(abs(v) > 0 for v in bias):
        text += f"    score bias = {bias}"
    return text


def _style_density_axis(ax, ylabel=None):
    x_obs = float(Y_OBS[0].item())
    ax.set_facecolor("white")
    ax.axvspan(x_obs - SIGMA_NOISE, x_obs + SIGMA_NOISE, color="#6baed6", alpha=0.12, lw=0)
    ax.axvline(x_obs, color="#3182bd", lw=0.9, alpha=0.9)
    ax.scatter(CLUSTER_CENTERS[:, 0].cpu().numpy(), CLUSTER_CENTERS[:, 1].cpu().numpy(),
               marker="+", s=18, linewidths=0.7, color="#4d4d4d", alpha=0.75, zorder=4)
    ax.scatter([X_TRUE[0].item()], [X_TRUE[1].item()],
               marker="x", s=28, linewidths=1.0, color="#111111", alpha=0.95, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel("$x_1$", fontsize=10, color="#333333")
    ax.set_ylabel(ylabel if ylabel else "$x_2$", fontsize=11, color="#333333")
    ax.grid(color="#d9d9d9", lw=0.5, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#8c8c8c")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors="#4d4d4d")


def plot_density_field(ax, density, xx, yy, title=None, ylabel=None, contour=None):
    ax.contourf(xx, yy, density, levels=18, cmap=DENSITY_CMAP)
    if contour is not None:
        levels = np.quantile(contour[contour > 0], [0.70, 0.85, 0.94])
        ax.contour(xx, yy, contour, levels=levels, colors="#2c3e50", linewidths=0.9, alpha=0.9)
    _style_density_axis(ax, ylabel=ylabel)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def plot_sample_density(ax, samples, title=None, ylabel=None, gt_contour=None):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[-LIM, LIM], [-LIM, LIM]],
              bins=100, density=True, cmap=DENSITY_CMAP, cmin=1e-6)
    if gt_contour is not None:
        xx, yy, density = gt_contour
        levels = np.quantile(density[density > 0], [0.70, 0.85, 0.94])
        ax.contour(xx, yy, density, levels=levels, colors="#2c3e50", linewidths=0.8, alpha=0.85)
    _style_density_axis(ax, ylabel=ylabel)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def plot_geometry(filename, grid_size=250):
    xx, yy, prior_density = density_on_grid(lambda x: _gmm_log_prob(x, sigma=0.0), grid_size)
    _, _, likelihood_density = density_on_grid(
        lambda x: -0.5 * ((Y_OBS.unsqueeze(0) - (x @ A.T)) ** 2).sum(-1) / SIGMA_NOISE ** 2,
        grid_size,
    )
    _, _, posterior_density = density_on_grid(posterior_log_prob, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    fig.patch.set_facecolor("white")
    plot_density_field(axes[0], prior_density, xx, yy, title="Prior")
    plot_density_field(axes[1], likelihood_density, xx, yy, title="Exact Likelihood")
    plot_density_field(axes[2], posterior_density, xx, yy, title="Exact Posterior", contour=posterior_density)

    fig.suptitle(f"{CURRENT_SCENARIO['name']}: prior geometry, exact likelihood, and exact posterior",
                 fontsize=13, y=1.09, color="#1b1b1b")
    fig.text(0.5, 0.995, _scenario_detail_text(), ha="center", va="center", fontsize=9, color="#333333")
    fig.text(0.5, 0.955, CURRENT_SCENARIO["summary"], ha="center", va="center", fontsize=9, color="#444444")
    fig.text(0.5, 0.918, _observability_note(), ha="center", va="center", fontsize=9, color="#444444")
    fig.legend(handles=_measurement_legend_handles(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.88), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_algorithm_comparison(results, gt_samples, filename, grid_size=250):
    methods = list(results.keys())
    xx, yy, posterior_density = density_on_grid(posterior_log_prob, grid_size)
    gt_contour = (xx, yy, posterior_density)

    fig, axes = plt.subplots(2, 1 + len(methods), figsize=(3.0 * (1 + len(methods)), 6.0))
    fig.patch.set_facecolor("white")
    top_row, bottom_row = axes[0], axes[1]

    plot_sample_density(top_row[0], gt_samples, title="True Posterior", ylabel="Final Samples")
    for idx, method in enumerate(methods, start=1):
        samples = results[method]["final"]
        in_view, frac = samples_in_view(samples)
        if len(in_view) >= 40 and frac >= 0.10:
            plot_sample_density(top_row[idx], in_view, title=method, gt_contour=gt_contour)
        else:
            _style_density_axis(top_row[idx])
            top_row[idx].set_title(method, fontsize=12, fontweight="bold")
            top_row[idx].text(0.5, 0.5, f"off-window\n{frac:.1%} in view",
                              transform=top_row[idx].transAxes, ha="center", va="center",
                              fontsize=10, color="#7f0000")

    x2_gt = gt_samples[:, 1]
    bins = np.linspace(-LIM, LIM, 70)
    gt_summary = summarize_samples(gt_samples)

    def plot_x2_marginal(ax, samples, title=None, ylabel=None):
        in_view, frac = samples_in_view(samples)
        ax.set_facecolor("white")
        ax.hist(x2_gt, bins=bins, density=True, histtype="step", lw=1.4, color="#2c3e50", label="True posterior")
        if len(in_view) >= 40 and frac >= 0.10:
            summary = summarize_samples(in_view)
            ax.hist(in_view[:, 1], bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color="#c8553d", label="Method")
            text = f"upper-mode fraction {summary['top_mass']:.2f}\ntrue posterior {gt_summary['top_mass']:.2f}"
            text_color = "#2c3e50"
        else:
            text = f"off-window\n{frac:.1%} in view"
            text_color = "#7f0000"
        ax.axvline(0.0, color="#9e9e9e", lw=0.8, alpha=0.8)
        ax.grid(color="#e0e0e0", lw=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#8c8c8c")
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=8, colors="#4d4d4d")
        ax.set_xlim(-LIM, LIM)
        ax.set_xlabel("$x_2$", fontsize=10, color="#333333")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        if title:
            ax.set_title(title, fontsize=11, color="#1b1b1b")
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=7, color=text_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdbdbd", lw=0.6, alpha=0.9))

    plot_x2_marginal(bottom_row[0], gt_samples, ylabel="$x_2$ Marginal")
    for idx, method in enumerate(methods, start=1):
        plot_x2_marginal(bottom_row[idx], results[method]["final"], title=method)

    handles, labels = bottom_row[1].get_legend_handles_labels()
    measurement_handles = _measurement_legend_handles()
    fig.legend(measurement_handles + handles, [h.get_label() for h in measurement_handles] + labels,
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.90), fontsize=9)
    fig.suptitle(f"{CURRENT_SCENARIO['name']}: final 2D densities and informative marginal along the weakly observed axis",
                 fontsize=13, y=1.15, color="#1b1b1b")
    fig.text(0.5, 1.08, _scenario_detail_text(), ha="center", va="center", fontsize=9, color="#333333")
    fig.text(0.5, 1.04, CURRENT_SCENARIO["summary"], ha="center", va="center", fontsize=9, color="#444444")
    fig.text(0.5, 1.00, _observability_note(), ha="center", va="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_progress_curves(results, gt_samples, budgets, filename):
    methods = list(results.keys())
    gt_summary = summarize_samples(gt_samples)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    fig.patch.set_facecolor("white")

    for method in methods:
        xs, times, errs = [], [], []
        for budget in budgets:
            progress = results[method]["progress"].get(budget)
            if progress is None:
                continue
            xs.append(budget)
            times.append(progress["seconds"])
            errs.append(compare_to_truth(progress["samples"], gt_summary)["fit_error"])
        color = METHOD_COLORS["P-DAPS-warm"] if method.startswith("P-DAPS-warm") else METHOD_COLORS[method]
        axes[0].plot(xs, times, marker="o", lw=1.6, ms=4, color=color, label=method)
        axes[1].plot(xs, errs, marker="o", lw=1.6, ms=4, color=color, label=method)

    axes[0].set_title("Latency vs Outer Steps", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("outer reverse steps")
    axes[0].set_ylabel("seconds")
    axes[1].set_title("Marginal Fit Error vs Outer Steps", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("outer reverse steps")
    axes[1].set_ylabel(r"$|$upper-mode fraction error$| + |$std$(x_2)$ error$|$")
    for ax in axes[:2]:
        ax.set_facecolor("white")
        ax.grid(color="#e0e0e0", lw=0.6, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#8c8c8c")
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=8, colors="#4d4d4d")

    summary_ax = axes[2]
    summary_ax.axis("off")
    summary_ax.set_title("Final Budget Summary", fontsize=12, fontweight="bold", color="#1b1b1b")
    final_budget = max(budgets)
    rows = []
    for method in methods:
        final_samples = results[method]["final"]
        in_view, frac = samples_in_view(final_samples)
        fit = compare_to_truth(final_samples, gt_summary)["fit_error"]
        summary = summarize_samples(in_view) if len(in_view) >= 40 and frac >= 0.10 else None
        seconds = results[method]["progress"].get(final_budget, {}).get("seconds", np.nan)
        rows.append([method,
                     f"{seconds:.2f}" if np.isfinite(seconds) else "n/a",
                     f"{fit:.3f}" if np.isfinite(fit) else "n/a",
                     f"{summary['top_mass']:.2f}" if summary else "n/a",
                     f"{frac:.2f}"])
    table = summary_ax.table(cellText=rows, colLabels=["Method", "sec", "fit err", "upper frac", "in view"],
                             cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c7c7c7")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#f3f3f3" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold", color="#333333")
        elif col == 0:
            method = cell.get_text().get_text()
            color = METHOD_COLORS["P-DAPS-warm"] if method.startswith("P-DAPS-warm") else METHOD_COLORS[method]
            cell.set_text_props(color=color, weight="bold")

    fig.suptitle(f"{CURRENT_SCENARIO['name']}: progress over outer reverse steps", fontsize=13, y=1.13, color="#1b1b1b")
    fig.text(0.5, 1.05, _scenario_detail_text(), ha="center", va="center", fontsize=9, color="#333333")
    fig.text(0.5, 1.00, CURRENT_SCENARIO["summary"], ha="center", va="center", fontsize=9, color="#444444")
    legend_handles, legend_labels = [], []
    for ax in axes[:2]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_results(results, gt_samples):
    gt_summary = summarize_samples(gt_samples)
    return [_summarize_method_result(method, results[method], gt_summary) for method in results]


def print_summary(rows, gt_samples):
    gt_summary = summarize_samples(gt_samples)
    print(f"\n{'=' * 72}")
    print(f"  {CURRENT_SCENARIO['name']}")
    print(f"  GT: upper-mode fraction={gt_summary['top_mass']:.3f}, std(x2)={gt_summary['std'][1]:.4f}")
    print(f"{'-' * 72}")
    print(f"  {'Method':8s} {'fit_err':>8s} {'upper_err':>10s} {'std_err':>9s} {'sec':>7s} {'in_view':>8s}")
    for row in rows:
        print(f"  {row['method']:8s} {row['fit_error']:8.4f} {row['upper_mode_error']:10.4f} {row['std_x2_error']:9.4f} {row['runtime_s']:7.2f} {row['in_view']:8.2f}")
    print(f"{'=' * 72}")


def plot_warm_sweep(rows, baseline_rows, filename):
    alphas = np.array([row["warm_fraction"] for row in rows], dtype=float)
    fit_err = np.array([row["fit_error"] for row in rows], dtype=float)
    upper_err = np.array([row["upper_mode_error"] for row in rows], dtype=float)
    std_err = np.array([row["std_x2_error"] for row in rows], dtype=float)
    runtime = np.array([row["runtime_s"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    fig.patch.set_facecolor("white")

    series = [
        ("Fit error", fit_err, "#1b9e77", r"$|$upper-mode fraction error$| + |$std$(x_2)$ error$|$"),
        ("Error split", None, None, "component error"),
        ("Runtime", runtime, "#2c7fb8", "seconds"),
    ]

    axes[0].plot(alphas, fit_err, marker="o", lw=1.8, ms=5, color=series[0][2])
    axes[0].set_title(series[0][0], fontsize=12, fontweight="bold")
    axes[0].set_ylabel(series[0][3])

    axes[1].plot(alphas, upper_err, marker="o", lw=1.8, ms=5, color="#c8553d", label="upper-mode error")
    axes[1].plot(alphas, std_err, marker="o", lw=1.8, ms=5, color="#7f3c8d", label="std(x2) error")
    axes[1].set_title(series[1][0], fontsize=12, fontweight="bold")
    axes[1].set_ylabel(series[1][3])
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(alphas, runtime, marker="o", lw=1.8, ms=5, color=series[2][2])
    axes[2].set_title(series[2][0], fontsize=12, fontweight="bold")
    axes[2].set_ylabel(series[2][3])

    for baseline in baseline_rows:
        label = baseline["method"]
        color = METHOD_COLORS[label]
        axes[0].axhline(baseline["fit_error"], color=color, lw=1.0, ls="--", alpha=0.8)
        axes[2].axhline(baseline["runtime_s"], color=color, lw=1.0, ls="--", alpha=0.8, label=label)

    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xlabel("warm_fraction")
        ax.set_xticks(alphas)
        ax.grid(color="#e0e0e0", lw=0.6, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#8c8c8c")
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=8, colors="#4d4d4d")
    axes[2].legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"{CURRENT_SCENARIO['name']}: P-DAPS warm-fraction sweep", fontsize=13, y=1.08, color="#1b1b1b")
    fig.text(0.5, 0.99, _scenario_detail_text(), ha="center", va="center", fontsize=9, color="#333333")
    fig.text(0.5, 0.95, CURRENT_SCENARIO["summary"], ha="center", va="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()


def print_warm_sweep_summary(rows, baseline_rows, gt_samples):
    gt_summary = summarize_samples(gt_samples)
    print(f"\n{'=' * 88}")
    print(f"  {CURRENT_SCENARIO['name']}  |  P-DAPS warm-fraction sweep")
    print(f"  GT: upper-mode fraction={gt_summary['top_mass']:.3f}, std(x2)={gt_summary['std'][1]:.4f}")
    print(f"{'-' * 88}")
    print("  Baselines")
    print(f"  {'method':12s} {'fit_err':>8s} {'upper_err':>10s} {'std_err':>9s} {'sec':>7s} {'in_view':>8s}")
    for row in baseline_rows:
        print(
            f"  {row['method']:12s} {row['fit_error']:8.4f} {row['upper_mode_error']:10.4f} "
            f"{row['std_x2_error']:9.4f} {row['runtime_s']:7.2f} {row['in_view']:8.2f}"
        )
    print(f"{'-' * 88}")
    print("  Warm sweep")
    print(f"  {'alpha':>5s} {'fit_err':>8s} {'upper_err':>10s} {'std_err':>9s} {'sec':>7s} {'in_view':>8s}")
    for row in rows:
        print(
            f"  {row['warm_fraction']:5.2f} {row['fit_error']:8.4f} {row['upper_mode_error']:10.4f} "
            f"{row['std_x2_error']:9.4f} {row['runtime_s']:7.2f} {row['in_view']:8.2f}"
        )
    print(f"{'=' * 88}")


def run_scenario(key, nb=2500, gt_samples_n=160_000, make_plots=True, print_rows=True):
    configure_scenario(key)
    rp = CURRENT_SCENARIO["run_params"]
    N = rp["N"]
    budgets = tuple(b for b in (20, 40, 60, 80, N) if b <= N)

    if make_plots:
        plot_geometry(filename=f"{CURRENT_SCENARIO_KEY}_geometry.png")
    gt = sample_ground_truth(n_samples=gt_samples_n)
    results = _run_baseline_methods(nb=nb, rp=rp, budget_steps=budgets)
    results["P-DAPS-warm"] = run_pdaps_warm(
        nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
        ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
        lgvd_step_size=rp["pdaps_langevin_step_size"], warm_fraction=rp["pdaps_warm_fraction"],
        budget_steps=budgets,
    )
    if make_plots:
        plot_algorithm_comparison(results, gt, filename=f"{CURRENT_SCENARIO_KEY}_algorithms.png")
        plot_progress_curves(results, gt, budgets, filename=f"{CURRENT_SCENARIO_KEY}_progress.png")
    rows = summarize_results(results, gt)
    if print_rows:
        print_summary(rows, gt)
    return rows


def run_warm_fraction_sweep(key, warm_fractions=WARM_SWEEP_FRACTIONS, nb=2500, gt_samples_n=160_000,
                            make_plot=True, print_rows=True):
    configure_scenario(key)
    rp = CURRENT_SCENARIO["run_params"]
    N = rp["N"]
    warm_fractions = _filtered_warm_fractions(warm_fractions)
    gt = sample_ground_truth(n_samples=gt_samples_n)
    gt_summary = summarize_samples(gt)
    baseline_results = _run_baseline_methods(nb=nb, rp=rp, budget_steps=(N,))
    baseline_rows = [_summarize_method_result(method, baseline_results[method], gt_summary) for method in baseline_results]

    rows = []
    for warm_fraction in warm_fractions:
        method = _warm_method_name(warm_fraction)
        result = run_pdaps_warm(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=rp["pdaps_langevin_step_size"], warm_fraction=warm_fraction,
            budget_steps=(N,),
        )
        row = _summarize_method_result(method, result, gt_summary)
        row["warm_fraction"] = float(warm_fraction)
        rows.append(row)

    rows.sort(key=lambda row: row["warm_fraction"])
    if print_rows:
        print_warm_sweep_summary(rows, baseline_rows, gt)
    if make_plot:
        plot_warm_sweep(rows, baseline_rows, filename=f"{CURRENT_SCENARIO_KEY}_pdaps_warm_sweep.png")
    return {"baseline_rows": baseline_rows, "warm_rows": rows}


def _aggregate_rows(row_groups, key_field):
    ordered_keys = []
    seen = set()
    for rows in row_groups:
        for row in rows:
            key = row[key_field]
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    aggregated = []
    metric_names = (
        "fit_error", "upper_mode_error", "std_x2_error", "runtime_s", "in_view",
        "upper_mode_mass_error", "nan_count", "method_var_x2", "posterior_var_x2_error",
    )
    for key in ordered_keys:
        group = [row for rows in row_groups for row in rows if row[key_field] == key]
        summary = {key_field: key, "n_runs": len(group)}
        for metric in metric_names:
            values = np.array([row[metric] for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values))
        aggregated.append(summary)
    return aggregated


def print_repeat_compare_summary(aggregated_rows):
    print(f"\n{'=' * 96}")
    print(f"  {CURRENT_SCENARIO['name']}  |  repeated compare summary")
    print(f"{'-' * 96}")
    print(f"  {'Method':12s} {'fit mean':>9s} {'fit std':>8s} {'upper mean':>10s} {'stdx2 mean':>10s} {'sec mean':>9s}")
    for row in aggregated_rows:
        print(
            f"  {row['method']:12s} {row['fit_error_mean']:9.4f} {row['fit_error_std']:8.4f} "
            f"{row['upper_mode_error_mean']:10.4f} {row['std_x2_error_mean']:10.4f} {row['runtime_s_mean']:9.2f}"
        )
    print(f"{'=' * 96}")


def print_repeat_warm_sweep_summary(baseline_agg, warm_agg):
    print(f"\n{'=' * 108}")
    print(f"  {CURRENT_SCENARIO['name']}  |  repeated warm-sweep summary")
    print(f"{'-' * 108}")
    print("  Baselines")
    print(f"  {'Method':12s} {'fit mean':>9s} {'fit std':>8s} {'upper mean':>10s} {'stdx2 mean':>10s} {'sec mean':>9s}")
    for row in baseline_agg:
        print(
            f"  {row['method']:12s} {row['fit_error_mean']:9.4f} {row['fit_error_std']:8.4f} "
            f"{row['upper_mode_error_mean']:10.4f} {row['std_x2_error_mean']:10.4f} {row['runtime_s_mean']:9.2f}"
        )
    print(f"{'-' * 108}")
    print("  Warm sweep")
    print(f"  {'alpha':>5s} {'fit mean':>9s} {'fit std':>8s} {'upper mean':>10s} {'stdx2 mean':>10s} {'sec mean':>9s}")
    for row in warm_agg:
        print(
            f"  {row['warm_fraction']:5.2f} {row['fit_error_mean']:9.4f} {row['fit_error_std']:8.4f} "
            f"{row['upper_mode_error_mean']:10.4f} {row['std_x2_error_mean']:10.4f} {row['runtime_s_mean']:9.2f}"
        )
    print(f"{'=' * 108}")


def run_repeated_compare(key, repeats=5, base_seed=42, nb=2500, gt_samples_n=160_000):
    all_rows = []
    for rep in range(repeats):
        set_global_seed(base_seed + rep)
        rows = run_scenario(key, nb=nb, gt_samples_n=gt_samples_n, make_plots=False, print_rows=False)
        all_rows.append(rows)
    aggregated = _aggregate_rows(all_rows, key_field="method")
    print_repeat_compare_summary(aggregated)
    return aggregated


def run_repeated_warm_sweep(key, warm_fractions=WARM_SWEEP_FRACTIONS, repeats=5, base_seed=42,
                            nb=2500, gt_samples_n=160_000):
    warm_fractions = _filtered_warm_fractions(warm_fractions)
    baseline_groups = []
    warm_groups = []
    for rep in range(repeats):
        set_global_seed(base_seed + rep)
        summary = run_warm_fraction_sweep(
            key, warm_fractions=warm_fractions, nb=nb, gt_samples_n=gt_samples_n,
            make_plot=False, print_rows=False,
        )
        baseline_groups.append(summary["baseline_rows"])
        warm_groups.append(summary["warm_rows"])
    baseline_agg = _aggregate_rows(baseline_groups, key_field="method")
    warm_agg = _aggregate_rows(warm_groups, key_field="warm_fraction")
    print_repeat_warm_sweep_summary(baseline_agg, warm_agg)
    return {"baseline_rows": baseline_agg, "warm_rows": warm_agg}


def main():
    parser = argparse.ArgumentParser(description="Toy 2D comparisons for DPS, DAPS, pULA, and P-DAPS variants.")
    parser.add_argument(
        "--mode",
        choices=("compare", "warm-sweep", "both", "repeat-compare", "repeat-warm-sweep"),
        default="compare",
        help="Run the standard comparison, warm sweep, both, or repeated variants for stability checks.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS.keys()) + ("all",),
        default="all",
        help="Which scenario to run.",
    )
    parser.add_argument(
        "--include-optional-scenarios",
        action="store_true",
        help="When using --scenario all, also include optional scenarios such as Toy C.",
    )
    parser.add_argument("--nb", type=int, default=2500, help="Number of samples per method.")
    parser.add_argument("--gt-samples", type=int, default=160_000, help="Number of ground-truth posterior samples.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated runs for repeated modes.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for repeated modes.")
    parser.add_argument(
        "--warm-fractions",
        type=float,
        nargs="+",
        default=list(WARM_SWEEP_FRACTIONS),
        help="Warm fractions to evaluate for warm-sweep modes. 0.0 is omitted because plain P-DAPS is already reported in baselines.",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        scenario_keys = list(CORE_SCENARIO_KEYS)
        if args.include_optional_scenarios:
            scenario_keys.extend(OPTIONAL_SCENARIO_KEYS)
    else:
        scenario_keys = [args.scenario]

    for key in scenario_keys:
        if args.mode in ("compare", "both"):
            run_scenario(key, nb=args.nb, gt_samples_n=args.gt_samples)
        if args.mode in ("warm-sweep", "both"):
            run_warm_fraction_sweep(
                key,
                warm_fractions=tuple(args.warm_fractions),
                nb=args.nb,
                gt_samples_n=args.gt_samples,
            )
        if args.mode == "repeat-compare":
            run_repeated_compare(
                key, repeats=args.repeats, base_seed=args.base_seed, nb=args.nb, gt_samples_n=args.gt_samples,
            )
        if args.mode == "repeat-warm-sweep":
            run_repeated_warm_sweep(
                key, warm_fractions=tuple(args.warm_fractions), repeats=args.repeats,
                base_seed=args.base_seed, nb=args.nb, gt_samples_n=args.gt_samples,
            )


if __name__ == "__main__":
    main()
