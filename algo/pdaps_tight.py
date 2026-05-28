import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import tqdm

from algo.base import Algo
from algo.pula import conjgrad, mri_A, mri_AH, mri_AHA
from utils.diffusion import DiffusionSampler
from utils.scheduler import Scheduler
from utilities import compute_ssim_nmse


DAPS_TAU = 0.002028752174814177


@dataclass
class PDAPSTightRecipe:
    """
    Per-domain frozen choices.

    These are not deployment-time validation knobs. They move when the forward
    operator, diffusion model, or dataset family changes.
    """

    tau: float = DAPS_TAU
    inner_sigma_max: float = 5.0
    solve_lam_floor: float = 3.0
    noise_lam_floor: float = 3.0
    noise_tau: float = 0.0
    noise_mode: str = "range_only"
    precond_mode: str = "standard"
    noise_rhs_mode: str = "standard"
    noise_gate_mode: str = "none"
    noise_residual_threshold: float | None = None
    noise_sigma_min: float | None = None
    noise_residual_min: float | None = None
    warm_mode: str = "fixed"
    warm_init: str = "previous"
    inner_gate_mode: str = "sigma"
    residual_threshold: float = 0.3
    cg_iter: int = 10
    penalty_scale: float = 1.0
    penalty_schedule: str = "lambda"
    penalty_eps: float = 0.0
    mask_split_eps: float | None = None
    reanchor_every: int = 0
    reanchor_blend_beta: float = 1.0
    eps: float = 1e-8
    annealing_scheduler_config: dict = field(
        default_factory=lambda: {
            "sigma_max": 100.0,
            "sigma_min": 0.1,
            "sigma_final": 0.0,
            "num_steps": 200,
            "schedule": "linear",
            "timestep": "poly-7",
        }
    )
    diffusion_scheduler_config: dict = field(
        default_factory=lambda: {
            "sigma_min": 0.01,
            "sigma_final": 0.0,
            "num_steps": 5,
            "schedule": "linear",
            "timestep": "poly-7",
        }
    )


def _recipe_from_config(recipe):
    if recipe is None:
        return PDAPSTightRecipe()
    if isinstance(recipe, PDAPSTightRecipe):
        return recipe
    if isinstance(recipe, dict):
        return PDAPSTightRecipe(**recipe)
    raise TypeError(f"recipe must be None, dict, or PDAPSTightRecipe; got {type(recipe)!r}")


class TightInnerPULA:
    """
    Inner deterministic/stochastic preconditioned Langevin block.

    The validation knobs are deliberately absent here. This class receives a
    per-domain recipe and runs the recipe's operator/noise policy.
    """

    VALID_NOISE_MODES = {"full", "range_only", "image_only", "null_only", "none"}
    VALID_PRECOND_MODES = {"standard", "laplacian", "laplacian_null", "mask_split"}
    VALID_NOISE_RHS_MODES = {"standard", "matched", "heuristic"}
    VALID_PENALTY_SCHEDULES = {"lambda", "constant"}

    def __init__(self, num_steps, gamma, target_lam_floor, recipe: PDAPSTightRecipe):
        self.num_steps = int(num_steps)
        self.step_size = float(gamma)
        self.target_lam_floor = float(target_lam_floor)
        self.tau = float(recipe.tau)
        self.solve_lam_floor = float(recipe.solve_lam_floor)
        self.noise_lam_floor = float(recipe.noise_lam_floor)
        self.noise_tau = float(recipe.noise_tau)
        self.noise_mode = str(recipe.noise_mode)
        self.precond_mode = str(recipe.precond_mode)
        self.noise_rhs_mode = str(recipe.noise_rhs_mode)
        self.cg_iter = int(recipe.cg_iter)
        self.penalty_scale = float(recipe.penalty_scale)
        self.penalty_schedule = str(recipe.penalty_schedule)
        self.penalty_eps = float(recipe.penalty_eps)
        self.mask_split_eps = None if recipe.mask_split_eps is None else float(recipe.mask_split_eps)
        self.reanchor_every = int(recipe.reanchor_every)
        self.reanchor_blend_beta = float(recipe.reanchor_blend_beta)

        if self.noise_mode not in self.VALID_NOISE_MODES:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")
        if self.precond_mode not in self.VALID_PRECOND_MODES:
            raise ValueError(f"Unknown precond_mode: {self.precond_mode}")
        if self.noise_rhs_mode not in self.VALID_NOISE_RHS_MODES:
            raise ValueError(f"Unknown noise_rhs_mode: {self.noise_rhs_mode}")
        if self.penalty_schedule not in self.VALID_PENALTY_SCHEDULES:
            raise ValueError(f"Unknown penalty_schedule: {self.penalty_schedule}")
        if self.precond_mode == "standard" and self.noise_rhs_mode == "heuristic":
            raise ValueError("noise_rhs_mode='heuristic' is only meaningful for nonstandard preconditioners")
        if self.precond_mode != "standard" and self.noise_rhs_mode == "standard":
            raise ValueError("noise_rhs_mode='standard' is only valid with precond_mode='standard'")
        if self.precond_mode == "mask_split" and self.noise_rhs_mode == "matched" and self.noise_mode == "image_only":
            raise ValueError("noise_mode='image_only' is invalid with matched mask_split noise")
        if (
            self.precond_mode in {"laplacian", "laplacian_null"}
            and self.noise_rhs_mode == "matched"
            and self.noise_mode in {"image_only", "null_only"}
        ):
            raise ValueError(f"noise_mode='{self.noise_mode}' is invalid with matched laplacian noise")
        if self.penalty_scale < 0.0 or self.penalty_eps < 0.0:
            raise ValueError("penalty_scale and penalty_eps must be nonnegative")
        if self.mask_split_eps is not None and self.mask_split_eps <= 0.0:
            raise ValueError("mask_split_eps must be positive")
        if self.reanchor_every < 0:
            raise ValueError("reanchor_every must be nonnegative")
        if not 0.0 <= self.reanchor_blend_beta <= 1.0:
            raise ValueError("reanchor_blend_beta must be in [0, 1]")

    def lambdas(self, sigma):
        lam_raw = 1.0 / float(sigma) ** 2
        return {
            "raw": lam_raw,
            "target": max(lam_raw * (self.tau ** 2), self.target_lam_floor),
            "solve": max(lam_raw, self.solve_lam_floor),
            "noise": max(lam_raw, self.noise_lam_floor),
        }

    def effective_gamma(self, sigma):
        # Tight core: gamma_schedule="lambda" with no floor/ceiling clamp.
        return self.step_size / float(sigma) ** 2

    def effective_alpha(self, lam_raw):
        if self.precond_mode not in {"laplacian", "laplacian_null"}:
            return 0.0
        if self.penalty_schedule == "constant":
            return self.penalty_scale
        return self.penalty_scale * lam_raw

    def step(self, pdaps, x, x0hat, y, sigma, noise_scale=1.0, return_stats=False):
        lams = self.lambdas(sigma)
        lam_raw = lams["raw"]
        lam_target = lams["target"]
        lam_solve = lams["solve"]
        lam_noise = lams["noise"]
        gamma = self.effective_gamma(sigma)
        alpha_eff = self.effective_alpha(lam_raw)

        score_lik = pdaps.AH(y - pdaps.A(x))
        score_prior = -(x - x0hat) * lam_target
        drift_rhs = score_lik + score_prior
        drift_result = pdaps.solve_precond(
            drift_rhs,
            lam_solve,
            precond_mode=self.precond_mode,
            alpha=alpha_eff,
            penalty_eps=self.penalty_eps,
            mask_split_eps=self.mask_split_eps,
            return_iters=return_stats,
        )
        if return_stats:
            drift, cg_drift = drift_result
        else:
            drift = drift_result
            cg_drift = 0

        noise_rhs = torch.zeros_like(x)
        range_rhs = torch.zeros_like(x)
        null_rhs = torch.zeros_like(x)
        effective_noise_tau = self.noise_tau * float(noise_scale)
        if effective_noise_tau > 0.0 and self.noise_mode != "none":
            n1 = math.sqrt(2.0) * torch.randn_like(y)
            n2 = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
            ah_n1 = pdaps.AH(n1)

            if self.precond_mode in {"laplacian", "laplacian_null"} and self.noise_rhs_mode == "matched":
                ng_x = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
                ng_y = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
                penalty_rhs = pdaps.divH(ng_x, ng_y)
                if self.precond_mode == "laplacian_null":
                    penalty_rhs = pdaps.project_null(penalty_rhs)
                penalty_rhs = math.sqrt(max(alpha_eff, 0.0)) * penalty_rhs
                if self.penalty_eps > 0.0:
                    penalty_rhs = penalty_rhs + math.sqrt(self.penalty_eps) * n2
                if self.noise_mode == "range_only":
                    noise_rhs = ah_n1
                elif self.noise_mode == "full":
                    noise_rhs = ah_n1 + penalty_rhs
                else:
                    raise ValueError(f"noise_mode='{self.noise_mode}' is invalid with matched laplacian noise")
                noise_result = pdaps.solve_precond(
                    noise_rhs,
                    lam_solve,
                    precond_mode=self.precond_mode,
                    alpha=alpha_eff,
                    penalty_eps=self.penalty_eps,
                    mask_split_eps=self.mask_split_eps,
                    return_iters=return_stats,
                )
            elif self.precond_mode == "mask_split" and self.noise_rhs_mode == "matched":
                n3 = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
                range_rhs = pdaps.project_range(ah_n1 + math.sqrt(lam_noise) * n2)
                null_eps = lam_solve
                null_rhs = math.sqrt(null_eps) * pdaps.project_null(n3)
                if self.noise_mode == "range_only":
                    noise_rhs = range_rhs
                    null_rhs = torch.zeros_like(x)
                elif self.noise_mode == "null_only":
                    noise_rhs = torch.zeros_like(x)
                elif self.noise_mode == "full":
                    noise_rhs = range_rhs
                else:
                    raise ValueError(f"noise_mode='{self.noise_mode}' is invalid with matched mask_split noise")
                noise_result = pdaps.solve_mask_split(
                    noise_rhs,
                    lam_solve,
                    null_rhs=null_rhs,
                    mask_split_eps=null_eps,
                    return_iters=return_stats,
                )
            else:
                if self.noise_mode == "range_only":
                    noise_rhs = ah_n1
                elif self.noise_mode == "image_only":
                    noise_rhs = math.sqrt(lam_noise) * n2
                elif self.noise_mode == "null_only":
                    noise_rhs = math.sqrt(lam_noise) * pdaps.project_null(n2)
                elif self.noise_mode == "full":
                    noise_rhs = ah_n1 + math.sqrt(lam_noise) * n2
                else:
                    raise ValueError(f"Unknown noise_mode: {self.noise_mode}")
                if self.precond_mode == "mask_split":
                    range_rhs = pdaps.project_range(noise_rhs)
                    null_rhs = pdaps.project_null(noise_rhs)
                noise_result = pdaps.solve_precond(
                    noise_rhs,
                    lam_solve,
                    precond_mode=self.precond_mode,
                    alpha=alpha_eff,
                    penalty_eps=self.penalty_eps,
                    mask_split_eps=self.mask_split_eps,
                    return_iters=return_stats,
                )
            if return_stats:
                noise, cg_noise = noise_result
            else:
                noise = noise_result
                cg_noise = 0
        else:
            noise = torch.zeros_like(x)
            cg_noise = 0

        step_drift = 0.5 * gamma * drift
        step_noise = math.sqrt(gamma * effective_noise_tau) * noise
        x_next = x + step_drift + step_noise
        if not return_stats:
            return x_next

        stats = {
            "gamma_eff": float(gamma),
            "lam_raw": float(lam_raw),
            "lam_target": float(lam_target),
            "lam_solve": float(lam_solve),
            "lam_noise": float(lam_noise),
            "noise_tau": float(self.noise_tau),
            "effective_noise_tau": float(effective_noise_tau),
            "noise_scale": float(noise_scale),
            "noise_mode": self.noise_mode,
            "precond_mode": self.precond_mode,
            "noise_rhs_mode": self.noise_rhs_mode,
            "alpha_eff": float(alpha_eff),
            "step_drift_mean": float(step_drift.abs().mean().item()),
            "step_noise_mean": float(step_noise.abs().mean().item()),
            "step_total_max": float((step_drift + step_noise).abs().max().item()),
            "noise_rhs_max": float(noise_rhs.abs().max().item()),
            "range_rhs_max": float(range_rhs.abs().max().item()),
            "null_rhs_max": float(null_rhs.abs().max().item()),
            "cg_iters": int(cg_drift + cg_noise),
        }
        return x_next, stats

    def sample(
        self,
        pdaps,
        x,
        x0hat,
        y,
        sigma,
        noise_gate_fn=None,
        return_stats=False,
        trace_records=None,
        target=None,
        outer=-1,
    ):
        x = x.detach()
        x0hat = x0hat.detach()
        total_cg = 0
        total_drift = 0.0
        total_noise = 0.0
        last_stats = {}

        for step in range(self.num_steps):
            noise_scale = 1.0
            noise_gate_resid = np.nan
            if noise_gate_fn is not None:
                gate_result = noise_gate_fn(x)
                if isinstance(gate_result, tuple):
                    gate_active, noise_gate_resid = gate_result
                else:
                    gate_active = gate_result
                noise_scale = 1.0 if gate_active else 0.0

            if return_stats or trace_records is not None:
                x, last_stats = self.step(pdaps, x, x0hat, y, sigma, noise_scale=noise_scale, return_stats=True)
                total_cg += last_stats["cg_iters"]
                total_drift += last_stats["step_drift_mean"]
                total_noise += last_stats["step_noise_mean"]
            else:
                x = self.step(pdaps, x, x0hat, y, sigma, noise_scale=noise_scale)

            if trace_records is not None:
                self._append_trace_record(
                    pdaps,
                    trace_records,
                    x,
                    y,
                    sigma,
                    outer,
                    step,
                    noise_scale,
                    noise_gate_resid,
                    last_stats,
                    target,
                )

            if (
                self.reanchor_every > 0
                and step != self.num_steps - 1
                and (step + 1) % self.reanchor_every == 0
            ):
                x_real = pdaps.to_real(x)
                x_real = pdaps.net(x_real, torch.as_tensor(sigma, device=x_real.device))
                x0hat_new = pdaps.to_complex(x_real).detach()
                beta = self.reanchor_blend_beta
                x0hat = ((1.0 - beta) * x0hat + beta * x0hat_new).detach()

        x = x.detach()
        if not return_stats:
            return x
        denom = max(1, self.num_steps)
        return x, total_cg / denom, total_drift / denom, total_noise / denom

    @staticmethod
    def _append_trace_record(
        pdaps,
        records,
        x,
        y,
        sigma,
        outer,
        inner,
        noise_scale,
        noise_gate_resid,
        stats,
        target,
    ):
        x_range = pdaps.project_range(x)
        x_null = pdaps.project_null(x)
        range_energy = x_range.abs().square().flatten(start_dim=1).sum(dim=1).mean().item()
        null_energy = x_null.abs().square().flatten(start_dim=1).sum(dim=1).mean().item()
        data_residual = pdaps.A(x) - y
        y_energy = y.abs().square().flatten(start_dim=1).sum(dim=1).clamp_min(pdaps.eps)
        data_misfit_inner = (
            data_residual.abs().square().flatten(start_dim=1).sum(dim=1) / y_energy
        ).mean().item()
        record = {
            "outer": int(outer),
            "inner": int(inner),
            "sigma": float(sigma),
            "inner_active": 1.0,
            "range_energy": float(range_energy),
            "null_energy": float(null_energy),
            "null_range_ratio": float(null_energy / max(range_energy, pdaps.eps)),
            "data_misfit_inner": float(data_misfit_inner),
            "x_abs_max": float(x.abs().max().item()),
            "noise_scale": float(noise_scale),
            "noise_gate_resid": float(noise_gate_resid),
        }
        for key, value in stats.items():
            if isinstance(value, (int, float, bool, np.number)):
                record[key] = float(value)
        if target is not None:
            ssim_inner, nmse_inner = compute_ssim_nmse(pdaps.to_real(x), pdaps.to_real(target))
            record["ssim_inner"] = ssim_inner
            record["nmse_inner"] = nmse_inner
        records.append(record)


class PDAPSTight(Algo):
    """
    Tight P-DAPS mainline.

    Public validation knobs:
      - lgvd_num_steps
      - sigma_stop_truncate
      - warm_fraction
      - gamma
      - target_lam_floor

    Per-domain choices are supplied through `recipe`; historical branch-routing
    subclasses and MRI-specific warm-start caches are intentionally absent.
    """

    VALID_WARM_MODES = {"fixed", "adaptive"}
    VALID_WARM_INIT = {"previous", "hook", "adjoint"}
    VALID_GATE_MODES = {"sigma", "residual", "compound"}
    VALID_NOISE_GATE_MODES = {
        "none",
        "sigma",
        "residual",
        "compound",
        "sigma_early",
        "residual_early",
        "compound_early",
    }

    def __init__(
        self,
        net,
        forward_op,
        recipe=None,
        lgvd_num_steps=50,
        sigma_stop_truncate=None,
        warm_fraction=0.8,
        gamma=0.5,
        target_lam_floor=1e-3,
        log_level="INFO",
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.recipe = _recipe_from_config(recipe)
        self.annealing = Scheduler(**self.recipe.annealing_scheduler_config)
        self.diffusion_config = dict(self.recipe.diffusion_scheduler_config)
        self.inner = TightInnerPULA(
            num_steps=lgvd_num_steps,
            gamma=gamma,
            target_lam_floor=target_lam_floor,
            recipe=self.recipe,
        )
        self.sigma_stop_truncate = None if sigma_stop_truncate is None else float(sigma_stop_truncate)
        self.warm_fraction = float(warm_fraction)
        self.inner_sigma_max = float(self.recipe.inner_sigma_max)
        self.warm_mode = str(self.recipe.warm_mode)
        self.warm_init = str(self.recipe.warm_init)
        self.inner_gate_mode = str(self.recipe.inner_gate_mode)
        self.residual_threshold = float(self.recipe.residual_threshold)
        self.noise_gate_mode = str(self.recipe.noise_gate_mode)
        self.noise_residual_threshold = (
            self.residual_threshold
            if self.recipe.noise_residual_threshold is None
            and self.noise_gate_mode in {"residual", "compound"}
            else self.recipe.noise_residual_threshold
        )
        self.noise_sigma_min = self.recipe.noise_sigma_min
        self.noise_residual_min = self.recipe.noise_residual_min
        self.eps = float(self.recipe.eps)
        self.log_level = str(log_level)
        self.last_gate_stats = []

        if self.warm_mode not in self.VALID_WARM_MODES:
            raise ValueError(f"Unknown warm_mode: {self.warm_mode}")
        if self.warm_init not in self.VALID_WARM_INIT:
            raise ValueError(f"Unknown warm_init: {self.warm_init}")
        if self.inner_gate_mode not in self.VALID_GATE_MODES:
            raise ValueError(f"Unknown inner_gate_mode: {self.inner_gate_mode}")
        if self.noise_gate_mode not in self.VALID_NOISE_GATE_MODES:
            raise ValueError(f"Unknown noise_gate_mode: {self.noise_gate_mode}")
        if self.noise_gate_mode in {"sigma_early", "compound_early"} and self.noise_sigma_min is None:
            raise ValueError(f"noise_sigma_min is required for noise_gate_mode='{self.noise_gate_mode}'")
        if self.noise_gate_mode in {"residual_early", "compound_early"} and self.noise_residual_min is None:
            raise ValueError(f"noise_residual_min is required for noise_gate_mode='{self.noise_gate_mode}'")

    @staticmethod
    def to_complex(x):
        return torch.complex(x[:, 0], x[:, 1])

    @staticmethod
    def to_real(x):
        return torch.stack([x.real, x.imag], dim=1)

    @staticmethod
    def grad2d(x):
        gx = torch.roll(x, shifts=-1, dims=-1) - x
        gy = torch.roll(x, shifts=-1, dims=-2) - x
        return gx, gy

    @staticmethod
    def divH(gx, gy):
        dx = torch.roll(gx, shifts=1, dims=-1) - gx
        dy = torch.roll(gy, shifts=1, dims=-2) - gy
        return dx + dy

    @staticmethod
    def laplacian(x):
        return (
            4 * x
            - torch.roll(x, shifts=1, dims=-1)
            - torch.roll(x, shifts=-1, dims=-1)
            - torch.roll(x, shifts=1, dims=-2)
            - torch.roll(x, shifts=-1, dims=-2)
        )

    def A(self, x):
        if hasattr(self.forward_op, "A"):
            return self.forward_op.A(x)
        return mri_A(self, x)

    def AH(self, y):
        if hasattr(self.forward_op, "AH"):
            return self.forward_op.AH(y)
        return mri_AH(self, y)

    def AHA(self, x):
        if hasattr(self.forward_op, "AHA"):
            return self.forward_op.AHA(x)
        return mri_AHA(self, x)

    def fourier_mask(self, x):
        mask = self.forward_op.mask.to(device=x.device, dtype=x.real.dtype)
        h, w = x.shape[-2], x.shape[-1]
        if mask.shape[-2] == 1 and mask.shape[-1] == w:
            return mask.reshape(1, 1, w)
        if mask.shape[-2] == h and mask.shape[-1] == 1:
            return mask.reshape(1, h, 1)
        if mask.shape[-2] == h and mask.shape[-1] == w:
            return mask.reshape(1, h, w)
        raise ValueError(f"Unsupported MRI mask shape {tuple(mask.shape)} for image shape {tuple(x.shape)}")

    def project_range(self, x):
        if hasattr(self.forward_op, "project_range"):
            return self.forward_op.project_range(x)
        mask = self.fourier_mask(x)
        return self.forward_op.ifft(mask * self.forward_op.fft(x))

    def project_null(self, x):
        if hasattr(self.forward_op, "project_null"):
            return self.forward_op.project_null(x)
        return x - self.project_range(x)

    def solve(self, rhs, lam, return_iters=False, normal_op=None, penalty_op=None):
        normal = self.AHA if normal_op is None else normal_op
        return conjgrad(
            normal,
            rhs,
            torch.zeros_like(rhs),
            lam,
            self.inner.cg_iter,
            x_is_zero=True,
            return_iters=return_iters,
            penalty_op=penalty_op,
        )

    def solve_laplacian(
        self,
        rhs,
        alpha,
        lam_solve=0.0,
        penalty_eps=0.0,
        null_only_penalty=False,
        return_iters=False,
    ):
        diag = lam_solve + penalty_eps
        normal_op = (lambda v: self.AHA(v) + diag * v) if diag > 0.0 else self.AHA
        penalty_op = self.laplacian
        if null_only_penalty:
            penalty_op = lambda v: self.laplacian(self.project_null(v))
        return self.solve(
            rhs,
            alpha,
            return_iters=return_iters,
            normal_op=normal_op,
            penalty_op=penalty_op,
        )

    def solve_mask_split(self, rhs, lam, null_rhs=None, mask_split_eps=None, return_iters=False):
        range_rhs = self.project_range(rhs)
        if null_rhs is None:
            null_rhs = self.project_null(rhs)
        else:
            null_rhs = self.project_null(null_rhs)

        def range_normal(v):
            return self.project_range(self.AHA(self.project_range(v)))

        range_sol = conjgrad(
            range_normal,
            range_rhs,
            torch.zeros_like(rhs),
            lam,
            self.inner.cg_iter,
            x_is_zero=True,
            return_iters=return_iters,
        )
        if return_iters:
            range_sol, cg_iters = range_sol
        null_scale = lam if mask_split_eps is None else float(mask_split_eps)
        sol = self.project_range(range_sol) + null_rhs / null_scale
        if return_iters:
            return sol, cg_iters
        return sol

    def solve_precond(
        self,
        rhs,
        lam,
        precond_mode="standard",
        alpha=0.0,
        penalty_eps=0.0,
        mask_split_eps=None,
        return_iters=False,
    ):
        if precond_mode == "standard":
            return self.solve(rhs, lam, return_iters=return_iters)
        if precond_mode == "laplacian":
            return self.solve_laplacian(
                rhs,
                alpha,
                lam_solve=lam,
                penalty_eps=penalty_eps,
                return_iters=return_iters,
            )
        if precond_mode == "laplacian_null":
            return self.solve_laplacian(
                rhs,
                alpha,
                lam_solve=lam,
                penalty_eps=penalty_eps,
                null_only_penalty=True,
                return_iters=return_iters,
            )
        if precond_mode == "mask_split":
            return self.solve_mask_split(
                rhs,
                lam,
                mask_split_eps=mask_split_eps,
                return_iters=return_iters,
            )
        raise ValueError(f"Unknown precond_mode: {precond_mode}")

    def residual_norm(self, x, y):
        r = self.A(x) - y
        if hasattr(self.forward_op, "num_measurements"):
            m = max(1.0, float(self.forward_op.num_measurements(y)))
        elif hasattr(self.forward_op, "mask"):
            m = max(1.0, float(self.forward_op.mask.expand_as(y[:1]).sum().item()))
        else:
            m = max(1.0, float(y[0].numel()))
        return r.abs().square().flatten(start_dim=1).sum(dim=1).sqrt() / math.sqrt(m)

    def nullspace_energy(self, x):
        diff = self.AHA(x) - x
        denom = x.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
        num = diff.abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        return num / denom

    def adaptive_alpha(self, x_prev, x0hat, y):
        drift = (x_prev - x0hat).abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        drift = drift / x0hat.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
        r_hat = self.residual_norm(x0hat, y)
        r_prev = self.residual_norm(x_prev, y)
        alpha = self.warm_fraction * r_hat / (r_hat + r_prev + self.eps) / (1.0 + drift)
        return alpha.clamp(0.0, self.warm_fraction).view(-1, 1, 1)

    def warm_init_source(self, x_prev, x0hat, y):
        if x_prev is not None:
            return x_prev
        if self.warm_init == "previous":
            return None
        if self.warm_init == "hook":
            hook = getattr(self.forward_op, "warm_init", None)
            if hook is None:
                return None
            candidate = hook(y, ref=x0hat)
            if torch.is_complex(candidate):
                return candidate
            return self.to_complex(candidate)
        if self.warm_init == "adjoint":
            return self.AH(y)
        raise ValueError(f"Unknown warm_init: {self.warm_init}")

    def init_inner(self, x_prev, x0hat, y):
        if self.warm_fraction <= 0.0:
            return x0hat
        warm_source = self.warm_init_source(x_prev, x0hat, y)
        if warm_source is None:
            return x0hat
        if self.warm_mode == "fixed":
            alpha = self.warm_fraction
        elif self.warm_mode == "adaptive":
            alpha = self.adaptive_alpha(warm_source, x0hat, y)
        else:
            raise ValueError(f"Unknown warm_mode: {self.warm_mode}")
        return alpha * warm_source + (1.0 - alpha) * x0hat

    def late_gate_active(self, mode, sigma, resid, residual_threshold):
        if mode == "sigma":
            return sigma <= self.inner_sigma_max
        if mode == "residual":
            return resid <= residual_threshold
        return sigma <= self.inner_sigma_max and resid <= residual_threshold

    def inner_gate_active(self, sigma, resid_pre):
        return self.late_gate_active(self.inner_gate_mode, sigma, resid_pre, self.residual_threshold)

    def noise_gate_active(self, sigma, resid):
        if self.noise_gate_mode == "none":
            return True
        if self.noise_gate_mode in {"sigma", "residual", "compound"}:
            return self.late_gate_active(
                self.noise_gate_mode,
                sigma,
                resid,
                self.noise_residual_threshold,
            )
        if self.noise_gate_mode == "sigma_early":
            return sigma >= self.noise_sigma_min
        if self.noise_gate_mode == "residual_early":
            return resid >= self.noise_residual_min
        return sigma >= self.noise_sigma_min and resid >= self.noise_residual_min

    @staticmethod
    def write_trace_npz(trace_path, records):
        if not records:
            return
        numeric_keys = sorted({key for row in records for key in row})
        arrays = {}
        for key in numeric_keys:
            vals = []
            numeric = True
            for row in records:
                val = row.get(key, np.nan)
                if isinstance(val, (int, float, bool, np.number)):
                    vals.append(float(val))
                else:
                    numeric = False
                    break
            if numeric:
                arrays[key] = np.asarray(vals, dtype=np.float64)
        np.savez_compressed(trace_path, **arrays)

    def _append_outer_trace(self, records, i, sigma, inner_active, x_clean, x0hat, y, resid_pre, target):
        resid = self.residual_norm(x_clean, y).mean().item()
        null_post = self.nullspace_energy(x_clean).mean().item()
        null_init = self.project_null(x0hat).norm()
        null_clean = self.project_null(x_clean).norm()
        norm_x = x_clean.abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        norm_x0 = x0hat.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
        norm_Ax = self.A(x_clean).abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        norm_Ax0 = self.A(x0hat).abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
        record = {
            "outer": int(i),
            "inner": -1.0,
            "sigma": float(sigma),
            "inner_active": float(inner_active),
            "resid_pre": float(resid_pre),
            "resid": float(resid),
            "null_idx": float(null_post),
            "inner_null_growth": float((null_clean / null_init.clamp_min(self.eps)).item()),
            "total_growth": float((norm_x / norm_x0).mean().item()),
            "meas_growth": float((norm_Ax / norm_Ax0).mean().item()),
            "inner_dist": float((x_clean - x0hat).abs().mean().item()) if inner_active else 0.0,
            "x_abs_max": float(x_clean.abs().max().item()),
        }
        if target is not None:
            ssim_inner, nmse_inner = compute_ssim_nmse(self.to_real(x_clean), self.to_real(target))
            record["ssim_inner"] = ssim_inner
            record["nmse_inner"] = nmse_inner
        records.append(record)

    def recipe_dict(self):
        return asdict(self.recipe)

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=True, target=None, trace_path=None):
        device = self.forward_op.device
        y = torch.view_as_complex(observation).to(device)
        if num_samples > 1:
            y = y.expand(num_samples, -1, -1, -1)

        target_complex = None
        if target is not None:
            target = target.to(device)
            if target.ndim == 3:
                target = target.unsqueeze(0)
            target_complex = self.to_complex(target)

        xt = torch.randn(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution,
            self.net.img_resolution,
            device=device,
        ) * self.annealing.sigma_max

        x_prev = None
        self.last_gate_stats = []
        trace_records = [] if trace_path is not None else None
        N = self.annealing.num_steps
        disable_tqdm = self.log_level in {"VAL", "WARN"} or not verbose

        if self.log_level in {"INFO", "DEBUG"}:
            print(
                "[P-DAPS-tight] "
                f"warm_mode={self.warm_mode} warm_fraction={self.warm_fraction:g} "
                f"warm_init={self.warm_init} inner_sigma_max={self.inner_sigma_max:g} "
                f"noise_tau={self.inner.noise_tau:g} noise_mode={self.inner.noise_mode} "
                f"precond_mode={self.inner.precond_mode} sigma_stop_truncate={self.sigma_stop_truncate}",
                flush=True,
            )

        for i in tqdm.trange(N, desc="P-DAPS-tight", disable=disable_tqdm):
            sigma = self.annealing.sigma_steps[i]
            sigma_f = float(sigma)
            if self.sigma_stop_truncate is not None and sigma_f < self.sigma_stop_truncate:
                if self.log_level in {"INFO", "DEBUG"}:
                    print(
                        f"[P-DAPS-tight] sigma_stop_truncate={self.sigma_stop_truncate:g} "
                        f"reached at outer={i} sigma={sigma_f:g}; returning last clean state",
                        flush=True,
                    )
                if x_prev is not None:
                    xt = self.to_real(x_prev)
                break

            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))
            resid_pre = self.residual_norm(x0hat, y).mean().item()
            inner_active = self.inner_gate_active(sigma_f, resid_pre)
            noise_gate_outer_active = self.noise_gate_active(sigma_f, resid_pre)
            self.last_gate_stats.append({
                "outer": int(i),
                "sigma": sigma_f,
                "resid_pre": float(resid_pre),
                "inner_active": bool(inner_active),
                "noise_gate_active": bool(noise_gate_outer_active),
            })

            if inner_active:
                x_init = self.init_inner(x_prev, x0hat, y)

                def noise_gate_fn(x_cur):
                    if self.noise_gate_mode == "none":
                        return True, np.nan
                    resid_cur = self.residual_norm(x_cur, y).mean().item()
                    return self.noise_gate_active(sigma_f, resid_cur), resid_cur

                x_clean, avg_cg, avg_drift, avg_noise = self.inner.sample(
                    self,
                    x_init,
                    x0hat,
                    y,
                    sigma,
                    noise_gate_fn=noise_gate_fn,
                    return_stats=True,
                    trace_records=trace_records,
                    target=target_complex,
                    outer=i,
                )
                inner_stats = f" CG={avg_cg:.1f} drift={avg_drift:.3e} noise={avg_noise:.3e}"
            else:
                x_clean = x0hat
                inner_stats = ""

            if trace_records is not None:
                self._append_outer_trace(
                    trace_records,
                    i,
                    sigma,
                    inner_active,
                    x_clean,
                    x0hat,
                    y,
                    resid_pre,
                    target_complex,
                )

            if self.log_level in {"INFO", "DEBUG"} and (i < 5 or i % 20 == 0 or i >= N - 3):
                msg = (
                    f"[P-DAPS-tight] outer={i:3d} sigma={sigma_f:.4f} "
                    f"inner={int(inner_active)} resid_pre={resid_pre:.3e} "
                    f"x0hat.max={x0hat.abs().max().item():.3e} "
                    f"x_clean.max={x_clean.abs().max().item():.3e}{inner_stats}"
                )
                if verbose:
                    tqdm.tqdm.write(msg)
                else:
                    print(msg)

            x_prev = x_clean
            xt = self.to_real(
                x_clean + math.sqrt(2.0) * torch.randn_like(x_clean) * self.annealing.sigma_steps[i + 1]
            )

        if trace_path is not None:
            self.write_trace_npz(trace_path, trace_records)
        return xt.float()
