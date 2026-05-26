import math

import numpy as np
import torch
import tqdm

from algo.base import Algo
from algo.pula import conjgrad, mri_A, mri_AH, mri_AHA
from utils.diffusion import DiffusionSampler
from utils.scheduler import Scheduler
from utilities import compute_ssim_nmse


class MRIInnerPULA:
    def __init__(
        self,
        num_steps,
        step_size=None,
        gamma=None,
        lr=None,
        cg_iter=10,
        lr_min_ratio=1.0,
        tau=None,
        check_finite=False,
        finite_check_interval=1,
        lam_floor=0.0,
        target_lam_floor=None,
        solve_lam_floor=None,
        noise_lam_floor=None,
        noise_tau=1.0,
        noise_mode="full",
        gamma_schedule="constant",
        gamma_floor=0.0,
        gamma_ceiling=float("inf"),
        precond_mode="standard",
        noise_rhs_mode="standard",
        penalty_scale=1.0,
        penalty_schedule="lambda",
        penalty_eps=0.0,
        mask_split_eps=None,
        mid_inner_project_every=0,
        tweedie_reanchor_every=0,
        reanchor_blend_beta=1.0,
    ):
        self.num_steps = int(num_steps)
        if step_size is None:
            step_size = gamma if gamma is not None else lr
        self.step_size = float(0.5 if step_size is None else step_size)
        self.cg_iter = int(cg_iter)
        self.lr_min_ratio = float(lr_min_ratio)
        self.check_finite = bool(check_finite)
        self.finite_check_interval = max(1, int(finite_check_interval))
        # lam_floor: clamp lam = 1/σ² from below. At high σ, the unfloored lam
        # makes M = AHA + λI nearly singular in nullspace directions, blowing
        # up the M⁻¹ noise. Setting lam_floor > 0 keeps the inner well-conditioned.
        self.lam_floor = float(lam_floor)
        self.target_lam_floor = float(lam_floor if target_lam_floor is None else target_lam_floor)
        self.solve_lam_floor = float(lam_floor if solve_lam_floor is None else solve_lam_floor)
        self.noise_lam_floor = float(lam_floor if noise_lam_floor is None else noise_lam_floor)
        # noise_tau: temperature on the Langevin noise. 1.0 = standard pULA;
        # 0.0 = drift-only (deterministic preconditioned gradient descent on
        # the data-fidelity + prior regularizer); intermediate = warm Langevin.
        self.noise_tau = float(noise_tau)
        self.tau = float(tau) if tau is not None else 1.0
        self.noise_mode = str(noise_mode)
        valid_noise_modes = {"full", "range_only", "image_only", "null_only", "none"}
        if self.noise_mode not in valid_noise_modes:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")
        self.gamma_schedule = str(gamma_schedule)
        valid_gamma_schedules = {
            "constant",
            "lambda",
            "lambda_cap",
            "sqrt_lambda",
            "sqrt_lambda_cap",
        }
        if self.gamma_schedule not in valid_gamma_schedules:
            raise ValueError(f"Unknown gamma_schedule: {self.gamma_schedule}")
        self.gamma_floor = float(gamma_floor)
        self.gamma_ceiling = float(gamma_ceiling)
        self.precond_mode = str(precond_mode)
        valid_precond_modes = {"standard", "laplacian", "laplacian_null", "mask_split"}
        if self.precond_mode not in valid_precond_modes:
            raise ValueError(f"Unknown precond_mode: {self.precond_mode}")
        self.noise_rhs_mode = str(noise_rhs_mode)
        valid_noise_rhs_modes = {"standard", "matched", "heuristic"}
        if self.noise_rhs_mode not in valid_noise_rhs_modes:
            raise ValueError(f"Unknown noise_rhs_mode: {self.noise_rhs_mode}")
        if self.precond_mode == "standard" and self.noise_rhs_mode == "heuristic":
            raise ValueError("noise_rhs_mode='heuristic' is only meaningful for nonstandard preconditioners")
        if self.precond_mode != "standard" and self.noise_rhs_mode == "standard":
            raise ValueError("noise_rhs_mode='standard' is only valid with precond_mode='standard'")
        if self.precond_mode == "mask_split" and self.noise_rhs_mode == "matched" and self.noise_mode == "image_only":
            raise ValueError(
                "noise_mode='image_only' is invalid with precond_mode='mask_split' "
                "and noise_rhs_mode='matched'"
            )
        if (
            self.precond_mode in {"laplacian", "laplacian_null"}
            and self.noise_rhs_mode == "matched"
            and self.noise_mode in {"image_only", "null_only"}
        ):
            raise ValueError(
                f"noise_mode='{self.noise_mode}' is invalid with precond_mode='{self.precond_mode}' "
                "and noise_rhs_mode='matched'"
            )
        self.penalty_scale = float(penalty_scale)
        self.penalty_schedule = str(penalty_schedule)
        valid_penalty_schedules = {"lambda", "constant"}
        if self.penalty_schedule not in valid_penalty_schedules:
            raise ValueError(f"Unknown penalty_schedule: {self.penalty_schedule}")
        self.penalty_eps = float(penalty_eps)
        self.mask_split_eps = None if mask_split_eps is None else float(mask_split_eps)
        if (
            self.penalty_scale < 0.0
            or self.penalty_eps < 0.0
            or (self.mask_split_eps is not None and self.mask_split_eps <= 0.0)
        ):
            raise ValueError("penalty_scale and penalty_eps must be nonnegative; mask_split_eps must be positive")
        self.mid_inner_project_every = int(mid_inner_project_every)
        self.tweedie_reanchor_every = int(tweedie_reanchor_every)
        self.reanchor_blend_beta = float(reanchor_blend_beta)
        if self.mid_inner_project_every < 0 or self.tweedie_reanchor_every < 0:
            raise ValueError("mid_inner_project_every and tweedie_reanchor_every must be nonnegative")
        if self.mid_inner_project_every > 0 and self.tweedie_reanchor_every > 0:
            raise ValueError("mid_inner_project_every and tweedie_reanchor_every are mutually exclusive")
        if not 0.0 <= self.reanchor_blend_beta <= 1.0:
            raise ValueError("reanchor_blend_beta must be in [0, 1]")

    def lambdas(self, sigma):
        lam_raw = 1.0 / float(sigma) ** 2
        lam_target_raw = lam_raw * (self.tau ** 2)
        return {
            "raw": lam_raw,
            "target": max(lam_target_raw, self.target_lam_floor),
            "solve": max(lam_raw, self.solve_lam_floor),
            "noise": max(lam_raw, self.noise_lam_floor),
        }

    def effective_gamma(self, sigma, ratio):
        lam_raw = 1.0 / float(sigma) ** 2
        gamma = self.step_size * (1.0 + ratio * (self.lr_min_ratio - 1.0))
        if self.gamma_schedule == "lambda":
            gamma *= lam_raw
        elif self.gamma_schedule == "lambda_cap":
            gamma *= min(1.0, lam_raw)
        elif self.gamma_schedule == "sqrt_lambda":
            gamma *= math.sqrt(lam_raw)
        elif self.gamma_schedule == "sqrt_lambda_cap":
            gamma *= min(1.0, math.sqrt(lam_raw))
        gamma = max(self.gamma_floor, gamma)
        gamma = min(self.gamma_ceiling, gamma)
        return gamma

    def effective_alpha(self, lam_raw):
        if self.precond_mode not in {"laplacian", "laplacian_null"}:
            return 0.0
        if self.penalty_schedule == "constant":
            return self.penalty_scale
        return self.penalty_scale * lam_raw

    def step(
        self,
        pdaps,
        x,
        x0hat,
        y,
        sigma,
        ratio,
        return_stats=False,
        return_full_stats=False,
        noise_scale=1.0,
    ):
        lams = self.lambdas(sigma)
        lam_raw = lams["raw"]
        lam_target = lams["target"]
        lam_solve = lams["solve"]
        lam_noise = lams["noise"]
        gamma = self.effective_gamma(sigma, ratio)
        alpha_eff = self.effective_alpha(lam_raw)
        want_iters = return_stats or return_full_stats

        def tensor_max(t):
            return t.abs().max().item() if isinstance(t, torch.Tensor) else 0.0

        score_lik = pdaps.AH(y - pdaps.A(x))
        score_prior = -(x - x0hat) * lam_target
        drift_rhs = score_lik + score_prior
        drift = pdaps.solve_precond(
            drift_rhs,
            lam_solve,
            precond_mode=self.precond_mode,
            alpha=alpha_eff,
            penalty_eps=self.penalty_eps,
            mask_split_eps=self.mask_split_eps,
            return_iters=want_iters,
        )
        if want_iters:
            drift, cg_drift = drift

        # torch complex randn has Var(real)=Var(imag)=1/2; scale by sqrt(2)
        # to match independent unit-variance real/imag Langevin noise.
        noise_rhs = torch.zeros_like(x)
        range_rhs = torch.zeros_like(x)
        null_rhs = torch.zeros_like(x)
        noise_scale = float(noise_scale)
        effective_noise_tau = self.noise_tau * noise_scale
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
                    raise ValueError(
                        f"noise_mode='{self.noise_mode}' is invalid with precond_mode='{self.precond_mode}' "
                        "and noise_rhs_mode='matched'"
                    )
                noise = pdaps.solve_precond(
                    noise_rhs,
                    lam_solve,
                    precond_mode=self.precond_mode,
                    alpha=alpha_eff,
                    penalty_eps=self.penalty_eps,
                    mask_split_eps=self.mask_split_eps,
                    return_iters=want_iters,
                )
            elif self.precond_mode == "mask_split" and self.noise_rhs_mode == "matched":
                n3 = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
                range_rhs = pdaps.project_range(ah_n1 + math.sqrt(lam_noise) * n2)
                # Matched null covariance is tied to the solve block. The public
                # mask_split_eps knob remains for non-matched mask-split modes.
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
                    raise ValueError(
                        f"noise_mode='{self.noise_mode}' is invalid with precond_mode='mask_split' "
                        "and noise_rhs_mode='matched'"
                    )
                noise = pdaps.solve_mask_split(
                    noise_rhs,
                    lam_solve,
                    null_rhs=null_rhs,
                    mask_split_eps=null_eps,
                    return_iters=want_iters,
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
                noise = pdaps.solve_precond(
                    noise_rhs,
                    lam_solve,
                    precond_mode=self.precond_mode,
                    alpha=alpha_eff,
                    penalty_eps=self.penalty_eps,
                    mask_split_eps=self.mask_split_eps,
                    return_iters=want_iters,
                )
            if want_iters:
                noise, cg_noise = noise
        else:
            # Drift-only ablation: skip the noise solve entirely.
            noise = torch.zeros_like(x)
            cg_noise = 0

        step_drift = 0.5 * gamma * drift
        step_noise = math.sqrt(gamma * effective_noise_tau) * noise
        step_total = step_drift + step_noise

        x_next = x + step_total
        if return_full_stats:
            x_null = pdaps.project_null(x_next)
            x_range = pdaps.project_range(x_next)
            x_null_norm = x_null.abs().square().mean().sqrt()
            x_range_norm = x_range.abs().square().mean().sqrt()
            stats = {
                "lam": lam_solve,
                "lam_raw": lam_raw,
                "lam_target": lam_target,
                "lam_solve": lam_solve,
                "lam_noise": lam_noise,
                "lam_floored": max(lam_target, lam_solve, lam_noise) > lam_raw + 1e-12,
                "lam_noise_floored": lam_noise > lam_raw + 1e-12,
                "noise_tau": self.noise_tau,
                "effective_noise_tau": effective_noise_tau,
                "noise_scale": noise_scale,
                "noise_mode": self.noise_mode,
                "precond_mode": self.precond_mode,
                "noise_rhs_mode": self.noise_rhs_mode,
                "penalty_scale": self.penalty_scale,
                "penalty_schedule": self.penalty_schedule,
                "penalty_eps": self.penalty_eps,
                "alpha_eff": alpha_eff,
                "gamma_schedule": self.gamma_schedule,
                "gamma_eff": gamma,
                "score_lik_max": score_lik.abs().max().item(),
                "score_lik_mean": score_lik.abs().mean().item(),
                "score_prior_max": score_prior.abs().max().item(),
                "score_prior_mean": score_prior.abs().mean().item(),
                "drift_rhs_max": drift_rhs.abs().max().item(),
                "noise_rhs_max": noise_rhs.abs().max().item(),
                "range_rhs_max": range_rhs.abs().max().item(),
                "null_rhs_max": null_rhs.abs().max().item(),
                "drift_solve_max": drift.abs().max().item(),
                "drift_solve_mean": drift.abs().mean().item(),
                "noise_solve_max": noise.abs().max().item(),
                "noise_solve_mean": noise.abs().mean().item(),
                "lap_drift_max": tensor_max(pdaps.laplacian(drift)) if self.precond_mode in {"laplacian", "laplacian_null"} else 0.0,
                "lap_noise_max": tensor_max(pdaps.laplacian(noise)) if self.precond_mode in {"laplacian", "laplacian_null"} else 0.0,
                "step_drift_max": step_drift.abs().max().item(),
                "step_drift_mean": step_drift.abs().mean().item(),
                "step_noise_max": step_noise.abs().max().item(),
                "step_noise_mean": step_noise.abs().mean().item(),
                "step_total_max": step_total.abs().max().item(),
                "step_total_over_sigma": step_total.abs().max().item() / max(float(sigma), 1e-12),
                "range_step_max": tensor_max(pdaps.project_range(step_noise)) if self.precond_mode == "mask_split" else 0.0,
                "null_step_max": tensor_max(pdaps.project_null(step_noise)) if self.precond_mode == "mask_split" else 0.0,
                "x_post_max": x_next.abs().max().item(),
                "x_post_mean": x_next.abs().mean().item(),
                "x_null_norm": x_null_norm.item(),
                "x_range_norm": x_range_norm.item(),
                "x_null_ratio": (x_null_norm / x_range_norm.clamp_min(pdaps.eps)).item(),
                "cg_drift": cg_drift,
                "cg_noise": cg_noise,
            }
            return x_next, stats
        if return_stats:
            return x_next, step_drift.abs().mean().item(), step_noise.abs().mean().item(), cg_drift + cg_noise
        return x_next

    def sample(
        self,
        pdaps,
        x,
        x0hat,
        y,
        sigma,
        ratio,
        return_stats=False,
        trace_log=None,
        target=None,
        trace_records=None,
        noise_gate_fn=None,
    ):
        """
        trace_log: optional dict with keys {"outer": int, "stride": int, "y": tensor}
        — when set, print per-inner-step diagnostics every `stride` steps
        (and on first/last step). residual is computed against y.
        """
        x = x.detach()
        x0hat = x0hat.detach()
        total_cg_iters = 0
        total_drift = 0.0
        total_noise = 0.0
        tracing = trace_log is not None
        stride = trace_log.get("stride", 0) if tracing else 0
        outer = trace_log.get("outer", -1) if tracing else -1
        for step in range(self.num_steps):
            trace_stats = {}
            noise_scale = 1.0
            noise_gate_resid = np.nan
            if noise_gate_fn is not None:
                gate_result = noise_gate_fn(x)
                if isinstance(gate_result, tuple):
                    gate_active, noise_gate_resid = gate_result
                else:
                    gate_active = gate_result
                noise_scale = 1.0 if gate_active else 0.0
            log_this = tracing and stride > 0 and (
                step == 0 or step == self.num_steps - 1 or (step + 1) % stride == 0
            )
            if log_this:
                x, full_stats = self.step(
                    pdaps, x, x0hat, y, sigma, ratio, return_full_stats=True, noise_scale=noise_scale
                )
                resid = pdaps.residual_norm(x, y).mean().item()
                floor_flag = " [λ floored]" if full_stats["lam_floored"] else ""
                noise_floor_flag = "*" if full_stats["lam_noise_floored"] else ""
                msg = (
                    f"[P-DAPS]   inner outer={outer:3d} k={step:3d}/{self.num_steps} "
                    f"σ={sigma:.4g} λ={full_stats['lam']:.3e}{floor_flag} γ={full_stats['gamma_eff']:.3e} "
                    f"λn={full_stats['lam_noise']:.3e}{noise_floor_flag} "
                    f"pre={full_stats['precond_mode']} noise_rhs={full_stats['noise_rhs_mode']} "
                    f"noise_scale={full_stats['noise_scale']:.1f} "
                    f"α={full_stats['alpha_eff']:.3e} μ={full_stats['penalty_scale']:.3g} "
                    f"pen_sched={full_stats['penalty_schedule']} eps={full_stats['penalty_eps']:.1e} "
                    f"lik.max={full_stats['score_lik_max']:.3e} prior.max={full_stats['score_prior_max']:.3e} "
                    f"rhs_n.max={full_stats['noise_rhs_max']:.3e} "
                    f"drift_M⁻¹.max={full_stats['drift_solve_max']:.3e} "
                    f"noise_M⁻¹.max={full_stats['noise_solve_max']:.3e} "
                    f"step_drift.max={full_stats['step_drift_max']:.3e} "
                    f"step_noise.max={full_stats['step_noise_max']:.3e} "
                    f"step_total.max={full_stats['step_total_max']:.3e} "
                    f"step/σ={full_stats['step_total_over_sigma']:.3e} "
                    f"range_step.max={full_stats['range_step_max']:.3e} "
                    f"null_step.max={full_stats['null_step_max']:.3e} "
                    f"x.max={full_stats['x_post_max']:.3e} "
                    f"x_null/range={full_stats['x_null_ratio']:.3e} resid={resid:.3e} "
                    f"CG_d={full_stats['cg_drift']} CG_n={full_stats['cg_noise']}"
                )
                print(msg, flush=True)
                total_drift += full_stats['step_drift_mean']
                total_noise += full_stats['step_noise_mean']
                total_cg_iters += full_stats['cg_drift'] + full_stats['cg_noise']
                trace_stats = {
                    "gamma_eff": float(full_stats["gamma_eff"]),
                    "step_total_max": float(full_stats["step_total_max"]),
                    "step_total_over_sigma": float(full_stats["step_total_over_sigma"]),
                    "step_drift_max": float(full_stats["step_drift_max"]),
                    "step_noise_max": float(full_stats["step_noise_max"]),
                    "noise_scale": float(full_stats["noise_scale"]),
                    "noise_gate_resid": float(noise_gate_resid),
                    "lam_noise": float(full_stats["lam_noise"]),
                    "lam_noise_floored": float(full_stats["lam_noise_floored"]),
                }
            elif return_stats:
                x, d_norm, n_norm, cg_iters = self.step(
                    pdaps, x, x0hat, y, sigma, ratio, return_stats=True, noise_scale=noise_scale
                )
                total_drift += d_norm
                total_noise += n_norm
                total_cg_iters += cg_iters
            else:
                x = self.step(pdaps, x, x0hat, y, sigma, ratio, noise_scale=noise_scale)
            if trace_records is not None:
                x_range = pdaps.project_range(x)
                x_null = pdaps.project_null(x)
                range_energy = x_range.abs().square().flatten(start_dim=1).sum(dim=1).mean().item()
                null_energy = x_null.abs().square().flatten(start_dim=1).sum(dim=1).mean().item()
                data_residual = pdaps.A(x) - y
                data_misfit_inner = (
                    data_residual.abs().square().flatten(start_dim=1).sum(dim=1)
                    / y.abs().square().flatten(start_dim=1).sum(dim=1).clamp_min(pdaps.eps)
                ).mean().item()
                record = {
                    "outer": int(outer),
                    "inner": int(step),
                    "sigma": float(sigma),
                    "ratio": float(ratio),
                    "inner_active": 1.0,
                    "range_energy": range_energy,
                    "null_energy": null_energy,
                    "null_range_ratio": null_energy / max(range_energy, pdaps.eps),
                    "data_misfit_inner": data_misfit_inner,
                    "x_abs_max": x.abs().max().item(),
                    "noise_scale": float(noise_scale),
                    "noise_gate_resid": float(noise_gate_resid),
                }
                record.update(trace_stats)
                if target is not None:
                    ssim_inner, nmse_inner = compute_ssim_nmse(pdaps.to_real(x), pdaps.to_real(target))
                    record["ssim_inner"] = ssim_inner
                    record["nmse_inner"] = nmse_inner
                trace_records.append(record)
            should_check = self.check_finite and (
                step == self.num_steps - 1 or (step + 1) % self.finite_check_interval == 0
            )
            if should_check and not torch.isfinite(torch.view_as_real(x)).all():
                if not return_stats:
                    return torch.zeros_like(x)
                return torch.zeros_like(x), 0.0, 0.0, 0.0
            if step != self.num_steps - 1:
                if self.mid_inner_project_every > 0 and (step + 1) % self.mid_inner_project_every == 0:
                    x_before = x
                    x_real = pdaps.to_real(x)
                    x_real = pdaps.net(x_real, torch.as_tensor(sigma, device=x_real.device))
                    x = pdaps.to_complex(x_real).detach()
                    if tracing:
                        shift = (x - x_before).abs().mean().item()
                        print(
                            f"[P-DAPS]   mid_inner_project outer={outer:3d} "
                            f"k={step:3d}/{self.num_steps} shift={shift:.3e}",
                            flush=True,
                        )
                elif self.tweedie_reanchor_every > 0 and (step + 1) % self.tweedie_reanchor_every == 0:
                    x_real = pdaps.to_real(x)
                    x_real = pdaps.net(x_real, torch.as_tensor(sigma, device=x_real.device))
                    x0hat_new = pdaps.to_complex(x_real).detach()
                    beta = self.reanchor_blend_beta
                    anchor_shift = (x0hat_new - x0hat).abs().mean().item()
                    x0hat = ((1.0 - beta) * x0hat + beta * x0hat_new).detach()
                    if tracing:
                        print(
                            f"[P-DAPS]   reanchor outer={outer:3d} "
                            f"k={step:3d}/{self.num_steps} beta={beta:.3g} "
                            f"anchor_shift={anchor_shift:.3e}",
                            flush=True,
                        )
        if not return_stats:
            return x.detach()
        return x.detach(), total_cg_iters / max(1, self.num_steps), total_drift / max(1, self.num_steps), total_noise / max(1, self.num_steps)


class PDAPS(Algo):
    def __init__(
        self,
        net,
        forward_op,
        annealing_scheduler_config={},
        diffusion_scheduler_config={},
        lgvd_config={},
        warm_mode="none",
        warm_fraction=0.0,
        inner_sigma_max=float("inf"),
        eps=1e-8,
        log_level="INFO",
        edm_project_post=False,
        warm_init_strategy="previous",
        inner_gate_mode="sigma",
        residual_threshold=0.3,
        noise_gate_mode="none",
        noise_residual_threshold=None,
        noise_sigma_min=None,
        noise_residual_min=None,
        sigma_stop_truncate=None,
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.annealing = Scheduler(**annealing_scheduler_config)
        self.diffusion_config = diffusion_scheduler_config
        self.inner = MRIInnerPULA(**lgvd_config)
        self.warm_mode = warm_mode
        self.warm_fraction = float(warm_fraction)
        self.inner_sigma_max = float(inner_sigma_max)
        self.eps = float(eps)
        self.last_gate_stats = []
        self.log_level = log_level
        self.warm_init_strategy = str(warm_init_strategy)
        if self.warm_init_strategy not in {"previous", "cgsense", "zero_filled"}:
            raise ValueError(f"Unknown warm_init_strategy: {self.warm_init_strategy}")
        self.inner_gate_mode = str(inner_gate_mode)
        if self.inner_gate_mode not in {"sigma", "residual", "compound"}:
            raise ValueError(f"Unknown inner_gate_mode: {self.inner_gate_mode}")
        self.residual_threshold = float(residual_threshold)
        self.noise_gate_mode = str(noise_gate_mode)
        valid_noise_gate_modes = {
            "none",
            "sigma",
            "residual",
            "compound",
            "sigma_early",
            "residual_early",
            "compound_early",
        }
        if self.noise_gate_mode not in valid_noise_gate_modes:
            raise ValueError(f"Unknown noise_gate_mode: {self.noise_gate_mode}")
        if noise_residual_threshold is None:
            self.noise_residual_threshold = (
                self.residual_threshold
                if self.noise_gate_mode in {"residual", "compound"}
                else None
            )
        else:
            self.noise_residual_threshold = float(noise_residual_threshold)
        self.noise_sigma_min = None if noise_sigma_min is None else float(noise_sigma_min)
        self.noise_residual_min = None if noise_residual_min is None else float(noise_residual_min)
        self.sigma_stop_truncate = None if sigma_stop_truncate is None else float(sigma_stop_truncate)
        if self.noise_gate_mode in {"sigma_early", "compound_early"} and self.noise_sigma_min is None:
            raise ValueError(f"noise_sigma_min is required for noise_gate_mode='{self.noise_gate_mode}'")
        if self.noise_gate_mode in {"residual_early", "compound_early"} and self.noise_residual_min is None:
            raise ValueError(f"noise_residual_min is required for noise_gate_mode='{self.noise_gate_mode}'")
        self._warm_init_cache = None
        self._inner_has_run = False
        # edm_project_post: after the inner Langevin produces x_clean, run
        # one Tweedie pass through the EDM denoiser at the *current* outer σ
        # before re-noising. Pulls x_clean back onto the natural-image manifold
        # so that nullspace excursions don't poison the next outer iteration.
        self.edm_project_post = bool(edm_project_post)

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
        return mri_A(self, x)

    def AH(self, y):
        return mri_AH(self, y)

    def AHA(self, x):
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
        mask = self.fourier_mask(x)
        return self.forward_op.ifft(mask * self.forward_op.fft(x))

    def project_null(self, x):
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
        # AHA is zero on forward-null directions. lam_solve I damps every
        # direction, including DC where the Laplacian has eigenvalue zero.
        diag = lam_solve + penalty_eps
        if diag > 0.0:
            normal_op = lambda v: self.AHA(v) + diag * v
        else:
            normal_op = self.AHA
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
        null_sol = null_rhs / null_scale
        sol = self.project_range(range_sol) + null_sol
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
        m = max(1.0, float(self.forward_op.mask.expand_as(y[:1]).sum().item()))
        return r.abs().square().flatten(start_dim=1).sum(dim=1).sqrt() / math.sqrt(m)

    def nullspace_energy(self, x):
        """Per-sample RMS of (I - A⁺A) x — the part of x A cannot constrain.
        Uses (I - AHA·M⁻¹) x with λ→0 limit approximated via a single CG-projected solve.
        Cheaper proxy: AH(A x) - x scale. Here we report ||AHAx - x|| / ||x|| as an
        unnormalized indicator of how far x sits from the range of AH (operator norm dependent)."""
        Mx = self.AHA(x)
        diff = Mx - x
        denom = x.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
        num = diff.abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        return num / denom

    def adaptive_alpha(self, x_prev, x0hat, y):
        drift = (x_prev - x0hat).abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
        drift = drift / x0hat.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)

        r_hat = self.residual_norm(x0hat, y)
        r_prev = self.residual_norm(x_prev, y)
        alpha = self.warm_fraction * r_hat / (r_hat + r_prev + self.eps) / (1.0 + drift)
        alpha = alpha.clamp(0.0, self.warm_fraction)

        self.last_gate_stats.append({
            "alpha_mean": float(alpha.mean().detach().cpu()),
            "alpha_min": float(alpha.min().detach().cpu()),
            "alpha_max": float(alpha.max().detach().cpu()),
            "drift_mean": float(drift.mean().detach().cpu()),
            "r_hat_mean": float(r_hat.mean().detach().cpu()),
            "r_prev_mean": float(r_prev.mean().detach().cpu()),
        })
        return alpha.view(-1, 1, 1)

    def cgsense_init(self, y, ref):
        if self._warm_init_cache is None:
            self._warm_init_cache = conjgrad(
                self.AHA,
                self.AH(y),
                torch.zeros_like(ref),
                0.0,
                20,
                x_is_zero=True,
            ).detach()
        return self._warm_init_cache

    def init_inner(self, x_prev, x0hat, y):
        if self.warm_mode == "none" or self.warm_fraction <= 0.0:
            return x0hat
        warm_source = x_prev
        if warm_source is None:
            if self.warm_init_strategy == "cgsense":
                warm_source = self.cgsense_init(y, x0hat)
            elif self.warm_init_strategy == "zero_filled":
                warm_source = self.AH(y)
            else:
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
        self._warm_init_cache = None
        self._inner_has_run = False
        self.last_gate_stats = []
        trace_records = [] if trace_path is not None else None
        N = self.annealing.num_steps
        if self.log_level in ["INFO", "DEBUG"]:
            print(f"[P-DAPS] init: xt.abs.max={xt.abs().max().item():.3e}  y.abs.max={y.abs().max().item():.3e}  warm_mode={self.warm_mode}  warm_fraction={self.warm_fraction}  inner_sigma_max={self.inner_sigma_max}  warm_init_strategy={self.warm_init_strategy}  inner_gate_mode={self.inner_gate_mode}  noise_gate_mode={self.noise_gate_mode}  noise_sigma_min={self.noise_sigma_min}  noise_residual_min={self.noise_residual_min}  sigma_stop_truncate={self.sigma_stop_truncate}")
        
        disable_tqdm = (self.log_level in ["VAL", "WARN"] or not verbose)
        steps = tqdm.trange(N, desc="P-DAPS", disable=disable_tqdm)
        # At DEBUG: log a handful of inner steps per outer step so the trace
        # is readable. Stride = ceil(num_steps/4) → ~4 samples + first + last.
        inner_log_stride = max(1, self.inner.num_steps // 4) if self.log_level == "DEBUG" else 0

        for i in steps:
            sigma = self.annealing.sigma_steps[i]
            if self.sigma_stop_truncate is not None and float(sigma) < self.sigma_stop_truncate:
                if self.log_level in ["INFO", "DEBUG"]:
                    print(
                        f"[P-DAPS] sigma_stop_truncate={self.sigma_stop_truncate:g} "
                        f"reached at outer={i} sigma={float(sigma):g}; breaking outer loop",
                        flush=True,
                    )
                break
            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))

            # Pre-inner residual (what the inner correction starts from).
            resid_pre = self.residual_norm(x0hat, y).mean().item()

            inner_active = self.inner_gate_active(float(sigma), resid_pre)
            noise_gate_outer_active = self.noise_gate_active(float(sigma), resid_pre)
            self.last_gate_stats.append({
                "kind": "gate",
                "outer": int(i),
                "sigma": float(sigma),
                "resid_pre": float(resid_pre),
                "inner_active": bool(inner_active),
                "inner_gate_mode": self.inner_gate_mode,
                "residual_threshold": float(self.residual_threshold),
                "noise_gate_active": bool(noise_gate_outer_active),
                "noise_gate_mode": self.noise_gate_mode,
                "noise_residual_threshold": self.noise_residual_threshold,
                "noise_sigma_min": self.noise_sigma_min,
                "noise_residual_min": self.noise_residual_min,
            })
            if not inner_active:
                x_clean = x0hat
                inner_stats = ""
                if trace_records is not None:
                    null_idx = self.nullspace_energy(x_clean).mean().item()
                    record = {
                        "outer": int(i),
                        "inner": -1.0,
                        "sigma": float(sigma),
                        "ratio": float(i / max(1, N)),
                        "inner_active": 0.0,
                        "resid_pre": float(resid_pre),
                        "resid": float(resid_pre),
                        "null_idx": float(null_idx),
                        "inner_null_growth": 1.0,
                        "total_growth": 1.0,
                        "meas_growth": 1.0,
                        "inner_dist": 0.0,
                    }
                    if target_complex is not None:
                        ssim_inner, nmse_inner = compute_ssim_nmse(self.to_real(x_clean), self.to_real(target_complex))
                        record["ssim_inner"] = ssim_inner
                        record["nmse_inner"] = nmse_inner
                    trace_records.append(record)
                if self.log_level == "DEBUG":
                    print(
                        f"[P-DAPS]   inner gate skip outer={i:3d} mode={self.inner_gate_mode} "
                        f"σ={sigma:.4f} resid_pre={resid_pre:.3e} "
                        f"threshold={self.residual_threshold:.3e}",
                        flush=True,
                    )
            else:
                warm_prev = x_prev
                if (
                    self.warm_init_strategy in {"cgsense", "zero_filled"}
                    and not self._inner_has_run
                ):
                    warm_prev = None
                    if self.log_level == "DEBUG":
                        print(
                            f"[P-DAPS]   {self.warm_init_strategy} warm init at first "
                            f"active inner outer={i:3d}",
                            flush=True,
                        )
                x_init = self.init_inner(warm_prev, x0hat, y)
                if self.log_level == "DEBUG":
                    def noise_gate_fn(x_cur):
                        if self.noise_gate_mode == "none":
                            return True, np.nan
                        resid_cur = self.residual_norm(x_cur, y).mean().item()
                        return self.noise_gate_active(float(sigma), resid_cur), resid_cur

                    trace_log = {"outer": i, "stride": inner_log_stride, "y": y} if inner_log_stride > 0 else None
                    x_clean, avg_cg, avg_drift, avg_noise = self.inner.sample(
                        self, x_init, x0hat, y, sigma, i / max(1, N),
                        return_stats=True, trace_log=trace_log,
                        target=target_complex, trace_records=trace_records,
                        noise_gate_fn=noise_gate_fn,
                    )
                    inner_dist = (x_clean - x0hat).abs().mean().item()
                    inner_stats = f" inner_dist={inner_dist:.3e} CG={avg_cg:.1f} drift={avg_drift:.3e} noise={avg_noise:.3e}"
                else:
                    def noise_gate_fn(x_cur):
                        if self.noise_gate_mode == "none":
                            return True, np.nan
                        resid_cur = self.residual_norm(x_cur, y).mean().item()
                        return self.noise_gate_active(float(sigma), resid_cur), resid_cur

                    x_clean = self.inner.sample(
                        self, x_init, x0hat, y, sigma, i / max(1, N),
                        target=target_complex, trace_records=trace_records,
                        noise_gate_fn=noise_gate_fn,
                    )
                    inner_stats = ""
                self._inner_has_run = True

            if self.log_level == "DEBUG" or i % 20 == 0 or i < 5 or i >= N - 3:
                msg = (f"[P-DAPS] outer={i:3d} σ={sigma:.4f} "
                       f"x0hat.max={x0hat.abs().max().item():.3e} "
                       f"x_clean.max={x_clean.abs().max().item():.3e}")

                if self.log_level == "DEBUG":
                    resid = self.residual_norm(x_clean, y).mean().item()
                    lams = self.inner.lambdas(sigma)
                    lam_raw = lams["raw"]
                    lam = lams["solve"]
                    gamma_eff = self.inner.effective_gamma(sigma, i / max(1, N))
                    # Range/null growth: how much did x grow in the measured
                    # subspace (||A·x||) vs in total (||x||)? Big total/meas ratio
                    # ⇒ inner Langevin is amplifying nullspace components.
                    norm_x = x_clean.abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
                    norm_x0 = x0hat.abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
                    norm_Ax = self.A(x_clean).abs().square().flatten(start_dim=1).sum(dim=1).sqrt()
                    norm_Ax0 = self.A(x0hat).abs().square().flatten(start_dim=1).sum(dim=1).sqrt().clamp_min(self.eps)
                    total_growth = (norm_x / norm_x0).mean().item()
                    meas_growth = (norm_Ax / norm_Ax0).mean().item()
                    null_post = self.nullspace_energy(x_clean).mean().item()
                    null_init = self.project_null(x0hat).norm()
                    null_clean = self.project_null(x_clean).norm()
                    inner_null_growth = (null_clean / null_init.clamp_min(self.eps)).item()
                    if trace_records is not None and inner_active:
                        trace_records.append({
                            "outer": int(i),
                            "inner": -1.0,
                            "sigma": float(sigma),
                            "ratio": float(i / max(1, N)),
                            "inner_active": float(inner_active),
                            "resid_pre": float(resid_pre),
                            "resid": float(resid),
                            "null_idx": float(null_post),
                            "inner_null_growth": float(inner_null_growth),
                            "total_growth": float(total_growth),
                            "meas_growth": float(meas_growth),
                            "inner_dist": float(inner_dist if inner_active else 0.0),
                            "x_abs_max": float(x_clean.abs().max().item()),
                            "gamma_eff": float(gamma_eff),
                        })
                    floor_flag = " [λ floored]" if lam > lam_raw + 1e-12 else ""
                    msg += (
                        f" resid_pre={resid_pre:.3e} resid={resid:.3e} "
                        f"λ={lam:.3e}{floor_flag} γ={gamma_eff:.3e} "
                        f"null_idx={null_post:.3e} inner_null_growth={inner_null_growth:.3f} "
                        f"grow_tot={total_growth:.3f} grow_meas={meas_growth:.3f}"
                        f"{inner_stats}"
                    )

                if self.log_level in ["INFO", "DEBUG"]:
                    if verbose:
                        tqdm.tqdm.write(msg)
                    else:
                        print(msg)
            # Optional EDM Tweedie projection: pull x_clean back onto the
            # natural-image manifold by passing it through the denoiser at
            # the current outer σ. Cheap (one forward pass), and washes out
            # nullspace blowup that would otherwise propagate through re-noise.
            if self.edm_project_post and sigma > self.inner_sigma_max:
                # Skip the projection in the σ-band where the inner already
                # ran — there x_clean is meant to be a refined Langevin sample
                # we don't want to over-smooth.
                pass
            elif self.edm_project_post:
                x_real = self.to_real(x_clean)
                x_real = self.net(x_real, torch.as_tensor(sigma, device=x_real.device))
                x_clean = self.to_complex(x_real)
                if self.log_level == "DEBUG":
                    print(f"[P-DAPS]   edm_project: post-proj x.max={x_clean.abs().max().item():.3e}", flush=True)

            x_prev = x_clean
            xt = self.to_real(
                x_clean + math.sqrt(2.0) * torch.randn_like(x_clean) * self.annealing.sigma_steps[i + 1]
            )

        if trace_path is not None:
            self.write_trace_npz(trace_path, trace_records)
        return xt.float()


class PDAPSWarm(PDAPS):
    def __init__(self, *args, warm_mode="fixed", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)


class PDAPSAdaptive(PDAPS):
    def __init__(self, *args, warm_mode="adaptive", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)
