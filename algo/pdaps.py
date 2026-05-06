import math

import torch
import tqdm

from algo.base import Algo
from algo.pula import conjgrad, mri_A, mri_AH, mri_AHA
from utils.diffusion import DiffusionSampler
from utils.scheduler import Scheduler


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
        noise_tau=1.0,
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
        # noise_tau: temperature on the Langevin noise. 1.0 = standard pULA;
        # 0.0 = drift-only (deterministic preconditioned gradient descent on
        # the data-fidelity + prior regularizer); intermediate = warm Langevin.
        self.noise_tau = float(noise_tau)

    def step(self, pdaps, x, x0hat, y, sigma, ratio, return_stats=False, return_full_stats=False):
        lam_raw = 1.0 / float(sigma) ** 2
        lam = max(lam_raw, self.lam_floor)
        gamma = self.step_size * (1.0 + ratio * (self.lr_min_ratio - 1.0))

        score_lik = pdaps.AH(y - pdaps.A(x))
        score_prior = -(x - x0hat) * lam
        drift = pdaps.solve(score_lik + score_prior, lam, return_iters=return_stats or return_full_stats)
        if return_stats or return_full_stats:
            drift, cg_drift = drift

        # torch complex randn has Var(real)=Var(imag)=1/2; scale by sqrt(2)
        # to match independent unit-variance real/imag Langevin noise.
        if self.noise_tau > 0.0:
            n1 = math.sqrt(2.0) * torch.randn_like(y)
            n2 = math.sqrt(2.0) * torch.randn(x.shape, dtype=x.dtype, device=x.device)
            noise = pdaps.solve(pdaps.AH(n1) + math.sqrt(lam) * n2, lam, return_iters=return_stats or return_full_stats)
            if return_stats or return_full_stats:
                noise, cg_noise = noise
        else:
            # Drift-only ablation: skip the noise solve entirely.
            noise = torch.zeros_like(x)
            cg_noise = 0

        step_drift = 0.5 * gamma * drift
        step_noise = math.sqrt(gamma * self.noise_tau) * noise

        x_next = x + step_drift + step_noise
        if return_full_stats:
            stats = {
                "lam": lam,
                "lam_raw": lam_raw,
                "lam_floored": lam > lam_raw + 1e-12,
                "noise_tau": self.noise_tau,
                "gamma_eff": gamma,
                "score_lik_max": score_lik.abs().max().item(),
                "score_lik_mean": score_lik.abs().mean().item(),
                "score_prior_max": score_prior.abs().max().item(),
                "score_prior_mean": score_prior.abs().mean().item(),
                "drift_solve_max": drift.abs().max().item(),
                "drift_solve_mean": drift.abs().mean().item(),
                "noise_solve_max": noise.abs().max().item(),
                "noise_solve_mean": noise.abs().mean().item(),
                "step_drift_max": step_drift.abs().max().item(),
                "step_drift_mean": step_drift.abs().mean().item(),
                "step_noise_max": step_noise.abs().max().item(),
                "step_noise_mean": step_noise.abs().mean().item(),
                "x_post_max": x_next.abs().max().item(),
                "x_post_mean": x_next.abs().mean().item(),
                "cg_drift": cg_drift,
                "cg_noise": cg_noise,
            }
            return x_next, stats
        if return_stats:
            return x_next, step_drift.abs().mean().item(), step_noise.abs().mean().item(), cg_drift + cg_noise
        return x_next

    def sample(self, pdaps, x, x0hat, y, sigma, ratio, return_stats=False, trace_log=None):
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
            log_this = tracing and stride > 0 and (
                step == 0 or step == self.num_steps - 1 or (step + 1) % stride == 0
            )
            if log_this:
                x, full_stats = self.step(
                    pdaps, x, x0hat, y, sigma, ratio, return_full_stats=True
                )
                resid = pdaps.residual_norm(x, y).mean().item()
                msg = (
                    f"[P-DAPS]   inner outer={outer:3d} k={step:3d}/{self.num_steps} "
                    f"σ={sigma:.4g} λ={full_stats['lam']:.3e} γ={full_stats['gamma_eff']:.3e} "
                    f"lik.max={full_stats['score_lik_max']:.3e} prior.max={full_stats['score_prior_max']:.3e} "
                    f"drift_M⁻¹.max={full_stats['drift_solve_max']:.3e} "
                    f"noise_M⁻¹.max={full_stats['noise_solve_max']:.3e} "
                    f"step_drift.max={full_stats['step_drift_max']:.3e} "
                    f"step_noise.max={full_stats['step_noise_max']:.3e} "
                    f"x.max={full_stats['x_post_max']:.3e} resid={resid:.3e} "
                    f"CG_d={full_stats['cg_drift']} CG_n={full_stats['cg_noise']}"
                )
                print(msg, flush=True)
                total_drift += full_stats['step_drift_mean']
                total_noise += full_stats['step_noise_mean']
                total_cg_iters += full_stats['cg_drift'] + full_stats['cg_noise']
            elif return_stats:
                x, d_norm, n_norm, cg_iters = self.step(pdaps, x, x0hat, y, sigma, ratio, return_stats=True)
                total_drift += d_norm
                total_noise += n_norm
                total_cg_iters += cg_iters
            else:
                x = self.step(pdaps, x, x0hat, y, sigma, ratio)
            should_check = self.check_finite and (
                step == self.num_steps - 1 or (step + 1) % self.finite_check_interval == 0
            )
            if should_check and not torch.isfinite(torch.view_as_real(x)).all():
                if not return_stats:
                    return torch.zeros_like(x)
                return torch.zeros_like(x), 0.0, 0.0, 0.0
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

    def A(self, x):
        return mri_A(self, x)

    def AH(self, y):
        return mri_AH(self, y)

    def AHA(self, x):
        return mri_AHA(self, x)

    def solve(self, rhs, lam, return_iters=False):
        return conjgrad(
            self.AHA,
            rhs,
            torch.zeros_like(rhs),
            lam,
            self.inner.cg_iter,
            x_is_zero=True,
            return_iters=return_iters,
        )

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

    def init_inner(self, x_prev, x0hat, y):
        if x_prev is None or self.warm_mode == "none" or self.warm_fraction <= 0.0:
            return x0hat
        if self.warm_mode == "fixed":
            alpha = self.warm_fraction
        elif self.warm_mode == "adaptive":
            alpha = self.adaptive_alpha(x_prev, x0hat, y)
        else:
            raise ValueError(f"Unknown warm_mode: {self.warm_mode}")
        return alpha * x_prev + (1.0 - alpha) * x0hat

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        y = torch.view_as_complex(observation).to(device)
        if num_samples > 1:
            y = y.expand(num_samples, -1, -1, -1)

        xt = torch.randn(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution,
            self.net.img_resolution,
            device=device,
        ) * self.annealing.sigma_max

        x_prev = None
        self.last_gate_stats = []
        N = self.annealing.num_steps
        if self.log_level in ["INFO", "DEBUG"]:
            print(f"[P-DAPS] init: xt.abs.max={xt.abs().max().item():.3e}  y.abs.max={y.abs().max().item():.3e}  warm_mode={self.warm_mode}  warm_fraction={self.warm_fraction}  inner_sigma_max={self.inner_sigma_max}")
        
        disable_tqdm = (self.log_level in ["VAL", "WARN"] or not verbose)
        steps = tqdm.trange(N, desc="P-DAPS", disable=disable_tqdm)
        # At DEBUG: log a handful of inner steps per outer step so the trace
        # is readable. Stride = ceil(num_steps/4) → ~4 samples + first + last.
        inner_log_stride = max(1, self.inner.num_steps // 4) if self.log_level == "DEBUG" else 0

        for i in steps:
            sigma = self.annealing.sigma_steps[i]
            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))

            # Pre-inner residual (what the inner correction starts from).
            resid_pre = self.residual_norm(x0hat, y).mean().item() if self.log_level == "DEBUG" else None

            if sigma > self.inner_sigma_max:
                x_clean = x0hat
                inner_stats = ""
            else:
                x_init = self.init_inner(x_prev, x0hat, y)
                if self.log_level == "DEBUG":
                    trace_log = {"outer": i, "stride": inner_log_stride, "y": y} if inner_log_stride > 0 else None
                    x_clean, avg_cg, avg_drift, avg_noise = self.inner.sample(
                        self, x_init, x0hat, y, sigma, i / max(1, N),
                        return_stats=True, trace_log=trace_log,
                    )
                    inner_dist = (x_clean - x0hat).abs().mean().item()
                    inner_stats = f" inner_dist={inner_dist:.3e} CG={avg_cg:.1f} drift={avg_drift:.3e} noise={avg_noise:.3e}"
                else:
                    x_clean = self.inner.sample(self, x_init, x0hat, y, sigma, i / max(1, N))
                    inner_stats = ""

            if self.log_level == "DEBUG" or i % 20 == 0 or i < 5 or i >= N - 3:
                msg = (f"[P-DAPS] outer={i:3d} σ={sigma:.4f} "
                       f"x0hat.max={x0hat.abs().max().item():.3e} "
                       f"x_clean.max={x_clean.abs().max().item():.3e}")

                if self.log_level == "DEBUG":
                    resid = self.residual_norm(x_clean, y).mean().item()
                    lam_raw = 1.0 / float(sigma) ** 2
                    lam = max(lam_raw, self.inner.lam_floor)
                    gamma_eff = self.inner.step_size * (
                        1.0 + (i / max(1, N)) * (self.inner.lr_min_ratio - 1.0)
                    )
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
                    floor_flag = " [λ floored]" if lam > lam_raw + 1e-12 else ""
                    msg += (
                        f" resid_pre={resid_pre:.3e} resid={resid:.3e} "
                        f"λ={lam:.3e}{floor_flag} γ={gamma_eff:.3e} "
                        f"null_idx={null_post:.3e} grow_tot={total_growth:.3f} grow_meas={meas_growth:.3f}"
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

        return xt.float()


class PDAPSWarm(PDAPS):
    def __init__(self, *args, warm_mode="fixed", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)


class PDAPSAdaptive(PDAPS):
    def __init__(self, *args, warm_mode="adaptive", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)
