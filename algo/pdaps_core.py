import math

import numpy as np
import torch
import tqdm

from algo.base import Algo
from algo.pula import conjgrad, mri_A, mri_AH, mri_AHA
from utils.diffusion import DiffusionSampler
from utils.scheduler import Scheduler
from utilities import compute_ssim_nmse


DAPS_TAU = 0.002028752174814177


class PDAPS(Algo):
    """
    Slim production P-DAPS core.

    This is the zero-temperature, standard-preconditioned inner correction
    selected by the pre-launch ablations. Historical branch controls remain in
    algo.pdaps.PDAPS for reproduction.
    """

    def __init__(
        self,
        net,
        forward_op,
        annealing_scheduler_config=None,
        diffusion_scheduler_config=None,
        lgvd_config=None,
        warm_fraction=0.8,
        inner_sigma_max=5.0,
        sigma_stop_truncate=None,
        eps=1e-8,
        log_level="INFO",
    ):
        super().__init__(net, forward_op)
        lgvd_config = {} if lgvd_config is None else dict(lgvd_config)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.annealing = Scheduler(**(annealing_scheduler_config or {}))
        self.diffusion_config = diffusion_scheduler_config or {}

        self.num_steps = int(lgvd_config.get("num_steps", 50))
        self.gamma = float(lgvd_config.get("gamma", lgvd_config.get("lr", 0.5)))
        self.tau = float(lgvd_config.get("tau", DAPS_TAU))
        self.solve_lam_floor = float(lgvd_config.get("solve_lam_floor", 3.0))
        self.lr_min_ratio = float(lgvd_config.get("lr_min_ratio", 0.01))
        self.cg_iter = int(lgvd_config.get("cg_iter", 10))

        self.warm_fraction = float(warm_fraction)
        self.inner_sigma_max = float(inner_sigma_max)
        self.sigma_stop_truncate = None if sigma_stop_truncate is None else float(sigma_stop_truncate)
        self.eps = float(eps)
        self.log_level = str(log_level)
        self.last_gate_stats = []

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
            self.cg_iter,
            x_is_zero=True,
            return_iters=return_iters,
        )

    def residual_norm(self, x, y):
        r = self.A(x) - y
        m = max(1.0, float(self.forward_op.mask.expand_as(y[:1]).sum().item()))
        return r.abs().square().flatten(start_dim=1).sum(dim=1).sqrt() / math.sqrt(m)

    def lambdas(self, sigma):
        lam_raw = 1.0 / float(sigma) ** 2
        return {
            "raw": lam_raw,
            "target": lam_raw * (self.tau ** 2),
            "solve": max(lam_raw, self.solve_lam_floor),
        }

    def effective_gamma(self, sigma, ratio):
        lam_raw = 1.0 / float(sigma) ** 2
        return self.gamma * (1.0 + float(ratio) * (self.lr_min_ratio - 1.0)) * lam_raw

    def _inner_step(self, x, x0hat, y, sigma, ratio, return_stats=False):
        lams = self.lambdas(sigma)
        gamma_eff = self.effective_gamma(sigma, ratio)
        score_lik = self.AH(y - self.A(x))
        score_prior = -(x - x0hat) * lams["target"]
        result = self.solve(score_lik + score_prior, lams["solve"], return_iters=return_stats)
        if return_stats:
            drift, cg_iters = result
        else:
            drift = result
            cg_iters = 0
        step = 0.5 * gamma_eff * drift
        x_next = x + step
        if not return_stats:
            return x_next
        return x_next, {
            "cg_iters": int(cg_iters),
            "gamma_eff": float(gamma_eff),
            "lam_raw": float(lams["raw"]),
            "lam_target": float(lams["target"]),
            "lam_solve": float(lams["solve"]),
            "step_drift_mean": float(step.abs().mean().item()),
            "step_total_max": float(step.abs().max().item()),
        }

    def _inner_sample(self, x, x0hat, y, sigma, ratio, return_stats=False):
        total_cg = 0
        total_drift = 0.0
        for _ in range(self.num_steps):
            if return_stats:
                x, stats = self._inner_step(x, x0hat, y, sigma, ratio, return_stats=True)
                total_cg += stats["cg_iters"]
                total_drift += stats["step_drift_mean"]
            else:
                x = self._inner_step(x, x0hat, y, sigma, ratio)
        x = x.detach()
        if not return_stats:
            return x
        denom = max(1, self.num_steps)
        return x, total_cg / denom, total_drift / denom

    def _init_inner(self, x_prev, x0hat):
        if x_prev is None or self.warm_fraction <= 0.0:
            return x0hat
        return self.warm_fraction * x_prev + (1.0 - self.warm_fraction) * x0hat

    @staticmethod
    def write_trace_npz(trace_path, records):
        if not records:
            return
        keys = sorted({key for row in records for key in row})
        arrays = {}
        for key in keys:
            vals = []
            for row in records:
                val = row.get(key, np.nan)
                if not isinstance(val, (int, float, bool, np.number)):
                    vals = None
                    break
                vals.append(float(val))
            if vals is not None:
                arrays[key] = np.asarray(vals, dtype=np.float64)
        np.savez_compressed(trace_path, **arrays)

    def _append_trace(self, records, outer, sigma, inner_active, x_clean, x0hat, y, resid_pre, target):
        resid = self.residual_norm(x_clean, y).mean().item()
        record = {
            "outer": int(outer),
            "inner": -1.0,
            "sigma": float(sigma),
            "inner_active": float(inner_active),
            "resid_pre": float(resid_pre),
            "resid": float(resid),
            "inner_dist": float((x_clean - x0hat).abs().mean().item()) if inner_active else 0.0,
            "x_abs_max": float(x_clean.abs().max().item()),
        }
        if target is not None:
            ssim_inner, nmse_inner = compute_ssim_nmse(self.to_real(x_clean), self.to_real(target))
            record["ssim_inner"] = ssim_inner
            record["nmse_inner"] = nmse_inner
        records.append(record)

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
        n_outer = self.annealing.num_steps
        disable_tqdm = self.log_level in {"VAL", "WARN"} or not verbose

        if self.log_level in {"INFO", "DEBUG"}:
            print(
                "[P-DAPS-core] "
                f"num_steps={self.num_steps} gamma={self.gamma:g} warm_fraction={self.warm_fraction:g} "
                f"inner_sigma_max={self.inner_sigma_max:g} solve_lam_floor={self.solve_lam_floor:g} "
                f"lr_min_ratio={self.lr_min_ratio:g} sigma_stop_truncate={self.sigma_stop_truncate}",
                flush=True,
            )

        for i in tqdm.trange(n_outer, desc="P-DAPS-core", disable=disable_tqdm):
            sigma = self.annealing.sigma_steps[i]
            sigma_f = float(sigma)
            if self.sigma_stop_truncate is not None and sigma_f < self.sigma_stop_truncate:
                if self.log_level in {"INFO", "DEBUG"}:
                    print(
                        f"[P-DAPS-core] sigma_stop_truncate={self.sigma_stop_truncate:g} "
                        f"reached at outer={i} sigma={sigma_f:g}; returning last clean state",
                        flush=True,
                    )
                if x_prev is not None:
                    xt = self.to_real(x_prev)
                break

            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))
            resid_pre = self.residual_norm(x0hat, y).mean().item()
            inner_active = sigma_f <= self.inner_sigma_max
            self.last_gate_stats.append({
                "kind": "gate",
                "outer": int(i),
                "sigma": sigma_f,
                "resid_pre": float(resid_pre),
                "inner_active": bool(inner_active),
            })

            if inner_active:
                x_init = self._init_inner(x_prev, x0hat)
                x_clean, avg_cg, avg_drift = self._inner_sample(
                    x_init,
                    x0hat,
                    y,
                    sigma,
                    i / max(1, n_outer),
                    return_stats=self.log_level in {"INFO", "DEBUG"},
                )
                inner_stats = f" CG={avg_cg:.1f} drift={avg_drift:.3e}" if self.log_level in {"INFO", "DEBUG"} else ""
            else:
                x_clean = x0hat
                inner_stats = ""

            if trace_records is not None:
                self._append_trace(
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

            if self.log_level in {"INFO", "DEBUG"} and (i < 5 or i % 20 == 0 or i >= n_outer - 3):
                msg = (
                    f"[P-DAPS-core] outer={i:3d} sigma={sigma_f:.4f} "
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


class PDAPSWarm(PDAPS):
    pass
