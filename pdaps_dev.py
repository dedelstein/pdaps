"""Development P-DAPS sampler with compact ablation controls.

The public default implementation is ``pdaps.PDAPS``.  This module keeps a small
variant surface for inspecting the deterministic exact split against the earlier
iterative inner correction.
"""

import math

import torch
import tqdm

from mri_ops import conjgrad, mri_A, mri_AH, mri_AHA, to_complex, to_real

try:
    from algo.base import Algo
    from utils.diffusion import DiffusionSampler
    from utils.scheduler import Scheduler
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "P-DAPS expects InverseBench to be installed or on PYTHONPATH. "
        "For the public repo, install the submodule with "
        "`pip install -e libs/inversebench`."
    ) from exc


DAPS_TAU = 0.002028752174814177
DEFAULT_NULL_BLEND = 1.0
DEFAULT_INNER_SIGMA_MAX = 2.0
DEFAULT_SPLIT_CG_ITER = 30
DEFAULT_INNER_STEPS = 50
DEFAULT_GAMMA = 0.5
DEFAULT_SOLVE_LAM_FLOOR = 3.0
DEFAULT_LR_MIN_RATIO = 0.01
DEFAULT_CG_ITER = 10

EXACT_SPLIT = "exact_split"
ITERATIVE = "iterative"
INNER_MODES = {EXACT_SPLIT, ITERATIVE}

NO_NOISE = "none"
FULL_NOISE = "full"
RANGE_NOISE = "range_only"
IMAGE_NOISE = "image_only"
NULL_NOISE = "null_only"
NOISE_MODES = {NO_NOISE, FULL_NOISE, RANGE_NOISE, IMAGE_NOISE, NULL_NOISE}

HALF_STEP = 0.5
COMPLEX_NOISE_SCALE = math.sqrt(2.0)
MIN_MASK_SAMPLES = 1.0
MIN_COUNT = 1


class PDAPSDev(Algo):
    """P-DAPS with exact-split and iterative inner correction modes."""

    def __init__(
        self,
        net,
        forward_op,
        annealing_scheduler_config=None,
        diffusion_scheduler_config=None,
        tau=DAPS_TAU,
        inner_mode=EXACT_SPLIT,
        null_blend=DEFAULT_NULL_BLEND,
        inner_sigma_max=DEFAULT_INNER_SIGMA_MAX,
        split_cg_iter=DEFAULT_SPLIT_CG_ITER,
        sigma_stop_truncate=None,
        inner_steps=DEFAULT_INNER_STEPS,
        gamma=DEFAULT_GAMMA,
        lr_min_ratio=DEFAULT_LR_MIN_RATIO,
        solve_lam_floor=DEFAULT_SOLVE_LAM_FLOOR,
        cg_iter=DEFAULT_CG_ITER,
        noise_tau=0.0,
        noise_mode=FULL_NOISE,
        target_null_rho=None,
        warm_fraction=0.0,
        log_level="INFO",
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.annealing = Scheduler(**(annealing_scheduler_config or {}))
        self.diffusion_config = diffusion_scheduler_config or {}

        self.tau = float(tau)
        self.inner_mode = str(inner_mode)
        self.null_blend = float(null_blend)
        self.inner_sigma_max = float(inner_sigma_max)
        self.split_cg_iter = int(split_cg_iter)
        self.sigma_stop_truncate = None if sigma_stop_truncate is None else float(sigma_stop_truncate)

        self.inner_steps = int(inner_steps)
        self.gamma = float(gamma)
        self.lr_min_ratio = float(lr_min_ratio)
        self.solve_lam_floor = float(solve_lam_floor)
        self.cg_iter = int(cg_iter)
        self.noise_tau = float(noise_tau)
        self.noise_mode = str(noise_mode)
        self.target_null_rho = None if target_null_rho is None else float(target_null_rho)
        self.warm_fraction = float(warm_fraction)
        self.log_level = str(log_level)

        if self.inner_mode not in INNER_MODES:
            raise ValueError(f"inner_mode must be one of {sorted(INNER_MODES)}")
        if not 0.0 <= self.null_blend <= 1.0:
            raise ValueError("null_blend must be in [0, 1]")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if self.cg_iter < 1 or self.split_cg_iter < 1:
            raise ValueError("cg_iter and split_cg_iter must be positive")
        if self.noise_tau < 0.0:
            raise ValueError("noise_tau must be nonnegative")
        if self.noise_mode not in NOISE_MODES:
            raise ValueError(f"noise_mode must be one of {sorted(NOISE_MODES)}")
        if not 0.0 <= self.warm_fraction <= 1.0:
            raise ValueError("warm_fraction must be in [0, 1]")

    @staticmethod
    def to_complex(x):
        return to_complex(x)

    @staticmethod
    def to_real(x):
        return to_real(x)

    def A(self, x):
        return mri_A(self, x)

    def AH(self, y):
        return mri_AH(self, y)

    def AHA(self, x):
        return mri_AHA(self, x)

    def fourier_mask(self, x):
        mask = self.forward_op.mask.to(device=x.device, dtype=x.real.dtype)
        height, width = x.shape[-2], x.shape[-1]
        if mask.shape[-2] == 1 and mask.shape[-1] == width:
            return mask.reshape(1, 1, width)
        if mask.shape[-2] == height and mask.shape[-1] == 1:
            return mask.reshape(1, height, 1)
        if mask.shape[-2] == height and mask.shape[-1] == width:
            return mask.reshape(1, height, width)
        raise ValueError(f"Unsupported MRI mask shape {tuple(mask.shape)} for image {tuple(x.shape)}")

    def project_range(self, x):
        mask = self.fourier_mask(x)
        return self.forward_op.ifft(mask * self.forward_op.fft(x))

    def project_null(self, x):
        return x - self.project_range(x)

    def residual_norm(self, x, y):
        residual = self.A(x) - y
        mask_samples = max(MIN_MASK_SAMPLES, float(self.forward_op.mask.expand_as(y[:1]).sum().item()))
        return residual.abs().square().flatten(start_dim=1).sum(dim=1).sqrt() / math.sqrt(mask_samples)

    def _lambdas(self, sigma):
        lam_raw = 1.0 / (float(sigma) ** 2)
        lam_target = (self.tau ** 2) * lam_raw
        lam_target_null = lam_target
        if self.target_null_rho is not None:
            lam_target_null = max(lam_target, self.target_null_rho * lam_raw)
        return {
            "raw": lam_raw,
            "target": lam_target,
            "target_null": lam_target_null,
            "solve": max(lam_raw, self.solve_lam_floor),
        }

    def _effective_gamma(self, sigma, ratio):
        lam_raw = 1.0 / (float(sigma) ** 2)
        anneal = 1.0 + float(ratio) * (self.lr_min_ratio - 1.0)
        return self.gamma * anneal * lam_raw

    def _solve(self, rhs, lam, max_iter):
        return conjgrad(
            self.AHA,
            rhs,
            torch.zeros_like(rhs),
            lam,
            max_iter,
            x_is_zero=True,
        )

    def _exact_correction(self, x_prev, x0hat, y, sigma):
        lam_target = (self.tau ** 2) / (float(sigma) ** 2)
        rhs = self.project_range(self.AH(y) + lam_target * x0hat)

        def range_normal(v):
            return self.project_range(self.AHA(self.project_range(v)))

        range_sol = conjgrad(
            range_normal,
            rhs,
            torch.zeros_like(rhs),
            lam_target,
            self.split_cg_iter,
            x_is_zero=True,
        )
        null_source = self.null_blend * x0hat + (1.0 - self.null_blend) * x_prev
        return (self.project_range(range_sol) + self.project_null(null_source)).detach()

    def _iterative_init(self, x_prev, x0hat):
        if x_prev is None or self.warm_fraction == 0.0:
            return x0hat
        return self.warm_fraction * x_prev + (1.0 - self.warm_fraction) * x0hat

    def _noise_rhs(self, x, y, lam_solve):
        if self.noise_mode == NO_NOISE or self.noise_tau == 0.0:
            return torch.zeros_like(x)

        kspace_noise = COMPLEX_NOISE_SCALE * torch.randn_like(y)
        image_noise = COMPLEX_NOISE_SCALE * torch.randn(x.shape, dtype=x.dtype, device=x.device)
        range_rhs = self.AH(kspace_noise)
        image_rhs = math.sqrt(lam_solve) * image_noise

        if self.noise_mode == RANGE_NOISE:
            return range_rhs
        if self.noise_mode == IMAGE_NOISE:
            return image_rhs
        if self.noise_mode == NULL_NOISE:
            return math.sqrt(lam_solve) * self.project_null(image_noise)
        if self.noise_mode == FULL_NOISE:
            return range_rhs + image_rhs
        raise ValueError(f"unknown noise_mode {self.noise_mode!r}")

    def _iterative_step(self, x, x0hat, y, sigma, ratio):
        lams = self._lambdas(sigma)
        delta = x - x0hat
        if lams["target"] == lams["target_null"]:
            prior_score = -lams["target"] * delta
        else:
            prior_score = -(
                lams["target"] * self.project_range(delta)
                + lams["target_null"] * self.project_null(delta)
            )

        likelihood_score = self.AH(y - self.A(x))
        drift = self._solve(likelihood_score + prior_score, lams["solve"], self.cg_iter)

        gamma_eff = self._effective_gamma(sigma, ratio)
        step = HALF_STEP * gamma_eff * drift
        if self.noise_tau > 0.0 and self.noise_mode != NO_NOISE:
            noise = self._solve(self._noise_rhs(x, y, lams["solve"]), lams["solve"], self.cg_iter)
            step = step + math.sqrt(gamma_eff * self.noise_tau) * noise
        return x + step

    def _iterative_correction(self, x_prev, x0hat, y, sigma, ratio):
        x = self._iterative_init(x_prev, x0hat).detach()
        x0hat = x0hat.detach()
        for _ in range(self.inner_steps):
            x = self._iterative_step(x, x0hat, y, sigma, ratio)
        return x.detach()

    def _log_start(self, n_outer):
        if self.log_level not in {"INFO", "DEBUG"}:
            return
        msg = (
            f"[P-DAPS-dev] n_outer={n_outer} mode={self.inner_mode} tau={self.tau:g} "
            f"inner_sigma_max={self.inner_sigma_max:g} null_blend={self.null_blend:g}"
        )
        if self.inner_mode == ITERATIVE:
            msg += (
                f" inner_steps={self.inner_steps} gamma={self.gamma:g} "
                f"noise_tau={self.noise_tau:g} noise_mode={self.noise_mode} "
                f"target_null_rho={self.target_null_rho}"
            )
        print(msg, flush=True)

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
        ) * float(self.annealing.sigma_max)

        x_prev = None
        n_outer = self.annealing.num_steps
        disable_tqdm = self.log_level in {"VAL", "WARN"} or not verbose
        self._log_start(n_outer)

        for i in tqdm.trange(n_outer, desc="P-DAPS-dev", disable=disable_tqdm):
            sigma = float(self.annealing.sigma_steps[i])
            if self.sigma_stop_truncate is not None and sigma < self.sigma_stop_truncate:
                if x_prev is not None:
                    xt = self.to_real(x_prev)
                break

            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))

            if sigma <= self.inner_sigma_max:
                if self.inner_mode == EXACT_SPLIT:
                    x_carry = x0hat if x_prev is None else x_prev
                    x_clean = self._exact_correction(x_carry, x0hat, y, sigma)
                else:
                    x_clean = self._iterative_correction(
                        x_prev,
                        x0hat,
                        y,
                        sigma,
                        i / max(MIN_COUNT, n_outer),
                    )
            else:
                x_clean = x0hat

            if self.log_level == "DEBUG":
                resid = self.residual_norm(x_clean, y).mean().item()
                tqdm.tqdm.write(
                    f"[P-DAPS-dev] outer={i:3d} sigma={sigma:.4g} "
                    f"inner={int(sigma <= self.inner_sigma_max)} resid={resid:.3e}"
                )

            x_prev = x_clean
            sigma_next = self.annealing.sigma_steps[i + 1]
            xt = self.to_real(x_clean + COMPLEX_NOISE_SCALE * torch.randn_like(x_clean) * sigma_next)

        return xt.float()


PDAPSCore = PDAPSDev
