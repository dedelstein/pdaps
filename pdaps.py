"""P-DAPS sampler for Cartesian multicoil MRI."""

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
DEFAULT_SPLIT_CG_ITER = 30
DEFAULT_INNER_SIGMA_MAX = 2.0

# Match sigma^2 variance per real channel for complex noise.
COMPLEX_NOISE_SCALE = math.sqrt(2.0)


class PDAPS(Algo):
    """P-DAPS with exact range correction and null-space blending."""

    def __init__(
        self,
        net,
        forward_op,
        annealing_scheduler_config=None,
        diffusion_scheduler_config=None,
        tau=DAPS_TAU,
        null_blend=DEFAULT_NULL_BLEND,
        split_cg_iter=DEFAULT_SPLIT_CG_ITER,
        inner_sigma_max=DEFAULT_INNER_SIGMA_MAX,
        sigma_stop_truncate=None,
        log_level="INFO",
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.annealing = Scheduler(**(annealing_scheduler_config or {}))
        self.diffusion_config = diffusion_scheduler_config or {}
        self.tau = float(tau)
        self.null_blend = float(null_blend)
        self.split_cg_iter = int(split_cg_iter)
        self.inner_sigma_max = float(inner_sigma_max)
        self.sigma_stop_truncate = None if sigma_stop_truncate is None else float(sigma_stop_truncate)
        self.log_level = str(log_level)
        if not 0.0 <= self.null_blend <= 1.0:
            raise ValueError("null_blend must be in [0, 1]")

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
        if self.log_level in {"INFO", "DEBUG"}:
            print(
                f"[P-DAPS] n_outer={n_outer} tau={self.tau:g} null_blend={self.null_blend:g} "
                f"split_cg_iter={self.split_cg_iter} inner_sigma_max={self.inner_sigma_max:g}",
                flush=True,
            )

        for i in tqdm.trange(n_outer, desc="P-DAPS", disable=disable_tqdm):
            sigma = float(self.annealing.sigma_steps[i])
            if self.sigma_stop_truncate is not None and sigma < self.sigma_stop_truncate:
                if x_prev is not None:
                    xt = self.to_real(x_prev)
                break

            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))

            if sigma <= self.inner_sigma_max:
                x_carry = x0hat if x_prev is None else x_prev
                x_clean = self._exact_correction(x_carry, x0hat, y, sigma)
            else:
                x_clean = x0hat

            x_prev = x_clean
            sigma_next = self.annealing.sigma_steps[i + 1]
            xt = self.to_real(x_clean + COMPLEX_NOISE_SCALE * torch.randn_like(x_clean) * sigma_next)

        return xt.float()


PDAPSv3 = PDAPS
