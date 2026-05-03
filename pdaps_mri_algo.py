import math

import torch
import tqdm

from libs.inversebench.algo.base import Algo
from libs.inversebench.algo.pula import conjgrad
from libs.inversebench.utils.diffusion import DiffusionSampler
from libs.inversebench.utils.scheduler import Scheduler


class MRIInnerPULA:
    def __init__(self, num_steps, step_size=None, gamma=None, lr=None, cg_iter=10, lr_min_ratio=1.0, tau=None):
        self.num_steps = int(num_steps)
        if step_size is None:
            step_size = gamma if gamma is not None else lr
        self.step_size = float(0.5 if step_size is None else step_size)
        self.cg_iter = int(cg_iter)
        self.lr_min_ratio = float(lr_min_ratio)

    def step(self, pdaps, x, x0hat, y, sigma, ratio):
        lam = 1.0 / float(sigma) ** 2
        gamma = self.step_size * (1.0 + ratio * (self.lr_min_ratio - 1.0))

        score_lik = pdaps.AH(y - pdaps.A(x))
        score_prior = -(x - x0hat) * lam
        drift = pdaps.solve(score_lik + score_prior, lam)

        n1 = torch.randn_like(y)
        n2 = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        noise = pdaps.solve(pdaps.AH(n1) + math.sqrt(lam) * n2, lam)

        return x + 0.5 * gamma * drift + math.sqrt(gamma) * noise

    def sample(self, pdaps, x, x0hat, y, sigma, ratio):
        x = x.detach()
        x0hat = x0hat.detach()
        for _ in range(self.num_steps):
            x = self.step(pdaps, x, x0hat, y, sigma, ratio)
            if not torch.isfinite(torch.view_as_real(x)).all():
                return torch.zeros_like(x)
        return x.detach()


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
        eps=1e-8,
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.annealing = Scheduler(**annealing_scheduler_config)
        self.diffusion_config = diffusion_scheduler_config
        self.inner = MRIInnerPULA(**lgvd_config)
        self.warm_mode = warm_mode
        self.warm_fraction = float(warm_fraction)
        self.eps = float(eps)
        self.last_gate_stats = []

    @staticmethod
    def to_complex(x):
        return torch.complex(x[:, 0], x[:, 1])

    @staticmethod
    def to_real(x):
        return torch.stack([x.real, x.imag], dim=1)

    def A(self, x):
        return self.forward_op.mask * self.forward_op.fft(self.forward_op.maps * x.unsqueeze(1))

    def AH(self, y):
        return (self.forward_op.ifft(y) * self.forward_op.maps.conj()).sum(dim=1)

    def AHA(self, x):
        return self.AH(self.A(x))

    def solve(self, rhs, lam):
        return conjgrad(self.AHA, rhs, torch.zeros_like(rhs), lam, self.inner.cg_iter)

    def residual_norm(self, x, y):
        r = self.A(x) - y
        m = max(1.0, float(self.forward_op.mask.expand_as(y[:1]).sum().item()))
        return r.abs().square().flatten(start_dim=1).sum(dim=1).sqrt() / math.sqrt(m)

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
        steps = tqdm.trange(self.annealing.num_steps, desc="P-DAPS") if verbose else range(self.annealing.num_steps)
        for i in steps:
            sigma = self.annealing.sigma_steps[i]
            ode = DiffusionSampler(Scheduler(**self.diffusion_config, sigma_max=sigma))
            x0hat = self.to_complex(ode.sample(self.net, xt, SDE=False, verbose=False))
            x_init = self.init_inner(x_prev, x0hat, y)
            x_clean = self.inner.sample(self, x_init, x0hat, y, sigma, i / max(1, self.annealing.num_steps))
            x_prev = x_clean
            xt = self.to_real(x_clean + torch.randn_like(x_clean) * self.annealing.sigma_steps[i + 1])

        return xt.float()


class PDAPSWarm(PDAPS):
    def __init__(self, *args, warm_mode="fixed", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)


class PDAPSAdaptive(PDAPS):
    def __init__(self, *args, warm_mode="adaptive", **kwargs):
        super().__init__(*args, warm_mode=warm_mode, **kwargs)
