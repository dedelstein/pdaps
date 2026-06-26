"""pULA baseline for Cartesian multicoil MRI."""

import math

import torch
from tqdm import tqdm

from mri_ops import conjgrad, mri_A, mri_AH, mri_AHA, to_complex, to_real

try:
    from algo.base import Algo
    from utils.scheduler import Scheduler
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pULA expects InverseBench to be installed or on PYTHONPATH. "
        "For the public repo, install the submodule with "
        "`pip install -e libs/inversebench`."
    ) from exc


class PULA(Algo):
    """Preconditioned Unadjusted Langevin Algorithm for multicoil MRI."""

    def __init__(
        self,
        net,
        forward_op,
        noise_scheduler_config,
        K=4,
        gamma=0.5,
        cg_iter=10,
        log_level="INFO",
    ):
        super().__init__(net, forward_op)
        self.net = net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.noise_scheduler = Scheduler(**noise_scheduler_config)
        self.K = int(K)
        self.gamma = float(gamma)
        self.cg_iter = int(cg_iter)
        self.log_level = str(log_level)

    def A(self, x):
        return mri_A(self, x)

    def AH(self, y):
        return mri_AH(self, y)

    def AHA(self, x):
        return mri_AHA(self, x)

    @staticmethod
    def to_complex(x):
        return to_complex(x)

    @staticmethod
    def to_real(x):
        return to_real(x)

    def score(self, x, sigma):
        x_real = self.to_real(x)
        sigma_t = torch.as_tensor(sigma, device=x.device)
        with torch.no_grad():
            denoised = self.net(x_real, sigma_t)
        return (self.to_complex(denoised) - x) / (sigma ** 2)

    def init_sample(self, AHy, device):
        sigma_max = self.noise_scheduler.sigma_max
        lam = 1.0 / (sigma_max ** 2)
        batch, height, width = AHy.shape

        n1 = torch.randn(
            batch,
            *self.forward_op.maps.shape[-3:],
            dtype=self.forward_op.maps.dtype,
            device=device,
        )
        n2 = torch.randn(batch, height, width, dtype=torch.cfloat, device=device)
        noise_rhs = self.AH(n1) + (1.0 / sigma_max) * n2
        rhs = AHy + math.sqrt(2.0) * noise_rhs

        x = torch.zeros(batch, height, width, dtype=torch.cfloat, device=device)
        return conjgrad(self.AHA, rhs, x, lam, self.cg_iter, x_is_zero=True)

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        y = torch.view_as_complex(observation).to(device)
        if num_samples > 1:
            y = y.expand(num_samples, -1, -1, -1)

        AHy = self.AH(y)
        sigmas = torch.tensor(self.noise_scheduler.sigma_steps, device=device)
        x = self.init_sample(AHy, device)

        verbose = kwargs.get("verbose", True)
        disable_tqdm = self.log_level in {"VAL", "WARN"} or not verbose
        pbar = tqdm(range(self.noise_scheduler.num_steps), desc="pULA", disable=disable_tqdm)

        for i in pbar:
            sigma = sigmas[i].item()
            lam = 1.0 / (sigma ** 2)

            for _ in range(self.K):
                score = self.score(x, sigma)
                AHAx = self.AHA(x)
                likelihood_grad = AHy - AHAx

                rhs = AHAx + lam * x
                rhs = rhs + (self.gamma / 2.0) * (likelihood_grad + score)

                batch, height, width = x.shape
                n1 = math.sqrt(2.0) * torch.randn_like(y)
                n2 = math.sqrt(2.0) * torch.randn(batch, height, width, dtype=x.dtype, device=device)
                rhs = rhs + math.sqrt(self.gamma) * self.AH(n1) + math.sqrt(self.gamma * lam) * n2

                x = conjgrad(self.AHA, rhs, x, lam, self.cg_iter)

            if not disable_tqdm:
                pbar.set_description(f"pULA sigma={sigma:.4f}")

        return self.to_real(x).float()


pULA = PULA
