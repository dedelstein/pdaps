import torch
import math
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler

# -----------------------------------------------------------------------------------------------
# Paper: Fast and Robust Diffusion Posterior Sampling for MR Image Reconstruction
#        Using the Preconditioned Unadjusted Langevin Algorithm
# Reference: Blumenthal et al., arXiv:2512.05791, Dec 2025
# BART C implementation: libs/bart/src/iter/italgos.c (eulermaruyama_precond)
# -----------------------------------------------------------------------------------------------


def conjgrad(normal_op, b, x, lam, max_iter, tol=1e-10):
    """
    Conjugate gradient solver for (A^H A + λI) x = b.

    Solves without forming the matrix explicitly — only uses normal_op(x) = A^H A x.
    Matches BART's conjgrad() in libs/bart/src/iter/italgos.c:497.

    Args:
        normal_op: callable, x -> A^H A x
        b: RHS tensor (complex)
        x: initial guess / warm start (complex, modified in-place)
        lam: scalar regularization (1/σ² for pULA)
        max_iter: number of CG iterations
        tol: early stopping tolerance on residual norm
    Returns:
        x: solution tensor (complex)
    """
    # r = b - (A^H A + λI) x
    r = b - normal_op(x) - lam * x
    p = r.clone()
    rs_old = torch.sum(r.conj() * r).real

    for _ in range(max_iter):
        Ap = normal_op(p) + lam * p            # (A^H A + λI) p
        pAp = torch.sum(p.conj() * Ap).real
        if pAp.abs() < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r.conj() * r).real
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


class pULA(Algo):
    """
    Preconditioned Unadjusted Langevin Algorithm for multicoil MRI.

    Matches BART's sample.c outer loop + italgos.c:eulermaruyama_precond.
    Uses exact likelihood (no annealing) with CG-based preconditioning
    M_t = (A^H A + σ_t^{-2} I)^{-1} so that no guidance_scale tuning is needed.
    """

    def __init__(self, net, forward_op,
                 noise_scheduler_config,
                 K=4,            # Langevin steps per noise level
                 gamma=0.5,      # fixed step size
                 cg_iter=10):    # CG iterations for preconditioning
        super().__init__(net, forward_op)
        self.noise_scheduler = Scheduler(**noise_scheduler_config)
        self.K = K
        self.gamma = gamma
        self.cg_iter = cg_iter

    # -- Complex-domain MRI operators using forward_op internals --

    def A(self, x):
        """Forward: complex image (B,H,W) -> masked complex k-space (B,C,H,W)."""
        coils = self.forward_op.maps * x.unsqueeze(1)       # (B,C,H,W)
        return self.forward_op.mask * self.forward_op.fft(coils)

    def AH(self, y):
        """Adjoint: masked complex k-space (B,C,H,W) -> complex image (B,H,W)."""
        return (self.forward_op.ifft(y) * self.forward_op.maps.conj()).sum(dim=1)

    def AHA(self, x):
        """Normal operator: A^H A x."""
        return self.AH(self.A(x))

    # -- Conversions between 2-channel real (B,2,H,W) and complex (B,H,W) --

    @staticmethod
    def to_complex(x_real):
        """(B,2,H,W) float -> (B,H,W) complex."""
        return torch.complex(x_real[:, 0], x_real[:, 1])

    @staticmethod
    def to_real(x_complex):
        """(B,H,W) complex -> (B,2,H,W) float."""
        return torch.stack([x_complex.real, x_complex.imag], dim=1)

    # -- Denoiser wrapper --

    def score(self, x, sigma):
        """
        Compute prior score via Tweedie's formula: score = (D(x,σ) - x) / σ².
        Handles conversion to/from 2-channel real for the network.

        Args:
            x: complex image (B,H,W)
            sigma: current noise level (scalar)
        Returns:
            score: complex image (B,H,W)
        """
        x_real = self.to_real(x)  # (B,2,H,W)
        sigma_t = torch.as_tensor(sigma, device=x.device)
        with torch.no_grad():
            denoised = self.net(x_real, sigma_t)       # (B,2,H,W)
        denoised_c = self.to_complex(denoised)
        return (denoised_c - x) / (sigma ** 2)

    # -- Initialization (sample.c:106-138) --

    def init_sample(self, AHy, device):
        """
        Draw initial sample from posterior with flat Gaussian prior:
            x ~ CN(M1 @ AHy, M1^{-1})
        where M1 = (A^H A + σ_max^{-2} I)^{-1}.

        This is done as one pULA step from zero with step=2 and no prior score,
        matching sample.c:get_init().
        """
        sigma_max = self.noise_scheduler.sigma_max
        lam = 1.0 / (sigma_max ** 2)
        B, H, W = AHy.shape

        # TODO: draw noise in measurement space and image space (Eq. 10)
        #   n1 has shape of k-space: (B, num_coils, H, W) complex
        #   n2 has shape of image:   (B, H, W) complex
        #   noise_rhs = AH(n1) + (1/sigma_max) * n2
        n1 = torch.randn_like(self.forward_op.maps).expand(B, -1, -1, -1)
        # TODO: verify n1 shape matches k-space dims — should be (B, C, H, W) complex Gaussian
        n2 = torch.randn(B, H, W, dtype=torch.cfloat, device=device)
        noise_rhs = self.AH(n1) + (1.0 / sigma_max) * n2

        # RHS for CG: AHy + sqrt(2) * noise_rhs  (step=2 in BART)
        rhs = AHy + math.sqrt(2.0) * noise_rhs

        # Solve (A^H A + λI) x = rhs, starting from zeros
        x = torch.zeros(B, H, W, dtype=torch.cfloat, device=device)
        x = conjgrad(self.AHA, rhs, x, lam, self.cg_iter)

        return x

    # -- Main inference loop --

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        """
        Run pULA posterior sampling.

        Args:
            observation: output of forward_op(data), i.e. view_as_real of masked k-space
                         shape (1, C, H, W, 2) float
            num_samples: number of posterior samples
        Returns:
            reconstruction in 2-channel real format (B,2,H,W) to match DPS/DAPS output
        """
        device = self.forward_op.device

        # Convert observation back to complex k-space (B,C,H,W)
        # observation comes from forward_op.__call__ which returns torch.view_as_real(masked_kspace)
        y = torch.view_as_complex(observation).to(device)  # (B,C,H,W)
        if num_samples > 1:
            y = y.expand(num_samples, -1, -1, -1)

        # Precompute A^H y
        AHy = self.AH(y)   # (B,H,W) complex

        sigmas = torch.tensor(self.noise_scheduler.sigma_steps, device=device)
        N = self.noise_scheduler.num_steps

        # Initialize from posterior with flat prior (Eq. 9)
        x = self.init_sample(AHy, device)
        print(f"[pULA] after init: x.abs.max={x.abs().max().item():.3e}  AHy.abs.max={AHy.abs().max().item():.3e}")

        # Anneal from sigma_max (sigma_steps[0]) down to sigma_min (sigma_steps[N-1]).
        # Inversebench's Scheduler stores sigma_steps in decreasing order, so iterate i=0..N-1.
        pbar = tqdm(range(N), desc='pULA')

        for i in pbar:
            sigma = sigmas[i].item()
            lam = 1.0 / (sigma ** 2)

            for k in range(self.K):
                # --- Prior score via denoiser (Tweedie) ---
                s = self.score(x, sigma)                    # (B,H,W) complex

                # --- Likelihood gradient: A^H(y - Ax) ---
                likelihood_grad = AHy - self.AHA(x)         # (B,H,W) complex

                # --- Build RHS for CG ---
                # EM step: x_new = x + (γ/2) M (lik + score) + sqrt(γ) M z
                # Implicit form: solve (AHA + λI) x_new = rhs where
                #   rhs = (AHA + λI)x + (γ/2)(lik + score) + sqrt(γ)·(AH(n1) + sqrt(λ)·n2)
                rhs = self.AHA(x) + lam * x                       # (A^H A + λI) x
                rhs = rhs + (self.gamma / 2.0) * likelihood_grad  # + (γ/2) * A^H(y - Ax)
                rhs = rhs + (self.gamma / 2.0) * s                # + (γ/2) * prior score

                # --- Noise injection (Eq. 10) ---
                # TODO: draw n1 ~ CN(0, I) in k-space shape, n2 ~ CN(0, I) in image shape
                # TODO: rhs += sqrt(γ) * AH(n1) + sqrt(γ * λ) * n2
                B, H, W = x.shape
                n1 = torch.randn_like(y)                     # (B,C,H,W) complex
                n2 = torch.randn(B, H, W, dtype=x.dtype, device=device)
                rhs = rhs + math.sqrt(self.gamma) * self.AH(n1)
                rhs = rhs + math.sqrt(self.gamma * lam) * n2

                # --- CG solve: (A^H A + λI) x_new = rhs ---
                # warm-started from current x
                x = conjgrad(self.AHA, rhs, x, lam, self.cg_iter)

            if i % 20 == 0 or i < 5:
                tqdm.write(f"[pULA] outer={i:3d} σ={sigma:.4f} x.abs.max={x.abs().max().item():.3e} s.abs.max={s.abs().max().item():.3e}")
            pbar.set_description(f'pULA σ={sigma:.4f}')

        # Convert back to 2-channel real (B,2,H,W) for InverseBench evaluator
        return self.to_real(x).float()
