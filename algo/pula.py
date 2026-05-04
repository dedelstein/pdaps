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


def conjgrad(normal_op, b, x, lam, max_iter, tol=1e-10, x_is_zero=False):
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
        x_is_zero: skip the initial normal_op(x) when x is known to be zero
    Returns:
        x: solution tensor (complex)
    """
    # r = b - (A^H A + λI) x
    r = b.clone() if x_is_zero else b - normal_op(x) - lam * x
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


def _mri_fft_cache(algo):
    maps = algo.forward_op.maps
    mask = algo.forward_op.mask
    even_shape = maps.shape[-2] % 2 == 0 and maps.shape[-1] % 2 == 0
    key = (
        maps.data_ptr(),
        mask.data_ptr(),
        tuple(maps.shape),
        tuple(mask.shape),
        maps.device,
        mask.device,
        maps.dtype,
        mask.dtype,
        even_shape,
    )
    if getattr(algo, "_mri_fft_cache_key", None) != key:
        algo._mri_fft_cache_key = key
        algo._mri_even_shape = even_shape
        if even_shape:
            algo._mri_maps_shift = torch.fft.fftshift(maps, dim=(-2, -1))
            algo._mri_maps_shift_conj = algo._mri_maps_shift.conj()
            algo._mri_mask_unshift = torch.fft.ifftshift(mask, dim=(-2, -1))
        else:
            algo._mri_maps_shift = None
            algo._mri_maps_shift_conj = None
            algo._mri_mask_unshift = None
    return algo._mri_even_shape, algo._mri_maps_shift, algo._mri_maps_shift_conj, algo._mri_mask_unshift


def _centered_fft(x, dim):
    return torch.fft.ifftshift(
        torch.fft.fft(torch.fft.fftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def _centered_ifft(x, dim):
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def _mri_cartesian_mask_dim(algo):
    mask = algo.forward_op.mask
    if mask.shape[-2] == 1 and mask.shape[-1] > 1:
        return -1
    if mask.shape[-1] == 1 and mask.shape[-2] > 1:
        return -2
    return None


def mri_A(algo, x):
    even_shape, maps_shift, _, mask_unshift = _mri_fft_cache(algo)
    if not even_shape:
        coils = algo.forward_op.maps * x.unsqueeze(1)
        return algo.forward_op.mask * algo.forward_op.fft(coils)

    x_shift = torch.fft.fftshift(x, dim=(-2, -1))
    kspace_unshift = torch.fft.fft2(maps_shift * x_shift.unsqueeze(1), dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(mask_unshift * kspace_unshift, dim=(-2, -1))


def mri_AH(algo, y):
    even_shape, maps_shift, maps_shift_conj, _ = _mri_fft_cache(algo)
    if not even_shape:
        return (algo.forward_op.ifft(y) * algo.forward_op.maps.conj()).sum(dim=1)

    y_unshift = torch.fft.ifftshift(y, dim=(-2, -1))
    image_shift = torch.fft.ifft2(y_unshift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift((image_shift * maps_shift_conj).sum(dim=1), dim=(-2, -1))


def mri_AHA(algo, x):
    even_shape, maps_shift, maps_shift_conj, mask_unshift = _mri_fft_cache(algo)
    mask_dim = _mri_cartesian_mask_dim(algo)
    if getattr(algo, "use_fast_aha", True) and even_shape and mask_dim is not None:
        coils = algo.forward_op.maps * x.unsqueeze(1)
        masked = algo.forward_op.mask * _centered_fft(coils, dim=mask_dim)
        return (_centered_ifft(masked, dim=mask_dim) * algo.forward_op.maps.conj()).sum(dim=1)

    if not even_shape:
        return mri_AH(algo, mri_A(algo, x))

    x_shift = torch.fft.fftshift(x, dim=(-2, -1))
    kspace_unshift = torch.fft.fft2(maps_shift * x_shift.unsqueeze(1), dim=(-2, -1), norm="ortho")
    image_shift = torch.fft.ifft2(mask_unshift * kspace_unshift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift((image_shift * maps_shift_conj).sum(dim=1), dim=(-2, -1))


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
                 cg_iter=10,     # CG iterations for preconditioning
                 use_fast_aha=True):
        super().__init__(net, forward_op)
        self.noise_scheduler = Scheduler(**noise_scheduler_config)
        self.K = K
        self.gamma = gamma
        self.cg_iter = cg_iter
        self.use_fast_aha = bool(use_fast_aha)

    # -- Complex-domain MRI operators using forward_op internals --

    def A(self, x):
        """Forward: complex image (B,H,W) -> masked complex k-space (B,C,H,W)."""
        return mri_A(self, x)

    def AH(self, y):
        """Adjoint: masked complex k-space (B,C,H,W) -> complex image (B,H,W)."""
        return mri_AH(self, y)

    def AHA(self, x):
        """Normal operator: A^H A x."""
        return mri_AHA(self, x)

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
        x = conjgrad(self.AHA, rhs, x, lam, self.cg_iter, x_is_zero=True)

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
                AHAx = self.AHA(x)                          # (B,H,W) complex
                likelihood_grad = AHy - AHAx                # (B,H,W) complex

                # --- Build RHS for CG ---
                # EM step: x_new = x + (γ/2) M (lik + score) + sqrt(γ) M z
                # Implicit form: solve (AHA + λI) x_new = rhs where
                #   rhs = (AHA + λI)x + (γ/2)(lik + score) + sqrt(γ)·(AH(n1) + sqrt(λ)·n2)
                rhs = AHAx + lam * x                              # (A^H A + λI) x
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
