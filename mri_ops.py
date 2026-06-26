"""Shared linear algebra helpers for Cartesian multicoil MRI samplers."""

import torch


def _batch_inner(x, y):
    inner = (x.conj() * y).real
    if inner.ndim <= 1:
        return inner.sum()
    return inner.flatten(start_dim=1).sum(dim=1)


def _batch_view(x, ref):
    if x.ndim == 0:
        return x
    return x.view((x.shape[0],) + (1,) * (ref.ndim - 1))


def conjgrad(
    normal_op,
    b,
    x,
    lam,
    max_iter,
    tol=1e-10,
    x_is_zero=False,
    return_iters=False,
    penalty_op=None,
):
    """Solve (normal_op + lam * penalty_op) x = b by conjugate gradient."""
    if penalty_op is None:
        penalty_op = lambda v: v

    r = b.clone() if x_is_zero else b - normal_op(x) - lam * penalty_op(x)
    p = r.clone()
    rs_old = _batch_inner(r, r)

    num_iters = 0
    for num_iters in range(1, max_iter + 1):
        Ap = normal_op(p) + lam * penalty_op(p)
        pAp = _batch_inner(p, Ap)
        active = pAp.abs() >= 1e-30
        if not active.any():
            break
        denom = torch.where(active, pAp, torch.ones_like(pAp))
        alpha = torch.where(active, rs_old / denom, torch.zeros_like(pAp))
        alpha_view = _batch_view(alpha, p)
        x = x + alpha_view * p
        r = r - alpha_view * Ap
        rs_new = _batch_inner(r, r)
        if (rs_new.sqrt() < tol).all():
            break
        beta = rs_new / rs_old.clamp_min(1e-30)
        p = r + _batch_view(beta, p) * p
        rs_old = rs_new

    if return_iters:
        return x, num_iters
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


def mri_A(algo, x):
    """Forward multicoil MRI operator: complex image -> masked complex k-space."""
    even_shape, maps_shift, _, mask_unshift = _mri_fft_cache(algo)
    if not even_shape:
        coils = algo.forward_op.maps * x.unsqueeze(1)
        return algo.forward_op.mask * algo.forward_op.fft(coils)

    x_shift = torch.fft.fftshift(x, dim=(-2, -1))
    kspace_unshift = torch.fft.fft2(maps_shift * x_shift.unsqueeze(1), dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(mask_unshift * kspace_unshift, dim=(-2, -1))


def mri_AH(algo, y):
    """Adjoint multicoil MRI operator: masked complex k-space -> complex image."""
    even_shape, maps_shift, maps_shift_conj, mask_unshift = _mri_fft_cache(algo)
    if not even_shape:
        return (algo.forward_op.ifft(algo.forward_op.mask * y) * algo.forward_op.maps.conj()).sum(dim=1)

    y_unshift = torch.fft.ifftshift(y, dim=(-2, -1))
    y_unshift = mask_unshift * y_unshift
    image_shift = torch.fft.ifft2(y_unshift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift((image_shift * maps_shift_conj).sum(dim=1), dim=(-2, -1))


def mri_AHA(algo, x):
    """Normal multicoil MRI operator: A^H A x."""
    even_shape, maps_shift, maps_shift_conj, mask_unshift = _mri_fft_cache(algo)
    if not even_shape:
        return mri_AH(algo, mri_A(algo, x))

    x_shift = torch.fft.fftshift(x, dim=(-2, -1))
    kspace_unshift = torch.fft.fft2(maps_shift * x_shift.unsqueeze(1), dim=(-2, -1), norm="ortho")
    image_shift = torch.fft.ifft2(mask_unshift * kspace_unshift, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift((image_shift * maps_shift_conj).sum(dim=1), dim=(-2, -1))


def to_complex(x_real):
    """Convert (B,2,H,W) real-channel tensors to (B,H,W) complex tensors."""
    return torch.complex(x_real[:, 0], x_real[:, 1])


def to_real(x_complex):
    """Convert (B,H,W) complex tensors to (B,2,H,W) real-channel tensors."""
    return torch.stack([x_complex.real, x_complex.imag], dim=1)

