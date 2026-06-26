"""Small MRI reconstruction metrics used by the public examples."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss


HFEN_LOG_SIGMA = 1.5
LAPLACIAN_KERNEL = torch.tensor(
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def complex_to_channels(x):
    if torch.is_complex(x):
        return torch.stack([x.real, x.imag], dim=1)
    return x


def magnitude_image(x):
    x = x.detach()
    if torch.is_complex(x):
        return x.abs()
    x = x.squeeze()
    if x.ndim == 3 and x.shape[0] == 2:
        return (x[0] ** 2 + x[1] ** 2).sqrt()
    if x.ndim == 4 and x.shape[1] == 2:
        return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).abs()
    return x


def nmse(reconstruction, target):
    reconstruction = complex_to_channels(reconstruction)
    target = complex_to_channels(target)
    return (
        torch.linalg.norm(reconstruction.detach() - target.detach()).square()
        / torch.linalg.norm(target.detach()).square().clamp_min(1e-12)
    ).item()


def _hfen_single(recon_mag, target_mag, sigma=HFEN_LOG_SIGMA):
    recon_log = ndimage.gaussian_laplace(recon_mag, sigma=sigma)
    target_log = ndimage.gaussian_laplace(target_mag, sigma=sigma)
    denom = float(np.linalg.norm(target_log))
    if denom < 1e-12:
        return float("nan")
    return float(np.linalg.norm(recon_log - target_log) / denom)


def hfen(reconstruction, target):
    recon_mag = magnitude_image(reconstruction).detach().cpu().numpy()
    target_mag = magnitude_image(target).detach().cpu().numpy()
    recon_mag = recon_mag.reshape(-1, *recon_mag.shape[-2:])
    target_mag = target_mag.reshape(-1, *target_mag.shape[-2:])
    values = [_hfen_single(recon_mag[i], target_mag[i]) for i in range(recon_mag.shape[0])]
    return float(np.nanmean(values)) if values else float("nan")


def data_misfit(forward_op, reconstruction, observation):
    residual = forward_op.forward(reconstruction) - observation
    total = torch.linalg.norm(residual).item()
    observed = residual.numel()
    if hasattr(forward_op, "mask"):
        try:
            complex_observation = torch.view_as_complex(observation)
            observed = int(forward_op.mask.expand_as(complex_observation[:1]).sum().item()) * 2
        except RuntimeError:
            observed = residual.numel()
    return total, total / max(1.0, float(observed) ** 0.5)


def compute_metrics(forward_op, reconstruction, target, observation):
    """Compute the compact metric set used in the paper experiments."""
    psnr_loss = DynamicRangePSNRLoss()
    ssim_loss = DynamicRangeSSIMLoss()

    with torch.no_grad():
        final_recon = forward_op.unnormalize(reconstruction).detach().cpu()
        final_target = forward_op.unnormalize(target).detach().cpu()
        misfit, misfit_per_observed = data_misfit(forward_op, reconstruction, observation)

        return {
            "psnr": -psnr_loss(final_recon, final_target).item(),
            "ssim": 1 - ssim_loss(final_recon, final_target).item(),
            "nmse": nmse(final_recon, final_target),
            "hfen": hfen(final_recon, final_target),
            "data_misfit": misfit,
            "data_misfit_per_observed": misfit_per_observed,
            "finite": bool(torch.isfinite(reconstruction).all().item()),
            "max_abs": float(reconstruction.detach().abs().max().item()),
        }


def _as_recon_stack(reconstructions):
    if isinstance(reconstructions, (list, tuple)):
        reconstructions = torch.stack(list(reconstructions), dim=0)
    if reconstructions.ndim == 4 and reconstructions.shape[1] == 2:
        reconstructions = reconstructions.unsqueeze(1)
    if torch.is_complex(reconstructions) and reconstructions.ndim == 3:
        reconstructions = reconstructions.unsqueeze(1)
    return reconstructions


def high_frequency_energy(magnitude):
    image = magnitude.view(1, 1, *magnitude.shape).float()
    kernel = LAPLACIAN_KERNEL.to(device=image.device, dtype=image.dtype)
    lap = F.conv2d(image, kernel, padding=1)
    return (lap.pow(2).mean().sqrt() / magnitude.mean().clamp_min(1e-12)).item()


def posterior_diagnostics(reconstructions, target=None, foreground_threshold=0.1):
    """Compute diversity/calibration diagnostics from multiple posterior samples.

    Args:
        reconstructions: sequence or tensor of reconstructions. Expected layout is
            one sample dimension followed by the normal reconstruction shape, e.g.
            (S,1,2,H,W), (S,2,H,W), or complex (S,1,H,W).
        target: optional target image. Required for uncertainty-error correlation
            and z_rms.
        foreground_threshold: target-magnitude fraction used to ignore background
            pixels in target-aware diagnostics.
    """
    stack = _as_recon_stack(reconstructions)
    mags = magnitude_image(stack.flatten(0, 1)).detach().float().cpu()
    mags = mags.reshape(stack.shape[0], stack.shape[1], *mags.shape[-2:])

    mean_mag = mags.mean(dim=0)
    std_mag = mags.std(dim=0)
    diversity = (std_mag.mean() / mean_mag.mean().clamp_min(1e-12)).item()

    single_hf = []
    for sample_idx in range(mags.shape[0]):
        for batch_idx in range(mags.shape[1]):
            single_hf.append(high_frequency_energy(mags[sample_idx, batch_idx]))
    mean_hf = [high_frequency_energy(mean_mag[batch_idx]) for batch_idx in range(mean_mag.shape[0])]
    collapse = float(np.mean(mean_hf) / max(1e-12, float(np.mean(single_hf))))

    diagnostics = {
        "diversity": float(diversity),
        "collapse": collapse,
        "unc_err_corr": float("nan"),
        "z_rms": float("nan"),
    }

    if target is None:
        return diagnostics

    target_mag = magnitude_image(target).detach().float().cpu()
    target_mag = target_mag.reshape(mean_mag.shape[0], *target_mag.shape[-2:])
    error = (mean_mag - target_mag).abs()
    foreground = target_mag > foreground_threshold * target_mag.amax(dim=(-2, -1), keepdim=True)

    corr_values = []
    z_values = []
    for batch_idx in range(mean_mag.shape[0]):
        err = error[batch_idx][foreground[batch_idx]]
        std = std_mag[batch_idx][foreground[batch_idx]]
        if err.numel() > 16 and std.std() > 0 and err.std() > 0:
            corr_values.append(float(torch.corrcoef(torch.stack([std, err]))[0, 1]))
            z_values.append(float((err / std.clamp_min(1e-8)).pow(2).mean().sqrt()))

    if corr_values:
        diagnostics["unc_err_corr"] = float(np.mean(corr_values))
    if z_values:
        diagnostics["z_rms"] = float(np.mean(z_values))
    return diagnostics


def posterior_metrics(forward_op, reconstructions, target, observation):
    """Score posterior samples with mean metrics and diversity diagnostics."""
    stack = _as_recon_stack(reconstructions)
    mean_recon = stack.mean(dim=0)

    mean_scores = compute_metrics(forward_op, mean_recon, target, observation)
    sample_scores = [
        compute_metrics(forward_op, stack[i], target, observation)
        for i in range(stack.shape[0])
    ]
    averaged_sample_scores = {
        f"sample_{key}": float(np.mean([scores[key] for scores in sample_scores]))
        for key in ("psnr", "ssim", "nmse", "hfen", "data_misfit_per_observed")
    }
    mean_scores = {f"mean_{key}": value for key, value in mean_scores.items()}
    return {
        **mean_scores,
        **averaged_sample_scores,
        **posterior_diagnostics(stack, target=target),
        "num_samples": int(stack.shape[0]),
    }
