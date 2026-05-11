import torch
import matplotlib.pyplot as plt
import numpy as np
from libs.inversebench.training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss


def complex_to_chan(x):
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


def compute_ssim_nmse(reconstruction, target):
    dr_ssim = DynamicRangeSSIMLoss()
    with torch.no_grad():
        reconstruction = complex_to_chan(reconstruction)
        target = complex_to_chan(target)
        if reconstruction.ndim == 3:
            reconstruction = reconstruction.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        ssim_val = 1 - dr_ssim(reconstruction.detach().cpu(), target.detach().cpu()).item()
        nmse_val = (
            torch.linalg.norm(reconstruction.detach() - target.detach()).square()
            / torch.linalg.norm(target.detach()).square().clamp_min(1e-12)
        ).item()
    return ssim_val, nmse_val


def compute_metrics_dict(forward_op, reconstruction, target, observation):
    dr_psnr = DynamicRangePSNRLoss()
    dr_ssim = DynamicRangeSSIMLoss()

    with torch.no_grad():
        measurement_residual = forward_op.forward(reconstruction) - observation
        data_misfit = torch.linalg.norm(measurement_residual).item()
        observed = measurement_residual.numel()
        if hasattr(forward_op, "mask"):
            try:
                complex_obs = torch.view_as_complex(observation)
                observed = int(forward_op.mask.expand_as(complex_obs[:1]).sum().item()) * 2
            except RuntimeError:
                observed = measurement_residual.numel()
        data_misfit_per_observed = data_misfit / max(1.0, float(observed) ** 0.5)

        final_recon = forward_op.unnormalize(reconstruction).detach().cpu()
        final_target = forward_op.unnormalize(target).detach().cpu()
        psnr_val = -dr_psnr(final_recon, final_target).item()
        ssim_val = 1 - dr_ssim(final_recon, final_target).item()
        nmse = (
            torch.linalg.norm(final_recon - final_target).square()
            / torch.linalg.norm(final_target).square().clamp_min(1e-12)
        ).item()
        finite = bool(torch.isfinite(reconstruction).all().item())
        max_abs = float(reconstruction.detach().abs().max().item())

    return {
        "data_misfit": data_misfit,
        "data_misfit_per_observed": data_misfit_per_observed,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "nmse": nmse,
        "finite": finite,
        "max_abs": max_abs,
    }


def compute_metrics(forward_op, reconstruction, target, observation):
    metrics = compute_metrics_dict(forward_op, reconstruction, target, observation)
    print("\nMetrics:")

    print(f"  Data Misfit: {metrics['data_misfit']:.4f}")
    print(f"  PSNR:        {metrics['psnr']:.2f} dB")
    print(f"  SSIM:        {metrics['ssim']:.4f}")
    return metrics


def visualize_recon(forward_op, final_recon, final_target, batch_idx, cfg, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    target_img = magnitude_image(final_target).squeeze().cpu()
    axes[0].imshow(target_img.numpy(), cmap="gray")
    axes[0].set_title("Ground Truth (MVUE)")
    axes[0].axis("off")

    # Visualize the 1D undersampling mask
    mask_np = forward_op.mask.squeeze().cpu().numpy()
    mask_2d = np.tile(mask_np, (len(mask_np), 1))
    axes[1].imshow(mask_2d, cmap="gray")
    axes[1].set_title(f"Mask (R={cfg.forward_op.acceleration_ratio}x)")
    axes[1].axis("off")

    recon_img = magnitude_image(final_recon).squeeze().cpu()
    axes[2].imshow(recon_img.numpy(), cmap="gray")
    axes[2].set_title(f"{cfg.algorithm._target_.split('.')[-1]} Reconstruction")
    axes[2].axis("off")

    denom = target_img.abs().max().clamp_min(1e-12)
    error_img = (recon_img - target_img).abs() / denom
    im = axes[3].imshow(error_img.numpy(), cmap="magma")
    axes[3].set_title("|Error| / max(|Target|)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is None:
        save_path = f"{cfg.algorithm._target_.split('.')[-1]}_recon_batch_{batch_idx}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
