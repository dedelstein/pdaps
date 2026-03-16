import torch
import matplotlib.pyplot as plt
import numpy as np
from libs.inversebench.training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss


def compute_metrics(forward_op, reconstruction, target, observation):
    dr_psnr = DynamicRangePSNRLoss()
    dr_ssim = DynamicRangeSSIMLoss()
    print("\nMetrics:")

    data_misfit = torch.linalg.norm(
        forward_op.forward(reconstruction) - observation
    ).item()

    final_recon = forward_op.unnormalize(reconstruction).cpu()
    final_target = forward_op.unnormalize(target).cpu()
    psnr_val = -dr_psnr(final_recon, final_target).item()
    ssim_val = 1 - dr_ssim(final_recon, final_target).item()

    print(f"  Data Misfit: {data_misfit:.4f}")
    print(f"  PSNR:        {psnr_val:.2f} dB")
    print(f"  SSIM:        {ssim_val:.4f}")


def visualize_recon(forward_op, final_recon, final_target, batch_idx, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    target_img = final_target.squeeze()
    if target_img.ndim == 3 and target_img.shape[0] == 2:
        target_img = (target_img[0]**2 + target_img[1]**2).sqrt()
    axes[0].imshow(target_img.numpy(), cmap="gray")
    axes[0].set_title("Ground Truth (MVUE)")
    axes[0].axis("off")

    # Visualize the 1D undersampling mask
    mask_np = forward_op.mask.squeeze().cpu().numpy()
    mask_2d = np.tile(mask_np, (len(mask_np), 1))
    axes[1].imshow(mask_2d, cmap="gray")
    axes[1].set_title(f"Mask (R={cfg.forward_op.acceleration_ratio}x)")
    axes[1].axis("off")

    recon_img = final_recon.detach().squeeze()
    if recon_img.ndim == 3 and recon_img.shape[0] == 2:
        recon_img = (recon_img[0]**2 + recon_img[1]**2).sqrt()
    axes[2].imshow(recon_img.numpy(), cmap="gray")
    axes[2].set_title(f"{cfg.algorithm._target_.split('.')[-1]} Reconstruction")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = f"{cfg.algorithm._target_.split('.')[-1]}_recon_batch_{batch_idx}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved {save_path}")