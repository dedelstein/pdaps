import hydra
import torch
import sys
import os

from utilities import compute_metrics, visualize_recon

# inversebench modules (forward_op, algorithm) are resolved by hydra via _target_
# but Python needs them on sys.path for the imports to work
sys.path.append(os.path.abspath("./libs/inversebench"))

from omegaconf import DictConfig
from hydra.utils import instantiate

from dataloader import MultiCoilMRIDataset


@hydra.main(version_base=None, config_path="configs", config_name="daps_config")
def main(cfg: DictConfig):
    if cfg.get("mode", "single") == "validation":
        from mri_validation import run_from_hydra

        run_from_hydra(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained diffusion model
    # The model is an EDMPrecond wrapper around a DhariwalUNet.
    # It takes 2-channel (real, imag) images of shape (batch, 2, 320, 320)
    # and a noise level sigma, and predicts the denoised image.
    ckpt_path = f"{cfg.paths.models_dir}/{cfg.pretrain.ckpt_name}"
    net = instantiate(cfg.pretrain.model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["ema"])
    net = net.to(device).eval()

    # Load dataset
    # MultiCoilMRIDataset reads multicoil k-space + ESPIRiT maps,
    # crops to image_size, computes MVUE target, and normalizes.
    # Each sample is a dict with 'target', 'kspace', 'maps'
    dataset = MultiCoilMRIDataset(
        kspace_dir=cfg.dataset.kspace_dir,
        maps_dir=cfg.dataset.maps_dir,
        image_size=cfg.dataset.image_size,
        filenames=cfg.dataset.get("filenames", None),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    # Instantiate forward operator and algorithm
    forward_op = instantiate(cfg.forward_op, device=device)
    algo = instantiate(cfg.algorithm, forward_op=forward_op, net=net)

    # Run reconstruction loop
    for batch_idx, data in enumerate(dataloader):
        # Move all tensors to GPU
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                data[key] = val.to(device)

        if batch_idx == 0:
            # Debug prints to verify data shapes, types, and check for NaNs/Infs
            print(f"\n--- Batch {batch_idx} ---")
            print(
                f"  kspace: {data['kspace'].shape}, dtype={data['kspace'].dtype}, device={data['kspace'].device}"
            )
            print(
                f"  maps:   {data['maps'].shape}, dtype={data['maps'].dtype}, device={data['maps'].device}"
            )
            print(
                f"  target: {data['target'].shape}, dtype={data['target'].dtype}, device={data['target'].device}"
            )
            print(
                f"  kspace has NaN: {torch.isnan(torch.view_as_real(data['kspace'])).any()}, Inf: {torch.isinf(torch.view_as_real(data['kspace'])).any()}"
            )
            print(
                f"  maps has NaN: {torch.isnan(torch.view_as_real(data['maps'])).any()}, Inf: {torch.isinf(torch.view_as_real(data['maps'])).any()}"
            )
            print(
                f"  target has NaN: {torch.isnan(data['target']).any()}, Inf: {torch.isinf(data['target']).any()}"
            )
            print(
                f"  mask: {forward_op.mask.shape}, dtype={forward_op.mask.dtype}, device={forward_op.mask.device}"
            )

        observation = forward_op(data)
        target = data["target"]

        if batch_idx == 0:
            print(f"  observation: {observation.shape}, dtype={observation.dtype}")
            print(
                f"  observation has NaN: {torch.isnan(observation).any()}, Inf: {torch.isinf(observation).any()}"
            )
            print(
                f"  forward_op.maps: {forward_op.maps.shape}, dtype={forward_op.maps.dtype}"
            )

        # Inference: iteratively denoise while enforcing data consistency
        print("Running inference...")
        reconstruction = algo.inference(observation, num_samples=1)

        compute_metrics(forward_op, reconstruction, target, observation)
        visualize_recon(
            forward_op,
            forward_op.unnormalize(reconstruction).cpu(),
            forward_op.unnormalize(target).cpu(),
            batch_idx,
            cfg,
        )

        break  # just one batch for proof of concept


if __name__ == "__main__":
    main()
