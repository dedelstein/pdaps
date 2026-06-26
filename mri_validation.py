"""Minimal single-slice MRI runner for P-DAPS and pULA.

This is intentionally not a sweep harness. It loads one fastMRI slice,
instantiates the InverseBench MRI prior and forward operator, runs one sampler,
and prints compact reconstruction/posterior metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
INVERSEBENCH = ROOT / "libs" / "inversebench"
if INVERSEBENCH.exists():
    sys.path.append(str(INVERSEBENCH))

from dataloader import MultiCoilMRIDataset
from metrics import compute_metrics, posterior_metrics
from pdaps import DAPS_TAU, PDAPS
from pula import PULA


MODEL_CONFIG = {
    "_target_": "models.precond.EDMPrecond",
    "model_type": "DhariwalUNet",
    "img_resolution": 320,
    "img_channels": 2,
    "label_dim": 0,
    "model_channels": 128,
    "channel_mult": [1, 1, 1, 2, 2],
    "attn_resolutions": [16],
    "num_blocks": 1,
    "dropout": 0.0,
}

PDAPS_ANNEALING = {
    "num_steps": 200,
    "sigma_max": 100,
    "sigma_min": 0.1,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

PDAPS_REVERSE_ODE = {
    "num_steps": 5,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

PULA_SCHEDULER = {
    "num_steps": 40,
    "sigma_max": 1,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "sqrt",
    "timestep": "log",
}


def load_model(models_dir, ckpt_name, image_size, device):
    if image_size[0] != image_size[1]:
        raise ValueError("MODEL_CONFIG assumes a square diffusion-model resolution")
    model_config = {**MODEL_CONFIG, "img_resolution": int(image_size[0])}
    net = hydra.utils.instantiate(OmegaConf.create(model_config))
    ckpt = torch.load(Path(models_dir) / ckpt_name, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["ema"])
    return net.to(device).eval().requires_grad_(False)


def make_forward_op(args, device):
    cfg = {
        "_target_": "inverse_problems.multi_coil_mri.MultiCoilMRI",
        "total_lines": args.image_size[1],
        "acceleration_ratio": args.acceleration,
        "pattern": args.pattern,
        "mask_seed": args.mask_seed,
        "device": str(device),
    }
    return hydra.utils.instantiate(OmegaConf.create(cfg))


def samples_by_file(dataset):
    by_file = {}
    for idx, (kspace_path, _maps_path, _slice_idx) in enumerate(dataset.samples):
        by_file.setdefault(Path(kspace_path).name, []).append(idx)
    return by_file


def get_sample(dataset, filename, slice_rank):
    by_file = samples_by_file(dataset)
    if filename not in by_file:
        raise ValueError(f"{filename!r} is not present in the indexed dataset")
    indices = by_file[filename]
    if not 0 <= slice_rank < len(indices):
        raise ValueError(f"slice_rank={slice_rank} outside 0..{len(indices) - 1} for {filename}")
    return dataset[indices[slice_rank]], indices[slice_rank]


def move_sample_to_device(sample, device):
    return {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def make_sampler(args, net, forward_op):
    if args.method == "pdaps":
        return PDAPS(
            net=net,
            forward_op=forward_op,
            annealing_scheduler_config={**PDAPS_ANNEALING, "num_steps": args.pdaps_steps},
            diffusion_scheduler_config={**PDAPS_REVERSE_ODE, "num_steps": args.ode_steps},
            tau=args.tau,
            null_blend=args.null_blend,
            split_cg_iter=args.split_cg_iter,
            inner_sigma_max=args.inner_sigma_max,
            log_level=args.log_level,
        )
    if args.method == "pula":
        return PULA(
            net=net,
            forward_op=forward_op,
            noise_scheduler_config={**PULA_SCHEDULER, "num_steps": args.pula_steps},
            K=args.pula_K,
            gamma=args.pula_gamma,
            cg_iter=args.pula_cg_iter,
            log_level=args.log_level,
        )
    raise ValueError(f"unknown method {args.method!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal single-slice MRI reconstruction.")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--ckpt-name", default="MRI-knee.pt")
    parser.add_argument("--kspace-dir", required=True)
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--slice-rank", type=int, default=0, help="Rank among indexed non-edge slices for filename.")
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--slice-range", nargs=2, type=int, default=[5, -5])
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--pattern", default="random", choices=("random", "equispaced"))
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--method", choices=("pdaps", "pula"), default="pdaps")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Optional .pt path for recon tensor(s).")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARN", "VAL"), default="INFO")

    parser.add_argument("--tau", type=float, default=DAPS_TAU)
    parser.add_argument("--null-blend", type=float, default=1.0)
    parser.add_argument("--inner-sigma-max", type=float, default=2.0)
    parser.add_argument("--split-cg-iter", type=int, default=30)
    parser.add_argument("--pdaps-steps", type=int, default=200)
    parser.add_argument("--ode-steps", type=int, default=5)

    parser.add_argument("--pula-steps", type=int, default=40)
    parser.add_argument("--pula-K", type=int, default=4)
    parser.add_argument("--pula-gamma", type=float, default=0.5)
    parser.add_argument("--pula-cg-iter", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    net = load_model(args.models_dir, args.ckpt_name, args.image_size, device)
    forward_op = make_forward_op(args, device)
    dataset = MultiCoilMRIDataset(
        args.kspace_dir,
        args.maps_dir,
        args.image_size,
        slice_range=tuple(args.slice_range),
        filenames=[args.filename],
    )
    sample, sample_idx = get_sample(dataset, args.filename, args.slice_rank)
    data = move_sample_to_device(sample, device)
    observation = forward_op(data)
    target = data["target"]

    sampler = make_sampler(args, net, forward_op)
    recon = sampler.inference(observation, num_samples=args.num_samples, verbose=True)

    if args.num_samples > 1:
        scores = posterior_metrics(forward_op, recon, target, observation)
    else:
        scores = compute_metrics(forward_op, recon, target, observation)
    scores["sample_idx"] = int(sample_idx)
    scores["filename"] = args.filename
    scores["slice_rank"] = int(args.slice_rank)
    scores["method"] = args.method
    scores["acceleration"] = int(args.acceleration)

    print(json.dumps(scores, indent=2, sort_keys=True))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(recon.detach().cpu(), out)
    if args.metrics_json:
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
