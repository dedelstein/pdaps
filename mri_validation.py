import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "libs/inversebench"))


from dataloader import MultiCoilMRIDataset
from utilities import compute_metrics_dict, visualize_recon


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

ANNEALING = {
    "num_steps": 200,
    "sigma_max": 100,
    "sigma_min": 0.1,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

REVERSE_ODE = {
    "num_steps": 5,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}

DPS_SCHEDULER = {
    "num_steps": 1000,
    "schedule": "vp",
    "timestep": "vp",
    "scaling": "vp",
}

PULA_SCHEDULER = {
    "num_steps": 200,
    "sigma_max": 10,
    "sigma_min": 0.01,
    "sigma_final": 0,
    "schedule": "linear",
    "timestep": "poly-7",
}


def grid(points):
    keys = list(points)
    for values in itertools.product(*(points[key] for key in keys)):
        yield dict(zip(keys, values))


def method_grid(preset="tiny"):
    if preset == "smoke":
        dps_scales = [1.0]
        daps_lrs = [1e-5]
        pula_gammas = [0.5]
        pdaps_gammas = [0.5]
        warm_fractions = [0.2]
    elif preset == "tiny":
        dps_scales = [0.5, 1.0, 2.0]
        daps_lrs = [3e-6, 1e-5, 3e-5]
        pula_gammas = [0.25, 0.5, 1.0]
        pdaps_gammas = [0.25, 0.5]
        warm_fractions = [0.1, 0.2]
    elif preset == "full":
        dps_scales = [0.5, 1.0, 2.0]
        daps_lrs = [3e-6, 1e-5, 3e-5]
        pula_gammas = [0.25, 0.5, 1.0]
        pdaps_gammas = [0.25, 0.5, 1.0]
        warm_fractions = [0.1, 0.2, 0.4]
    else:
        raise ValueError(f"Unknown grid preset: {preset}")

    methods = []
    for p in grid({"guidance_scale": dps_scales}):
        methods.append({
            "method": "DPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.dps.DPS",
                "diffusion_scheduler_config": DPS_SCHEDULER,
                "guidance_scale": p["guidance_scale"],
            },
        })

    for p in grid({"lr": daps_lrs}):
        methods.append({
            "method": "DAPS",
            "params": p,
            "algorithm": {
                "_target_": "algo.daps.DAPS",
                "annealing_scheduler_config": ANNEALING,
                "diffusion_scheduler_config": REVERSE_ODE,
                "lgvd_config": {"num_steps": 100, "lr": p["lr"], "tau": 0.002028752174814177, "lr_min_ratio": 0.01},
            },
        })

    for p in grid({"gamma": pula_gammas}):
        methods.append({
            "method": "pULA",
            "params": p,
            "algorithm": {
                "_target_": "algo.pula.pULA",
                "noise_scheduler_config": PULA_SCHEDULER,
                "K": 4,
                "gamma": p["gamma"],
                "cg_iter": 10,
            },
        })

    for p in grid({"gamma": pdaps_gammas}):
        methods.append(pdaps_entry("P-DAPS", "none", p["gamma"], 0.0))

    for p in grid({"gamma": pdaps_gammas, "warm_fraction": warm_fractions}):
        methods.append(pdaps_entry("P-DAPS-fixed", "fixed", p["gamma"], p["warm_fraction"]))
    return methods


def pdaps_entry(method, warm_mode, gamma, warm_fraction):
    return {
        "method": method,
        "params": {"gamma": gamma, "warm_fraction": warm_fraction},
        "algorithm": {
            "_target_": "algo.pdaps.PDAPS",
            "annealing_scheduler_config": ANNEALING,
            "diffusion_scheduler_config": REVERSE_ODE,
            "lgvd_config": {"num_steps": 25, "gamma": gamma, "cg_iter": 10, "lr_min_ratio": 0.01},
            "warm_mode": warm_mode,
            "warm_fraction": warm_fraction,
        },
    }


def load_model(args, device):
    net = hydra.utils.instantiate(OmegaConf.create(MODEL_CONFIG))
    ckpt = torch.load(Path(args.models_dir) / args.ckpt_name, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["ema"])
    return net.to(device).eval()


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


def move_to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def run_one(entry, sample, sample_idx, split, net, args, out_dir, save_image=False):
    device = next(net.parameters()).device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    forward_op = make_forward_op(args, device)
    algo = hydra.utils.instantiate(OmegaConf.create(entry["algorithm"]), forward_op=forward_op, net=net)
    data = move_to_device(sample, device)
    data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    observation = forward_op(data)
    target = data["target"]

    row = {
        "split": split,
        "sample_idx": sample_idx,
        "method": entry["method"],
        "params_json": json.dumps(entry["params"], sort_keys=True),
        "failed": False,
    }

    start = time.perf_counter()
    try:
        recon = algo.inference(observation, num_samples=1, verbose=args.verbose)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row.update(compute_metrics_dict(forward_op, recon, target, observation))
        row["runtime_s"] = time.perf_counter() - start
        row["gate_stats_json"] = json.dumps(getattr(algo, "last_gate_stats", []))
        if save_image:
            cfg = OmegaConf.create({
                "algorithm": {"_target_": entry["algorithm"]["_target_"]},
                "forward_op": {"acceleration_ratio": args.acceleration},
            })
            image_dir = out_dir / "figures" / entry["method"].replace("/", "_")
            image_dir.mkdir(parents=True, exist_ok=True)
            old_cwd = os.getcwd()
            os.chdir(image_dir)
            try:
                visualize_recon(forward_op, forward_op.unnormalize(recon).cpu(), forward_op.unnormalize(target).cpu(), sample_idx, cfg)
            finally:
                os.chdir(old_cwd)
    except Exception as exc:
        row["failed"] = True
        row["error"] = repr(exc)
        row["runtime_s"] = time.perf_counter() - start
    return row


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["params_json"])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, params_json), group in groups.items():
        ok = [row for row in group if not row.get("failed")]
        summary = {
            "method": method,
            "params_json": params_json,
            "n": len(group),
            "n_ok": len(ok),
            "failure_rate": 1.0 - len(ok) / max(1, len(group)),
        }
        for metric in ("psnr", "ssim", "nmse", "data_misfit", "data_misfit_per_observed", "runtime_s"):
            vals = [float(row[metric]) for row in ok if metric in row]
            if vals:
                summary[f"{metric}_mean"] = sum(vals) / len(vals)
                summary[f"{metric}_std"] = torch.tensor(vals).std(unbiased=False).item()
        out.append(summary)
    return sorted(out, key=lambda row: (row["method"], -row.get("psnr_mean", -1e9)))


def select_best(validation_summary):
    best = {}
    for row in validation_summary:
        if row["n_ok"] == 0:
            continue
        current = best.get(row["method"])
        candidate = (row.get("psnr_mean", -1e9), row.get("ssim_mean", -1e9), -row.get("data_misfit_mean", 1e9))
        if current is None or candidate > current[0]:
            best[row["method"]] = (candidate, row["params_json"])
    return {method: params_json for method, (_, params_json) in best.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny multi-coil knee MRI validation for DPS/DAPS/pULA/P-DAPS.")
    parser.add_argument("--models-dir", default="/dtu/blackhole/1d/214141/Thesis/models")
    parser.add_argument("--ckpt-name", default="MRI-knee.pt")
    parser.add_argument("--kspace-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val")
    parser.add_argument("--maps-dir", default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit")
    parser.add_argument("--filename", default="file1000196.h5")
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--val-slices", type=int, default=2)
    parser.add_argument("--test-slices", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--grid-preset", choices=("smoke", "tiny", "full"), default="tiny")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-grid", action="store_true")
    return parser.parse_args()


def run_validation(args):
    entries = method_grid(args.grid_preset)
    methods_filter = getattr(args, "methods", None)
    if methods_filter:
        wanted = {m.strip() for m in methods_filter.split(",") if m.strip()}
        entries = [e for e in entries if e["method"] in wanted]
    if args.list_grid:
        print(json.dumps(entries, indent=2))
        return None

    out_dir = Path(args.out_dir or f"results/mri_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not out_dir.is_absolute():
        try:
            out_dir = Path(get_original_cwd()) / out_dir
        except ValueError:
            out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(entries, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args, device)
    dataset = MultiCoilMRIDataset(args.kspace_dir, args.maps_dir, args.image_size, filenames=[args.filename])

    n = args.val_slices + args.test_slices
    indices = list(range(args.slice_offset, args.slice_offset + n))
    samples = [(idx, dataset[idx]) for idx in indices]
    val_samples = samples[:args.val_slices]
    test_samples = samples[args.val_slices:]

    val_rows = []
    for entry in entries:
        for idx, sample in val_samples:
            val_rows.append(run_one(entry, sample, idx, "validation", net, args, out_dir))
            write_csv(out_dir / "validation_raw.csv", val_rows)
    val_summary = summarize(val_rows)
    write_csv(out_dir / "validation_summary.csv", val_summary)

    selected = select_best(val_summary)
    selected_entries = [entry for entry in entries if selected.get(entry["method"]) == json.dumps(entry["params"], sort_keys=True)]
    with open(out_dir / "selected.json", "w") as f:
        json.dump(selected, f, indent=2)

    test_rows = []
    for entry in selected_entries:
        for idx, sample in test_samples:
            test_rows.append(run_one(entry, sample, idx, "test", net, args, out_dir, save_image=True))
            write_csv(out_dir / "test_raw.csv", test_rows)
    write_csv(out_dir / "test_summary.csv", summarize(test_rows))
    print(f"Wrote {out_dir}")
    return out_dir


def run_from_hydra(cfg):
    validation = cfg.get("validation", {})
    args = argparse.Namespace(
        models_dir=cfg.paths.models_dir,
        ckpt_name=cfg.pretrain.ckpt_name,
        kspace_dir=cfg.dataset.kspace_dir,
        maps_dir=cfg.dataset.maps_dir,
        filename=validation.get("filename", cfg.dataset.get("filenames", ["file1000196.h5"])[0]),
        image_size=list(cfg.dataset.image_size),
        acceleration=int(cfg.forward_op.acceleration_ratio),
        pattern=cfg.forward_op.get("pattern", "random"),
        mask_seed=int(cfg.forward_op.get("mask_seed", 0)),
        seed=int(validation.get("seed", 123)),
        slice_offset=int(validation.get("slice_offset", 0)),
        val_slices=int(validation.get("val_slices", 2)),
        test_slices=int(validation.get("test_slices", 3)),
        out_dir=validation.get("out_dir", None),
        verbose=bool(validation.get("verbose", False)),
        list_grid=bool(validation.get("list_grid", False)),
        grid_preset=validation.get("grid_preset", "tiny"),
        methods=validation.get("methods", None),
    )
    return run_validation(args)


def main():
    run_validation(parse_args())


if __name__ == "__main__":
    main()
