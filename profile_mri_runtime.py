import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "libs/inversebench"))

from dataloader import MultiCoilMRIDataset
from mri_validation import (
    load_model,
    make_forward_op,
    method_grid,
    move_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight profiler for one multi-coil MRI reconstruction run."
    )
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
    parser.add_argument("--grid-preset", default="smoke")
    parser.add_argument("--method", default="P-DAPS-fixed")
    parser.add_argument("--entry-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=None)
    parser.add_argument("--reverse-steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--pula-k", type=int, default=None)
    parser.add_argument("--cg-iter", type=int, default=None)
    parser.add_argument("--no-fast-aha", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compile-net", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--compile-warmup",
        type=int,
        default=0,
        help="Run this many dummy net forwards after torch.compile and before profiling.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=None,
        help="Forwarded to torch.set_float32_matmul_precision. 'high' enables TF32-style speedups on Ampere+.",
    )
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--profiler-rows", type=int, default=25)
    parser.add_argument("--trace-path", default=None)
    parser.add_argument(
        "--sync-method-timing",
        action="store_true",
        help="Synchronize around wrapped method calls for more accurate timings. Adds overhead.",
    )
    return parser.parse_args()


def pick_entry(entries, method, entry_index):
    matches = [entry for entry in entries if entry["method"] == method]
    if not matches:
        names = sorted({entry["method"] for entry in entries})
        raise ValueError(f"No method {method!r} in grid. Available methods: {names}")
    if entry_index < 0 or entry_index >= len(matches):
        raise IndexError(f"--entry-index must be in [0, {len(matches) - 1}] for {method}")
    return copy.deepcopy(matches[entry_index])


def override_steps(entry, args):
    algorithm = entry["algorithm"]
    if args.outer_steps is not None:
        if "annealing_scheduler_config" in algorithm:
            algorithm["annealing_scheduler_config"]["num_steps"] = args.outer_steps
        if "noise_scheduler_config" in algorithm:
            algorithm["noise_scheduler_config"]["num_steps"] = args.outer_steps
    if args.reverse_steps is not None and "diffusion_scheduler_config" in algorithm:
        algorithm["diffusion_scheduler_config"]["num_steps"] = args.reverse_steps
    if args.inner_steps is not None and "lgvd_config" in algorithm:
        algorithm["lgvd_config"]["num_steps"] = args.inner_steps
        entry["params"]["lgvd_num_steps"] = args.inner_steps
    if args.pula_k is not None and "K" in algorithm:
        algorithm["K"] = args.pula_k
    if args.cg_iter is not None:
        if "lgvd_config" in algorithm:
            algorithm["lgvd_config"]["cg_iter"] = args.cg_iter
        if "cg_iter" in algorithm:
            algorithm["cg_iter"] = args.cg_iter
    if args.no_fast_aha:
        algorithm["use_fast_aha"] = False
    return entry


def synchronize_if_needed(device, enabled):
    if enabled and device.type == "cuda":
        torch.cuda.synchronize()


def wrap_method(obj, name, stats, sync_timing):
    if not hasattr(obj, name):
        return
    original = getattr(obj, name)
    if not callable(original):
        return

    def wrapped(*args, **kwargs):
        device = None
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        synchronize_if_needed(device or torch.device("cpu"), sync_timing)
        start = time.perf_counter()
        with torch.profiler.record_function(f"{obj.__class__.__name__}.{name}"):
            result = original(*args, **kwargs)
        synchronize_if_needed(device or torch.device("cpu"), sync_timing)
        stats[name]["count"] += 1
        stats[name]["wall_s"] += time.perf_counter() - start
        return result

    setattr(obj, name, wrapped)


def instrument_algo(algo, sync_timing):
    stats = defaultdict(lambda: {"count": 0, "wall_s": 0.0})
    for name in ("A", "AH", "AHA", "solve", "score", "init_sample", "init_inner", "adaptive_alpha"):
        wrap_method(algo, name, stats, sync_timing)
    if hasattr(algo, "inner"):
        for name in ("step", "sample"):
            wrap_method(algo.inner, name, stats, sync_timing)
    return stats


def print_method_stats(stats):
    rows = [
        (name, values["count"], values["wall_s"])
        for name, values in stats.items()
        if values["count"]
    ]
    if not rows:
        return
    print("\nWrapped method timings:")
    print(f"{'method':<16} {'calls':>10} {'wall_s':>12} {'ms/call':>12}")
    for name, count, wall_s in sorted(rows, key=lambda row: row[2], reverse=True):
        print(f"{name:<16} {count:>10d} {wall_s:>12.3f} {1000.0 * wall_s / count:>12.3f}")


def profiler_activities(device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def run_profile(args):
    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    entries = method_grid(args.grid_preset)
    entry = override_steps(pick_entry(entries, args.method, args.entry_index), args)
    print("Profiling entry:")
    print(json.dumps({"method": entry["method"], "params": entry["params"], "algorithm": entry["algorithm"]}, indent=2))

    net = load_model(args, device)
    if args.compile_net:
        net = torch.compile(net, mode=args.compile_mode)
        if args.compile_warmup > 0:
            x = torch.randn(
                args.num_samples,
                2,
                args.image_size[0],
                args.image_size[1],
                device=device,
            )
            sigma = torch.tensor(1.0, device=device)
            with torch.inference_mode():
                for _ in range(args.compile_warmup):
                    net(x, sigma)
            synchronize_if_needed(device, True)

    dataset = MultiCoilMRIDataset(args.kspace_dir, args.maps_dir, args.image_size, filenames=[args.filename])
    sample = dataset[args.slice_offset]
    data = move_to_device(sample, device)
    data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    forward_op = make_forward_op(args, device)
    observation = forward_op(data)
    algo = hydra.utils.instantiate(OmegaConf.create(entry["algorithm"]), forward_op=forward_op, net=net)
    stats = instrument_algo(algo, args.sync_method_timing)

    synchronize_if_needed(device, True)
    start = time.perf_counter()
    if args.torch_profiler:
        sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        with torch.profiler.profile(
            activities=profiler_activities(device),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            recon = algo.inference(observation, num_samples=args.num_samples, verbose=args.verbose)
            prof.step()
        synchronize_if_needed(device, True)
        print("\nTorch profiler:")
        print(prof.key_averages().table(sort_by=sort_by, row_limit=args.profiler_rows))
        if args.trace_path:
            prof.export_chrome_trace(args.trace_path)
            print(f"\nWrote trace: {args.trace_path}")
    else:
        recon = algo.inference(observation, num_samples=args.num_samples, verbose=args.verbose)
        synchronize_if_needed(device, True)
    elapsed = time.perf_counter() - start

    print_method_stats(stats)
    print(f"\nTotal inference wall time: {elapsed:.3f}s")
    print(f"Output: shape={tuple(recon.shape)} dtype={recon.dtype} device={recon.device}")


def main():
    run_profile(parse_args())


if __name__ == "__main__":
    main()
