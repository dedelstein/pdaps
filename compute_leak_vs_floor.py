"""Compare mask-null coil leakage against the model-mismatch floor.

Both quantities are measurement-space L2 residual norms divided by sqrt(m), where
m is the number of observed real coordinates. Their ratio is therefore
normalization-free.

Definitions (per slice, per acceleration):

    floor  = ||A(x_target) - y|| / sqrt(m)
    leak   = ||A(N x_target)|| / sqrt(m)
    ratio  = leak / floor

where N x = x - ifft(mask * fft(x)) is the Fourier mask-null projector and A is
the true multicoil forward operator.

CPU-only; no model checkpoint. Mirrors compute_tau_floor.py's normalization
so the floor column matches.

    ./.venv/bin/python3 compute_leak_vs_floor.py --accelerations 4 8 \\
        --filenames file1000196.h5 --out results/leak_vs_floor.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "libs/inversebench"))

EPS = 1e-30


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kspace-dir",
        default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val",
    )
    parser.add_argument(
        "--maps-dir",
        default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=None,
        help="Files to evaluate; defaults to every file in --kspace-dir.",
    )
    parser.add_argument("--image-size", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--accelerations", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--pattern", default="random")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument(
        "--val-slices",
        type=int,
        default=2,
        help="Slices per file, matching mri_validation's val split.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--bundle",
        nargs="+",
        default=None,
        help=(
            "Geometry-diagnostic .pt bundle(s). Switches to during-run "
            "exact-projector vs mask-split residual-shift analysis."
        ),
    )
    return parser.parse_args()


def residual_shift_from_bundle(path):
    """Measure residual shift from a saved geometry-diagnostic bundle."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    diagnostic = bundle["diagnostic"]
    y = torch.view_as_complex(bundle["observation"].contiguous())
    y_norm = torch.linalg.norm(y).item()

    a_range = diagnostic["A_range_part"]
    a_null = diagnostic["A_null_part"]
    a_clean = diagnostic["A_x_clean"]
    split_err = torch.linalg.norm(a_clean - (a_range + a_null)).item() / max(
        torch.linalg.norm(a_clean).item(),
        EPS,
    )

    residual_exact = a_range - y
    residual_mask = a_clean - y
    residual_exact_norm = torch.linalg.norm(residual_exact).item()
    residual_mask_norm = torch.linalg.norm(residual_mask).item()
    leak_norm = torch.linalg.norm(a_null).item()
    shift = residual_mask_norm - residual_exact_norm

    cross = torch.real(torch.vdot(residual_exact.flatten(), a_null.flatten())).item()
    cos = cross / max(residual_exact_norm * leak_norm, EPS)

    return {
        "bundle": str(path),
        "filename": bundle["meta"]["filename"],
        "acceleration": bundle["meta"]["acceleration"],
        "null_blend": bundle["meta"]["pdaps_v3"]["null_blend"],
        "outer_step": diagnostic["outer"],
        "sigma": diagnostic["sigma"],
        "y_norm": y_norm,
        "split_check_rel": split_err,
        "leak_frac_y": leak_norm / y_norm,
        "resid_exact_frac_y": residual_exact_norm / y_norm,
        "resid_mask_frac_y": residual_mask_norm / y_norm,
        "residual_shift_frac_y": shift / y_norm,
        "decomp_2Re_cross": 2 * cross,
        "decomp_leak_sq": leak_norm**2,
        "cos_leak_residual": cos,
    }


def run_bundle_mode(bundle_paths, out_path):
    print("During-run residual-shift analysis (exact projector vs mask split).\n")
    rows = [residual_shift_from_bundle(p) for p in bundle_paths]
    for r in rows:
        print(
            f"{r['filename']} accel={r['acceleration']} "
            f"blend={r['null_blend']} step={r['outer_step']} "
            f"sigma={r['sigma']:.3f}"
        )
        print(
            f"  leak ||A N null|| = {r['leak_frac_y'] * 100:.3f}% of ||y||   "
            f"resid: exact={r['resid_exact_frac_y'] * 100:.2f}%  "
            f"mask={r['resid_mask_frac_y'] * 100:.2f}% of ||y||"
        )
        print(
            f"  residual shift = {r['residual_shift_frac_y'] * 100:.4f}% of ||y||"
        )
        print(
            f"  trace: 2Re<r,leak>={r['decomp_2Re_cross']:.2f}, "
            f"||leak||^2={r['decomp_leak_sq']:.2f}, "
            f"cos(leak,resid)={r['cos_leak_residual']:.4f}"
        )
        print(
            "  split check A x_clean = A range + A null: "
            f"rel err {r['split_check_rel']:.1e}\n"
        )
    if out_path:
        Path(out_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


def select_val_samples(dataset, slice_offset, val_slices):
    by_file = {}
    for global_idx, (kspace_path, _maps_path, _slice) in enumerate(dataset.samples):
        by_file.setdefault(kspace_path.name, []).append(global_idx)
    selected = []
    for filename, global_indices in by_file.items():
        sample_indices = global_indices[slice_offset : slice_offset + val_slices]
        for rank, global_idx in enumerate(sample_indices):
            selected.append((global_idx, filename, slice_offset + rank))
    return selected


def project_range(forward_op, x_complex):
    """Fourier mask range projector R x = ifft(mask * fft(x)); pdaps_v4:180-182."""
    mask = forward_op.mask[0].to(device=x_complex.device)
    return forward_op.ifft(mask * forward_op.fft(x_complex))


def to_complex_image(image):
    """(B,2,H,W) real -> (B,H,W) complex, matching MultiCoilMRI.forward."""
    return torch.view_as_complex(image.permute(0, 2, 3, 1).contiguous())


def to_real_image(x_complex):
    """(B,H,W) complex -> (B,2,H,W) real, inverse of to_complex_image."""
    return torch.view_as_real(x_complex).permute(0, 3, 1, 2).contiguous()


def leak_and_floor_per_observed(forward_op, data):
    observation = forward_op(data)
    target = data["target"]
    observed_real = (
        int(forward_op.mask.expand_as(torch.view_as_complex(observation)[:1]).sum().item())
        * 2
    )
    denom = math.sqrt(max(1.0, float(observed_real)))
    y_norm = torch.linalg.norm(observation).item()

    floor_abs = torch.linalg.norm(forward_op.forward(target) - observation).item()

    x_c = to_complex_image(target).to(torch.complex128)
    null_c = x_c - project_range(forward_op, x_c)
    null_image = to_real_image(null_c).to(target.dtype)
    leak_abs = torch.linalg.norm(forward_op.forward(null_image)).item()

    a_target_masked = torch.linalg.norm(forward_op.forward(target)).item()
    coils = data["maps"] * x_c.unsqueeze(1)
    a_target_full = torch.linalg.norm(forward_op.fft(coils)).item()

    return {
        "leak": leak_abs / denom,
        "floor": floor_abs / denom,
        "shift_frac_y": leak_abs / y_norm,
        "floor_frac_y": floor_abs / y_norm,
        "shift_frac_Atarget": leak_abs / a_target_masked,
        "shift_frac_Afull": leak_abs / a_target_full,
        "observed_real": observed_real,
    }


def main():
    args = parse_args()
    if args.bundle:
        run_bundle_mode(args.bundle, args.out)
        return

    from dataloader import MultiCoilMRIDataset  # noqa: E402
    from inverse_problems.multi_coil_mri import MultiCoilMRI  # noqa: E402

    dataset = MultiCoilMRIDataset(
        args.kspace_dir,
        args.maps_dir,
        args.image_size,
        filenames=args.filenames,
    )
    samples = select_val_samples(dataset, args.slice_offset, args.val_slices)
    print(f"{len(samples)} slices across {len({f for _, f, _ in samples})} file(s)")

    results = {}
    for accel in args.accelerations:
        forward_op = MultiCoilMRI(
            total_lines=args.image_size[1],
            acceleration_ratio=accel,
            pattern=args.pattern,
            mask_seed=args.mask_seed,
            device="cpu",
        )
        rows = []
        for global_idx, filename, file_slice_idx in samples:
            data = dataset[global_idx]
            data = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
            stats = leak_and_floor_per_observed(forward_op, data)
            leak, floor = stats["leak"], stats["floor"]
            stats["leak_over_floor"] = leak / floor if floor > 0 else float("nan")
            stats.update(
                {
                    "filename": filename,
                    "global_idx": global_idx,
                    "file_slice_idx": file_slice_idx,
                }
            )
            rows.append(stats)
            print(
                f"  accel={accel} {filename} slice_idx={global_idx}: "
                f"leak={leak:.5f} floor={floor:.5f} "
                f"leak/floor={stats['leak_over_floor']:.3f}"
            )
            print(
                "      shift as % of denominator -> "
                f"||y||(masked)={stats['shift_frac_y'] * 100:.3f}%  "
                f"||A target||={stats['shift_frac_Atarget'] * 100:.3f}%  "
                f"||F S target||(unmasked)={stats['shift_frac_Afull'] * 100:.3f}%   "
                f"[floor={stats['floor_frac_y'] * 100:.2f}% of ||y||]"
            )

        def col(key):
            return torch.tensor([r[key] for r in rows])

        summary = {
            "leak_mean": col("leak").mean().item(),
            "floor_mean": col("floor").mean().item(),
            "leak_over_floor_mean": col("leak_over_floor").mean().item(),
            "leak_over_floor_median": col("leak_over_floor").median().item(),
            "leak_over_floor_min": col("leak_over_floor").min().item(),
            "leak_over_floor_max": col("leak_over_floor").max().item(),
            "shift_frac_y_mean": col("shift_frac_y").mean().item(),
            "shift_frac_y_max": col("shift_frac_y").max().item(),
            "floor_frac_y_mean": col("floor_frac_y").mean().item(),
            "n": len(rows),
        }
        results[f"accel_{accel}"] = {"summary": summary, "slices": rows}
        print(
            f"accel={accel}: leak/floor mean={summary['leak_over_floor_mean']:.3f} "
            f"median={summary['leak_over_floor_median']:.3f} "
            f"range=[{summary['leak_over_floor_min']:.3f}, "
            f"{summary['leak_over_floor_max']:.3f}]"
        )
        print(
            f"          residual shift={summary['shift_frac_y_mean'] * 100:.3f}% "
            f"of ||y|| (max {summary['shift_frac_y_max'] * 100:.3f}%) vs "
            f"floor={summary['floor_frac_y_mean'] * 100:.3f}% of ||y|| "
            f"(leak_mean={summary['leak_mean']:.5f}, "
            f"floor_mean={summary['floor_mean']:.5f}, n={summary['n']})"
        )

    print(
        "\nInterpretation: leak/floor > 1 means mask-null coil leakage exceeds "
        "the irreducible model-mismatch floor."
    )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
