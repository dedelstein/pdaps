"""Pre-compute ESPIRiT sensitivity maps for fastMRI multicoil volumes."""

import argparse
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm import tqdm


FASTMRI_DEFAULT_CONFIG = {
    "calib_width": 24,
    "thresh": 0.02,
    "kernel_width": 6,
    "crop": 0.01,
    "max_iter": 30,
}


def _espirit_app(device_id=0):
    try:
        import cupy as cp
        import sigpy.mri as mri
    except ImportError as exc:
        raise ImportError(
            "ESPIRiT map generation requires cupy and sigpy. Install a CuPy build "
            "matching your CUDA runtime, then install sigpy."
        ) from exc
    return mri.app.EspiritCalib, cp.cuda.Device(device_id)


def compute_espirit_for_volume(kspace_path, output_path, device_id=0, config=None):
    """Compute ESPIRiT sensitivity maps for every slice in one HDF5 volume."""
    kspace_path = Path(kspace_path)
    output_path = Path(output_path)
    config = {**FASTMRI_DEFAULT_CONFIG, **(config or {})}
    espirit_calib, device = _espirit_app(device_id=device_id)
    config["device"] = device

    with h5.File(kspace_path, "r") as handle:
        kspace = handle["kspace"][:]

    num_slices = kspace.shape[0]
    full_maps = np.zeros_like(kspace)
    for slice_idx in tqdm(range(num_slices), desc=kspace_path.name):
        maps = espirit_calib(kspace[slice_idx], **config).run()
        if hasattr(maps, "get"):
            maps = maps.get()
        full_maps[slice_idx] = maps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5.File(output_path, "w") as handle:
        handle.create_dataset("s_maps", data=full_maps.astype(np.complex64))


def main():
    parser = argparse.ArgumentParser(description="Compute ESPIRiT maps for fastMRI multicoil HDF5 files.")
    parser.add_argument("--data", required=True, help="Directory containing fastMRI multicoil .h5 files.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <data-parent>/s_maps.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("-n", type=int, default=None, help="Number of volumes to process.")
    args = parser.parse_args()

    input_dir = Path(args.data)
    output_dir = Path(args.out_dir) if args.out_dir else input_dir.parent / "s_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))
    if args.n is not None:
        h5_files = h5_files[: args.n]
    print(f"Found {len(h5_files)} volumes in {input_dir}")

    for kspace_path in tqdm(h5_files, desc="Computing ESPIRiT maps"):
        output_path = output_dir / kspace_path.name
        if output_path.exists():
            continue
        compute_espirit_for_volume(kspace_path, output_path, device_id=args.device_id)

    print(f"Done. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()

