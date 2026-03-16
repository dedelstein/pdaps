import argparse
import h5py as h5
import numpy as np
import cupy as cp
from pathlib import Path
from tqdm import tqdm

import sigpy.mri as mri

device = cp.cuda.Device(0)

# From ESPIRiT paper
fastmri_default_configs = {
    "calib_width": 24,
    "thresh": 0.02,
    "kernel_width": 6,
    "crop": 0.01,
    "max_iter": 30,
    "device": device
}


def compute_espirit_for_volume(kspace_path: Path, output_path: Path):
    """
    Compute ESPIRiT sensitivity maps for every slice in one HDF5 volume.

    Args:
        kspace_path: Path to fastMRI HDF5 file, must contain 'kspace' dataset of shape
                     (num_slices, num_coils, height, width), dtype complex64.
        output_path: Where to save the companion HDF5 with 's_maps'.
                     Will have the same shape as kspace.
    """

    with h5.File(kspace_path, "r") as hf:
        # kspace shape: (num_slices, num_coils, height, width)
        # Each slice is an independent 2D acquisition.
        # Complex-valued: each entry is a complex64 number representing
        # amplitude and phase of the MR signal at that k-space location.
        kspace = hf["kspace"][:]

    # debug print
    print(f"kspace shape, type: {kspace.shape}, {kspace.dtype}")
    print("slices, coils, height, width")

    num_slices, num_coils, height, width = kspace.shape

    full_maps = np.zeros_like(kspace)

    for slice in range(num_slices):
        k_slice = kspace[slice]
        sensitivity_maps = mri.app.EspiritCalib(
            k_slice, **fastmri_default_configs
        ).run()

        # cupy -> numpy
        if hasattr(sensitivity_maps, "get"):
            sensitivity_maps = sensitivity_maps.get()

        full_maps[slice] = sensitivity_maps

    with h5.File(output_path, "w") as hf:
        hf.create_dataset("s_maps", data=full_maps.astype(np.complex64))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute ESPIRiT sensitivity maps for fastMRI multicoil data"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Root data directory containing multicoil data",
    )

    parser.add_argument(
        "-n",
        type=int,
        required=False,
        help="how many volumes to process (for debugging, default: all)",
    )

    args = parser.parse_args()
    input_dir = Path(args.data)
    output_dir = input_dir.parent / "s_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    

    # Each .h5 file = one "volume" = one patient scan = many slices
    h5_files = sorted(input_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} volumes in {input_dir}")

    if args.n is not None:
        h5_files = h5_files[:args.n]

    for h5_path in tqdm(h5_files, desc="Computing ESPIRiT maps"):
        output_path = output_dir / h5_path.name

        # Skip if already computed (allows resuming interrupted runs)
        if output_path.exists():
            continue

        compute_espirit_for_volume(h5_path, output_path)

    print(f"Done. Maps saved to {output_dir}")


if __name__ == "__main__":
    main()
