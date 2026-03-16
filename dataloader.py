"""
Multicoil MRI Dataset
=====================

Loads multicoil fastMRI k-space and pre-computed ESPIRiT sensitivity maps,
then crops/resizes to the target image size and computes the MVUE
(Minimum Variance Unbiased Estimate) as the reconstruction target.

This follows the same preprocessing as inversebench's MultiCoilMRIData
(libs/inversebench/training/dataset.py) but takes explicit directory paths
instead of relying on a naming convention.

Data flow per slice:
    1. Load raw kspace (num_coils, H_raw, W_raw) from fastMRI HDF5
    2. Load ESPIRiT maps (num_coils, H_raw, W_raw) from companion HDF5
    3. Resize both to (num_coils, H_target, W_target) via sigpy
       - This is a FoV reduction, not interpolation. We crop in k-space
         (frequency domain) which is equivalent to reducing the field of
         view in image space. Standard practice for fastMRI.
    4. Compute MVUE = sum_c(ifft(kspace_c) * conj(maps_c)) / sqrt(sum_c(|maps_c|^2))
       - This is the optimal linear combination of coil images
    5. Normalize by 99th percentile of |MVUE| so pixel values are ~O(1)
    6. Return dict with:
       - target: (2, H, W) real tensor — real/imag channels of normalized MVUE
         This is what the diffusion model operates on (2-channel "image").
       - kspace: (num_coils, H, W) complex tensor — normalized k-space
         Used by the forward operator to create the observation.
       - maps: (num_coils, H, W) complex tensor — sensitivity maps in image domain
         Used by the forward operator: A(x) = mask * FFT(maps * x)
"""

import h5py
import numpy as np
import sigpy as sp
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiCoilMRIDataset(Dataset):
    def __init__(self, kspace_dir, maps_dir, image_size, slice_range=(5, -5), filenames=None):
        """
        Args:
            kspace_dir: Directory containing fastMRI multicoil HDF5 files.
                        Each file has key 'kspace' of shape (num_slices, num_coils, H, W).
            maps_dir:   Directory containing ESPIRiT maps HDF5 files.
                        Same filenames as kspace_dir, each with key 's_maps'.
            image_size:  (H, W) target image dimensions after cropping.
            slice_range: (start, end) — skip edge slices which are often empty.
                         Default (5, -5) means skip first 5 and last 5 slices.
            filenames:  Optional list of specific filenames to load (e.g. ["file1000196.h5"]).
                        If None, loads all .h5 files in maps_dir.
        """
        self.kspace_dir = Path(kspace_dir)
        self.maps_dir = Path(maps_dir)
        self.image_size = image_size

        # Build an index of (filename, slice_idx) pairs.
        # Each HDF5 file is one "volume" (one patient scan) containing many 2D slices.
        # We index individual slices so each __getitem__ returns one 2D slice.
        if filenames is not None:
            maps_files = [self.maps_dir / f for f in filenames]
        else:
            maps_files = sorted(self.maps_dir.glob("*.h5"))

        self.samples = []
        for maps_path in tqdm(maps_files, desc="Indexing data"):
            kspace_path = self.kspace_dir / maps_path.name
            if not kspace_path.exists():
                print(f"Warning: no matching kspace for {maps_path.name}, skipping")
                continue

            with h5py.File(maps_path, "r") as f:
                num_slices = f["s_maps"].shape[0]

            # slice_range=(5, -5) means indices 5, 6, ..., num_slices-6
            start = slice_range[0]
            end = num_slices + slice_range[1]
            for sl in range(start, end):
                self.samples.append((kspace_path, maps_path, sl))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_mvue(kspace, s_maps):
        """
        Minimum Variance Unbiased Estimate — the optimal way to combine
        coil images when you know the sensitivity maps.

        MVUE(x,y) = sum_c [ ifft(kspace_c)(x,y) * conj(maps_c(x,y)) ]
                     / sqrt( sum_c |maps_c(x,y)|^2 )

        The numerator weights each coil's image by how strongly that coil
        "sees" each pixel (conjugate of sensitivity). The denominator
        normalizes so we don't amplify regions seen by many coils.

        Args:
            kspace: (1, num_coils, H, W) complex — k-space measurements
            s_maps: (1, num_coils, H, W) complex — sensitivity maps
        Returns:
            (1, H, W) complex — combined image
        """
        # ifft each coil's k-space to get coil images, then combine
        return (
            np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1)
            / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))
        )

    @staticmethod
    def normalize(data, mvue):
        """
        Normalize by 99th percentile of |MVUE|.

        This makes pixel values ~O(1), which is what the diffusion model
        expects (it was trained on data in this range). The 99th percentile
        is used instead of max to be robust to hot-pixel outliers.
        """
        scaling = np.quantile(np.abs(mvue), 0.99)
        return data / scaling

    def __getitem__(self, idx):
        kspace_path, maps_path, slice_idx = self.samples[idx]

        # ── Load raw data ─────────────────────────────────────────────
        with h5py.File(kspace_path, "r") as f:
            # shape: (num_coils, H_raw, W_raw), complex64
            gt_ksp = f["kspace"][slice_idx]

        with h5py.File(maps_path, "r") as f:
            # shape: (num_coils, H_raw, W_raw), complex64
            maps = f["s_maps"][slice_idx]

        # ── Resize k-space to target image size ───────────────────────
        #
        # This is a FoV (field-of-view) reduction done in k-space.
        # sigpy.resize zero-pads or crops in k-space, which corresponds
        # to changing the field of view in image space.
        #
        # Phase-encode direction (last dim, W): direct crop
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Readout direction (second-to-last dim, H): go to hybrid space,
        # crop, then back to k-space. This avoids aliasing artifacts.
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0], gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,))

        # ── Resize sensitivity maps the same way ─────────────────────
        #
        # Maps live in image domain, so we first go to k-space, crop the
        # same way, then come back to image domain. This ensures the maps
        # are consistent with the cropped k-space.
        maps = sp.fft(maps, axes=(-2, -1))
        maps = sp.resize(maps, (maps.shape[0], maps.shape[1], self.image_size[1]))
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0], maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,))
        maps = sp.ifft(maps, axes=(-2, -1))

        # ── Compute MVUE target and normalize ─────────────────────────
        #
        # Add batch dim for get_mvue: (1, coils, H, W)
        mvue = self.get_mvue(
            gt_ksp.reshape((1,) + gt_ksp.shape),
            maps.reshape((1,) + maps.shape),
        )
        # Normalize everything by the same scaling factor
        mvue_scaled = self.normalize(mvue, mvue)
        gt_ksp_scaled = self.normalize(gt_ksp, mvue)

        # ── Convert to tensors ────────────────────────────────────────
        #
        # target: the diffusion model's "image" — 2-channel (real, imag)
        #   mvue_scaled is (1, H, W) complex -> squeeze -> (H, W) complex
        #   view_as_real -> (H, W, 2) -> permute -> (2, H, W) float
        target = torch.view_as_real(
            torch.from_numpy(mvue_scaled).squeeze(0)
        ).permute(2, 0, 1).contiguous()

        # kspace and maps: stay as complex tensors for the forward operator
        kspace = torch.from_numpy(gt_ksp_scaled.copy()).to(torch.complex64)
        maps = torch.from_numpy(maps.copy()).to(torch.complex64)

        return {
            "target": target,       # (2, H, W) float — what the model reconstructs
            "kspace": kspace,       # (coils, H, W) complex — for forward_op.__call__
            "maps": maps,           # (coils, H, W) complex — for forward_op.__call__ & .forward()
        }
