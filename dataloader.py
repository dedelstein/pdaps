"""fastMRI multicoil dataset with precomputed ESPIRiT maps.

Each item returns a normalized MVUE target, normalized multicoil k-space, and
the corresponding sensitivity maps. Cropping uses SigPy's Fourier-domain resize
so the k-space, maps, and target stay mutually consistent.
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
        """Index slices from matching k-space and sensitivity-map HDF5 files."""
        self.kspace_dir = Path(kspace_dir)
        self.maps_dir = Path(maps_dir)
        self.image_size = image_size

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

            start = slice_range[0]
            end = num_slices + slice_range[1]
            for sl in range(start, end):
                self.samples.append((kspace_path, maps_path, sl))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_mvue(kspace, s_maps):
        """Minimum-variance coil combination from k-space and sensitivity maps."""
        return (
            np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1)
            / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))
        )

    @staticmethod
    def normalize(data, mvue):
        """Normalize by the 99th percentile of |MVUE|."""
        scaling = np.quantile(np.abs(mvue), 0.99)
        return data / scaling

    def __getitem__(self, idx):
        kspace_path, maps_path, slice_idx = self.samples[idx]

        with h5py.File(kspace_path, "r") as f:
            gt_ksp = f["kspace"][slice_idx]

        with h5py.File(maps_path, "r") as f:
            maps = f["s_maps"][slice_idx]

        # FoV reduction: crop phase encode directly, readout through hybrid space.
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0], gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,))

        # Resize image-domain maps with the same Fourier-domain geometry.
        maps = sp.fft(maps, axes=(-2, -1))
        maps = sp.resize(maps, (maps.shape[0], maps.shape[1], self.image_size[1]))
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0], maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,))
        maps = sp.ifft(maps, axes=(-2, -1))

        mvue = self.get_mvue(
            gt_ksp.reshape((1,) + gt_ksp.shape),
            maps.reshape((1,) + maps.shape),
        )
        mvue_scaled = self.normalize(mvue, mvue)
        gt_ksp_scaled = self.normalize(gt_ksp, mvue)

        target = torch.view_as_real(
            torch.from_numpy(mvue_scaled).squeeze(0)
        ).permute(2, 0, 1).contiguous()

        kspace = torch.from_numpy(gt_ksp_scaled.copy()).to(torch.complex64)
        maps = torch.from_numpy(maps.copy()).to(torch.complex64)

        return {
            "target": target,
            "kspace": kspace,
            "maps": maps,
        }
