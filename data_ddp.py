"""
DDP compatible data pipeline for fastMRI single coil knee data.

HDF5 handles are cached per worker via a singleton pattern and cleared
in worker_init_fn on fork. The .copy() call on numpy reads prevents
the h5py memory leak documented in fastMRI Issue 215.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from mri_utils import to_kspace, from_kspace, center_crop, create_mask


class FastMRISliceDataset(Dataset):
    """fastMRI single coil dataset returning normalized magnitude images.

    Builds a flat (path, slice_idx) index during __init__ by scanning
    file metadata. K-space data is loaded lazily in __getitem__ with
    per worker HDF5 handle caching for performance.
    """

    def __init__(self, data_dir, acceleration=4, center_fraction=0.08,
                 mask_type='random', target_type='reconstruction_esc',
                 fixed_masks=False, max_volumes=None, augment=False):
        self.data_dir = Path(data_dir)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.mask_type = mask_type
        self.target_type = target_type
        self.fixed_masks = fixed_masks
        self.augment = augment
        self._h5_cache = {}

        self.examples = []
        h5_files = sorted(self.data_dir.glob("*.h5"))
        if max_volumes:
            h5_files = h5_files[:max_volumes]

        skipped = 0
        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    if "kspace" not in f or self.target_type not in f:
                        skipped += 1
                        continue
                    num_slices = f["kspace"].shape[0]
                    self.examples += [(h5_path, i) for i in range(num_slices)]
            except Exception:
                skipped += 1

        n_vols = len(set(p for p, _ in self.examples))
        print(f"FastMRISliceDataset: {len(self.examples)} slices "
              f"from {n_vols} volumes ({self.data_dir.name})")
        if skipped:
            print(f"  Skipped {skipped} volumes (missing keys or unreadable)")

    def _get_h5(self, path):
        key = str(path)
        if key not in self._h5_cache:
            self._h5_cache[key] = h5py.File(key, "r")
        return self._h5_cache[key]

    def close_handles(self):
        for h in self._h5_cache.values():
            try:
                h.close()
            except Exception:
                pass
        self._h5_cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        h5_path, slice_idx = self.examples[idx]
        f = self._get_h5(h5_path)

        kspace = torch.from_numpy(f["kspace"][slice_idx].copy())
        target = torch.from_numpy(f[self.target_type][slice_idx].copy()).float()

        seed = idx if self.fixed_masks else None
        mask = create_mask(kspace.shape, acceleration=self.acceleration,
                           center_fraction=self.center_fraction,
                           mask_type=self.mask_type, seed=seed)

        mag = center_crop(torch.abs(from_kspace(kspace * mask)), (320, 320))
        mag_min, mag_max = mag.min(), mag.max()
        input_img = ((mag - mag_min) / (mag_max - mag_min + 1e-8)).unsqueeze(0)

        t_min, t_max = target.min(), target.max()
        target_norm = ((target - t_min) / (t_max - t_min + 1e-8)).unsqueeze(0)

        if self.augment and torch.rand(1).item() > 0.5:
            input_img = torch.flip(input_img, dims=[-1])
            target_norm = torch.flip(target_norm, dims=[-1])

        return input_img, target_norm


class SyntheticMRIDataset(Dataset):
    """Random ellipse phantoms for pipeline testing without real data."""

    def __init__(self, num_samples=200, image_size=(320, 320),
                 acceleration=4, mask_type='random'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.acceleration = acceleration
        self.mask_type = mask_type

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        h, w = self.image_size
        img = np.zeros((h, w), dtype=np.float32)
        y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
        img += 1.0 * ((x / 0.9) ** 2 + (y / 0.7) ** 2 < 1)
        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.uniform(-0.5, 0.5, 2)
            rx, ry = rng.uniform(0.05, 0.25, 2)
            img += rng.uniform(0.3, 0.8) * (
                ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1
            )

        target = torch.from_numpy(img).float()
        kspace = to_kspace(target)
        mask = create_mask(self.image_size, acceleration=self.acceleration,
                           mask_type=self.mask_type, seed=idx)

        mag = torch.abs(from_kspace(kspace * mask))
        mag_min, mag_max = mag.min(), mag.max()
        input_img = ((mag - mag_min) / (mag_max - mag_min + 1e-8)).unsqueeze(0)

        t_min, t_max = target.min(), target.max()
        target_norm = ((target - t_min) / (t_max - t_min + 1e-8)).unsqueeze(0)

        return input_img, target_norm


def _worker_init_fn(worker_id):
    """Reset HDF5 handle cache in each DataLoader worker.
    Forked processes inherit the parent's file descriptors, which
    are invalid across the fork boundary for HDF5.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, 'close_handles'):
        info.dataset.close_handles()


def create_dataloader(dataset, batch_size, num_workers=4,
                      distributed=False, shuffle=True, drop_last=True):
    """Build a DataLoader with optional DistributedSampler.

    When distributed, shuffling is handled by the sampler
    (call sampler.set_epoch(epoch) each epoch).
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    needs_worker_init = num_workers > 0 and hasattr(dataset, 'close_handles')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if needs_worker_init else None,
    )
