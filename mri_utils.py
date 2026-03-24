"""
MRI forward model, Fourier utilities, and undersampling masks.
All FFTs use norm='ortho' so the transform pair is unitary.
"""

import numpy as np
import torch


def to_kspace(image: torch.Tensor) -> torch.Tensor:
    x = image.to(torch.complex64) if not image.is_complex() else image
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                        dim=(-2, -1), norm='ortho'),
        dim=(-2, -1)
    )


def from_kspace(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)),
                         dim=(-2, -1), norm='ortho'),
        dim=(-2, -1)
    )


def center_crop(image: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    h, w = image.shape[-2], image.shape[-1]
    th, tw = target_shape
    start_h, start_w = (h - th) // 2, (w - tw) // 2
    return image[..., start_h:start_h + th, start_w:start_w + tw]


def create_mask(shape, acceleration=4, center_fraction=0.08,
                mask_type='random', seed=None):
    """Create Cartesian undersampling mask.

    Always keeps a dense center region and samples outer lines
    according to the chosen strategy. Adapts to any k-space width.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    H, W = shape[-2], shape[-1]
    num_center = int(W * center_fraction)
    num_total = max(int(W / acceleration), num_center)
    num_outer = num_total - num_center

    mask = np.zeros(W, dtype=np.float32)
    center_start = (W - num_center) // 2
    mask[center_start:center_start + num_center] = 1.0
    outer_indices = np.where(mask == 0)[0]

    if mask_type == 'random':
        chosen = rng.choice(outer_indices,
                            size=min(num_outer, len(outer_indices)), replace=False)
        mask[chosen] = 1.0

    elif mask_type == 'equispaced':
        if num_outer > 0 and len(outer_indices) > 0:
            step = max(len(outer_indices) // num_outer, 1)
            offset = rng.randint(0, step) if step > 1 else 0
            mask[outer_indices[offset::step][:num_outer]] = 1.0

    elif mask_type == 'gaussian':
        sigma = W / 6.0
        probs = np.exp(-0.5 * ((outer_indices - W / 2.0) / sigma) ** 2)
        probs /= probs.sum()
        chosen = rng.choice(outer_indices,
                            size=min(num_outer, len(outer_indices)),
                            replace=False, p=probs)
        mask[chosen] = 1.0

    elif mask_type == 'poisson_disc':
        if num_outer > 0 and len(outer_indices) > 0:
            min_dist = max(len(outer_indices) / (num_outer * 1.5), 1.0)
            selected = []
            candidates = list(outer_indices)
            rng.shuffle(candidates)
            for c in candidates:
                if len(selected) >= num_outer:
                    break
                if all(abs(c - s) >= min_dist for s in selected):
                    selected.append(c)
            if len(selected) < num_outer:
                remaining = [c for c in outer_indices if c not in selected]
                rng.shuffle(remaining)
                selected += remaining[:num_outer - len(selected)]
            mask[np.array(selected)] = 1.0
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return torch.from_numpy(mask).unsqueeze(0)


def ssim(prediction: 'torch.Tensor', target: 'torch.Tensor',
        window_size: int = 7, data_range: float = 1.0) -> float:
   """Compute mean SSIM between two batches of single channel images.
   Operates on CPU or GPU tensors of shape (B, 1, H, W).
   Uses the Wang et al. 2004 formulation with Gaussian weighting."""
   import torch
   import torch.nn.functional as F

   C1 = (0.01 * data_range) ** 2
   C2 = (0.03 * data_range) ** 2

   # 1D Gaussian kernel, outer product for 2D
   coords = torch.arange(window_size, dtype=torch.float32,
                          device=prediction.device) - window_size // 2
   g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
   g = g / g.sum()
   window = g.unsqueeze(1) * g.unsqueeze(0)
   window = window.unsqueeze(0).unsqueeze(0)

   pad = window_size // 2
   mu_x = F.conv2d(prediction, window, padding=pad)
   mu_y = F.conv2d(target, window, padding=pad)

   mu_x_sq = mu_x ** 2
   mu_y_sq = mu_y ** 2
   mu_xy = mu_x * mu_y

   sigma_x_sq = F.conv2d(prediction ** 2, window, padding=pad) - mu_x_sq
   sigma_y_sq = F.conv2d(target ** 2, window, padding=pad) - mu_y_sq
   sigma_xy = F.conv2d(prediction * target, window, padding=pad) - mu_xy

   num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
   den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

   return (num / den).mean().item()
