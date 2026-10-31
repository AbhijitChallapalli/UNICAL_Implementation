"""3D volumetric augmentations for self-supervised MRI pretraining.

All transforms operate on tensors of shape [C, D, H, W] (typically C=1).
They are designed to be composed and applied on-the-fly during training.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class RandomCrop3D:
    """Random crop to a fixed spatial size.  If the volume is smaller than the
    target on any axis the volume is returned unchanged on that axis."""

    def __init__(self, size: Sequence[int], prob: float = 0.8) -> None:
        self.size = tuple(int(s) for s in size)
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return x
        _, d, h, w = x.shape
        td, th, tw = self.size
        td, th, tw = min(td, d), min(th, h), min(tw, w)
        d0 = random.randint(0, d - td)
        h0 = random.randint(0, h - th)
        w0 = random.randint(0, w - tw)
        return x[:, d0 : d0 + td, h0 : h0 + th, w0 : w0 + tw]


class RandomFlip3D:
    """Randomly flip along one or more spatial axes."""

    def __init__(self, axes: Sequence[int] = (0, 1, 2), prob: float = 0.5) -> None:
        # axes are spatial: 0→D, 1→H, 2→W  (tensor dims 1,2,3)
        self.axes = [a + 1 for a in axes]  # shift to tensor dims
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for ax in self.axes:
            if random.random() < self.prob:
                x = x.flip(ax)
        return x


class IntensityShift:
    """Additive Gaussian intensity shift (global per-volume)."""

    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            x = x + torch.randn(1, device=x.device) * self.std
        return x


class IntensityScale:
    """Multiplicative intensity scaling (global per-volume)."""

    def __init__(self, scale_range: Sequence[float] = (0.9, 1.1)) -> None:
        self.lo, self.hi = float(scale_range[0]), float(scale_range[1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(self.lo, self.hi)
        return x * factor


class GaussianNoise:
    """Per-voxel Gaussian noise."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            x = x + torch.randn_like(x) * self.std
        return x


class GaussianBlur3D:
    """Isotropic 3D Gaussian blur applied with probability `prob`."""

    def __init__(
        self,
        sigma_range: Sequence[float] = (0.5, 1.0),
        prob: float = 0.2,
        kernel_size: int = 5,
    ) -> None:
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))
        self.prob = prob
        self.kernel_size = kernel_size | 1  # ensure odd

    def _gaussian_kernel_1d(self, sigma: float, device: torch.device) -> torch.Tensor:
        ks = self.kernel_size
        x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return x
        sigma = random.uniform(*self.sigma_range)
        k = self._gaussian_kernel_1d(sigma, x.device)
        # Separable convolution: D → H → W
        c = x.shape[0]
        pad = self.kernel_size // 2
        # Reshape for grouped conv1d along each axis
        for axis in (1, 2, 3):
            # Move target axis to last, flatten others
            perm = list(range(4))
            perm.remove(axis)
            perm.append(axis)
            xt = x.permute(*perm).contiguous()
            shape = xt.shape
            xt = xt.reshape(-1, 1, shape[-1])
            xt = F.conv1d(xt, k.view(1, 1, -1), padding=pad)
            xt = xt.reshape(*shape)
            inv_perm = [0] * 4
            for i, p in enumerate(perm):
                inv_perm[p] = i
            x = xt.permute(*inv_perm).contiguous()
        return x


class PadToShape:
    """Zero-pad a volume to at least the given shape (used after random crop
    to ensure a consistent size for batching when crop is skipped)."""

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(int(s) for s in shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, d, h, w = x.shape
        td, th, tw = self.shape
        if d >= td and h >= th and w >= tw:
            return x
        pd = max(td - d, 0)
        ph = max(th - h, 0)
        pw = max(tw - w, 0)
        # F.pad order is reversed: (w_left, w_right, h_left, h_right, d_left, d_right)
        x = F.pad(x, (0, pw, 0, ph, 0, pd))
        return x


class CenterCropOrPad:
    """Deterministic center crop/pad to a fixed shape for validation."""

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(int(s) for s in shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, d, h, w = x.shape
        td, th, tw = self.shape
        # Pad if needed
        pd = max(td - d, 0)
        ph = max(th - h, 0)
        pw = max(tw - w, 0)
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))
            _, d, h, w = x.shape
        # Crop
        d0 = (d - td) // 2
        h0 = (h - th) // 2
        w0 = (w - tw) // 2
        return x[:, d0 : d0 + td, h0 : h0 + th, w0 : w0 + tw]


# ---------------------------------------------------------------------------
# Compose helper
# ---------------------------------------------------------------------------

class Compose3D:
    """Apply a list of 3D transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# Factory: build augmentation pipeline from config
# ---------------------------------------------------------------------------

def build_ssl_augmentation(config: dict[str, Any]) -> Compose3D:
    """Build the full SSL augmentation pipeline from the ssl.augmentation config block."""
    aug = config.get("ssl", {}).get("augmentation", {})

    transforms: list = []

    # Random crop
    crop_size = aug.get("random_crop_size")
    if crop_size:
        transforms.append(RandomCrop3D(crop_size, prob=float(aug.get("random_crop_prob", 0.8))))
        # Ensure minimum size for batching
        # A DataLoader can only stack fixed-size tensors. When the random crop
        # is skipped, use a deterministic center crop to the same shape.
        transforms.append(CenterCropOrPad(crop_size))

    # Random flip
    flip_prob = float(aug.get("flip_prob", 0.5))
    if flip_prob > 0:
        axes = aug.get("flip_axes", [0, 1, 2])
        transforms.append(RandomFlip3D(axes, prob=flip_prob))

    # Intensity augmentations
    shift_std = float(aug.get("intensity_shift_std", 0.0))
    if shift_std > 0:
        transforms.append(IntensityShift(std=shift_std))

    scale_range = aug.get("intensity_scale_range")
    if scale_range and (float(scale_range[0]) != 1.0 or float(scale_range[1]) != 1.0):
        transforms.append(IntensityScale(scale_range))

    noise_std = float(aug.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        transforms.append(GaussianNoise(std=noise_std))

    # Gaussian blur
    blur_prob = float(aug.get("gaussian_blur_prob", 0.0))
    if blur_prob > 0:
        transforms.append(GaussianBlur3D(
            sigma_range=aug.get("gaussian_blur_sigma_range", [0.5, 1.0]),
            prob=blur_prob,
        ))

    return Compose3D(transforms)


def build_ssl_validation_transform(config: dict[str, Any]) -> Compose3D:
    """Build a deterministic transform for held-out SSL validation volumes."""
    aug = config.get("ssl", {}).get("augmentation", {})
    crop_size = aug.get("random_crop_size")
    return Compose3D([CenterCropOrPad(crop_size)]) if crop_size else Compose3D([])
