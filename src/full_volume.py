from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def load_nifti_canonical(path: str | Path, *, allow_4d_mean: bool = False) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NIfTI volume: {path}")
    try:
        import nibabel as nib
    except Exception as exc:
        raise ImportError("nibabel is required for full-volume preprocessing.") from exc
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim == 4 and allow_4d_mean:
        # OASIS-3 AV45 PET NIfTI files are often dynamic frame series
        # [X,Y,Z,T]. For this classification pipeline we use one static PET
        # volume, so frames are collapsed by temporal mean before resampling.
        volume = volume.mean(axis=3)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume at {path}, got shape {volume.shape}")
    return np.nan_to_num(volume, copy=False)


def foreground_mask(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=bool)
    positive = volume > np.percentile(volume[finite], 20)
    mask = finite & positive
    if mask.sum() == 0:
        mask = finite & (volume != 0)
    if mask.sum() == 0:
        mask = finite
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=1)
    return mask.astype(bool)


def resize_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    zoom = [target / max(source, 1) for target, source in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom, order=order).astype(np.float32)


def robust_zscore(volume: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = volume[mask]
    if values.size == 0:
        raise ValueError("Foreground mask is empty; cannot normalize volume.")
    lo, hi = np.percentile(values, [1, 99])
    clipped = np.clip(volume, lo, hi)
    values = clipped[mask]
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad if mad > eps else values.std()
    return ((clipped - median) / (scale + eps)).astype(np.float32)


def robust_minmax(volume: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = volume[mask]
    if values.size == 0:
        raise ValueError("Foreground mask is empty; cannot normalize PET volume.")
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((volume - lo) / (hi - lo + eps), 0, 1).astype(np.float32)


def preprocess_mri(path: str | Path, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    volume = load_nifti_canonical(path, allow_4d_mean=False)
    volume = resize_to_shape(volume, target_shape, order=1)
    mask = foreground_mask(volume)
    normalized = robust_zscore(volume, mask)
    return normalized[None].astype(np.float32), mask[None].astype(np.uint8)


def preprocess_pet(path: str | Path, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    volume = load_nifti_canonical(path, allow_4d_mean=True)
    volume = resize_to_shape(volume, target_shape, order=1)
    mask = foreground_mask(volume)
    normalized = robust_minmax(volume, mask)
    return normalized[None].astype(np.float32), mask[None].astype(np.uint8)


def load_cached_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached full-volume array: {path}")
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"Expected cached [C,D,H,W] or [D,H,W] array, got {arr.shape}: {path}")
    return arr
