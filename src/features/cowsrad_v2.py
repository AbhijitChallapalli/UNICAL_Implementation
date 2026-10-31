from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import warnings

import numpy as np
from scipy import ndimage

from src.features.mri_summary import extract_mri_summary_features


@dataclass(frozen=True)
class CoWSRadV2Config:
    wavelet: str = "db2"
    wavelet_levels: int = 2
    log_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    gradient_hist_bins: int = 16
    grid_sizes: tuple[tuple[int, int, int], ...] = ((4, 4, 4), (6, 6, 6))
    sampled_slices: tuple[float, ...] = (0.25, 0.5, 0.75)
    warnings: list[str] = field(default_factory=list)


def _blocks(volume: np.ndarray, grid: tuple[int, int, int]) -> Iterable[np.ndarray]:
    splits = [np.array_split(np.arange(volume.shape[axis]), grid[axis]) for axis in range(3)]
    for xs in splits[0]:
        for ys in splits[1]:
            for zs in splits[2]:
                yield volume[np.ix_(xs, ys, zs)]


def _append_stats(prefix: str, arr: np.ndarray, names: list[str], vals: list[float]) -> None:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float64), copy=False).ravel()
    if arr.size == 0:
        arr = np.asarray([0.0])
    for name, value in {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "abs_mean": np.mean(np.abs(arr)),
        "energy": np.mean(arr * arr),
        "p10": np.percentile(arr, 10),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
    }.items():
        names.append(f"{prefix}_{name}")
        vals.append(float(value))


def _wavelet_features(volume: np.ndarray, config: CoWSRadV2Config, names: list[str], vals: list[float]) -> None:
    try:
        import pywt
    except Exception:
        warnings.warn("PyWavelets is not available; using Gaussian pyramid fallback for CoWS-Rad v2.")
        for level in range(1, config.wavelet_levels + 1):
            smooth = ndimage.gaussian_filter(volume, sigma=float(level))
            detail = volume - smooth
            _append_stats(f"wavelet_fallback_l{level}", detail, names, vals)
        return
    try:
        coeffs = pywt.wavedecn(volume, wavelet=config.wavelet, level=config.wavelet_levels)
        for level_idx, level in enumerate(coeffs):
            if isinstance(level, dict):
                for key in sorted(level):
                    _append_stats(f"wavelet_l{level_idx}_{key}", level[key], names, vals)
            else:
                _append_stats(f"wavelet_l{level_idx}_aaa", level, names, vals)
    except Exception as exc:
        warnings.warn(f"Wavelet extraction failed ({exc}); using Gaussian pyramid fallback.")
        for level in range(1, config.wavelet_levels + 1):
            detail = volume - ndimage.gaussian_filter(volume, sigma=float(level))
            _append_stats(f"wavelet_fallback_l{level}", detail, names, vals)


def extract_cowsrad_v2_features(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    config: CoWSRadV2Config | None = None,
) -> tuple[np.ndarray, list[str]]:
    config = config or CoWSRadV2Config()
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32), copy=False)
    mask = np.asarray(mask if mask is not None else np.abs(volume) > 0, dtype=bool)
    names: list[str] = []
    vals: list[float] = []

    summary, summary_names = extract_mri_summary_features(volume, mask)
    names.extend([f"basic__{name}" for name in summary_names])
    vals.extend([float(v) for v in summary])

    _wavelet_features(volume, config, names, vals)
    gradients = [ndimage.sobel(volume, axis=axis) for axis in range(3)]
    for axis, grad in enumerate(gradients):
        _append_stats(f"sobel_axis{axis}", grad[mask], names, vals)
    grad_mag = np.sqrt(sum(grad * grad for grad in gradients))
    _append_stats("gradient_magnitude", grad_mag[mask], names, vals)
    hist, _ = np.histogram(grad_mag[mask], bins=config.gradient_hist_bins, density=False)
    denom = max(float(hist.sum()), 1.0)
    for i, count in enumerate(hist):
        names.append(f"gradient_hist_bin_{i:02d}")
        vals.append(float(count) / denom)

    lap = ndimage.laplace(volume)
    _append_stats("laplacian", lap[mask], names, vals)
    for sigma in config.log_sigmas:
        response = ndimage.gaussian_laplace(volume, sigma=float(sigma))
        tag = str(sigma).replace(".", "p")
        _append_stats(f"log_sigma_{tag}", response[mask], names, vals)

    for grid in config.grid_sizes:
        grid_tag = "x".join(str(x) for x in grid)
        family = "grid4__" if grid == (4, 4, 4) else "grid6__" if grid == (6, 6, 6) else f"grid{grid_tag}__"
        for i, block in enumerate(_blocks(volume, grid)):
            _append_stats(f"{family}block_{grid_tag}_{i:03d}", block, names, vals)

    mid = volume.shape[0] // 2
    left = volume[:mid]
    right = np.flip(volume[mid:], axis=0)
    n = min(left.shape[0], right.shape[0])
    asym = left[:n] - right[:n]
    _append_stats("asym__left_right_asymmetry", asym, names, vals)

    for axis_name, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
        for frac in config.sampled_slices:
            idx = min(max(int(round((volume.shape[axis] - 1) * frac)), 0), volume.shape[axis] - 1)
            plane = np.take(volume, idx, axis=axis)
            plane_mask = np.take(mask, idx, axis=axis)
            tag = str(frac).replace(".", "p")
            _append_stats(f"slice__{axis_name}_{tag}", plane[plane_mask], names, vals)

    vector = np.asarray(vals, dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0), names
