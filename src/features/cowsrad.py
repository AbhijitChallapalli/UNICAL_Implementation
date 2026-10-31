from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage, stats


@dataclass
class CoWSRadConfig:
    wavelet: str = "db2"
    levels: int = 2
    resize_shape: tuple[int, int, int] = (32, 32, 32)
    eps: float = 1e-8


def load_volume(path: str | Path, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing volume file: {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        volume = np.load(path)
    elif suffixes.endswith(".npz"):
        data = np.load(path)
        first = data.files[0]
        volume = data[first]
    else:
        try:
            import nibabel as nib
        except Exception as exc:
            raise ImportError("nibabel is required to load NIfTI volumes.") from exc
        volume = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    volume = np.squeeze(np.asarray(volume, dtype=np.float32))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {volume.shape}")
    if shape is not None:
        zoom = [n / max(o, 1) for n, o in zip(shape, volume.shape)]
        volume = ndimage.zoom(volume, zoom, order=1)
    return volume.astype(np.float32)


def normalize_volume(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32), copy=False)
    lo, hi = np.percentile(volume, [1, 99])
    volume = np.clip(volume, lo, hi)
    return (volume - volume.mean()) / (volume.std() + eps)


def _basic_stats(x: np.ndarray, eps: float) -> list[float]:
    flat = x.astype(np.float64).ravel()
    if flat.size == 0:
        return [0.0] * 8
    hist, _ = np.histogram(flat, bins=32, density=True)
    hist = hist + eps
    return [
        float(flat.mean()),
        float(flat.std()),
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 50)),
        float(np.percentile(flat, 90)),
        float(stats.skew(flat, bias=False) if flat.size > 2 else 0.0),
        float(stats.kurtosis(flat, bias=False) if flat.size > 3 else 0.0),
        float(-(hist * np.log(hist)).sum()),
    ]


def _wavelet_features(volume: np.ndarray, config: CoWSRadConfig) -> list[float]:
    try:
        import pywt
    except Exception as exc:
        raise ImportError("PyWavelets is required for CoWS-Rad stationary wavelet features.") from exc
    features: list[float] = []
    max_level = pywt.swt_max_level(min(volume.shape))
    levels = max(1, min(config.levels, max_level))
    coeffs = pywt.swtn(volume, wavelet=config.wavelet, level=levels, trim_approx=False)
    for level in coeffs:
        for name in sorted(level):
            arr = np.asarray(level[name], dtype=np.float32)
            energy = float(np.mean(arr * arr))
            features.extend([energy, float(np.mean(np.abs(arr))), float(arr.std())])
    return features


def _directional_features(volume: np.ndarray) -> list[float]:
    features: list[float] = []
    # Contourlet-like directional energy approximation. This is not a full
    # contourlet transform; it uses stable 3D Sobel, Laplacian, and 3-plane
    # Gabor-like directional filter responses as a practical local substitute.
    for axis in range(3):
        sobel = ndimage.sobel(volume, axis=axis)
        features.extend([float(np.mean(np.abs(sobel))), float(np.mean(sobel * sobel))])
    lap = ndimage.laplace(volume)
    features.extend([float(np.mean(np.abs(lap))), float(np.mean(lap * lap))])
    planes = [
        volume[volume.shape[0] // 2, :, :],
        volume[:, volume.shape[1] // 2, :],
        volume[:, :, volume.shape[2] // 2],
    ]
    for plane in planes:
        for theta in np.linspace(0, np.pi, 6, endpoint=False):
            real = ndimage.convolve(plane, _gabor_kernel(theta), mode="reflect")
            features.extend([float(np.mean(np.abs(real))), float(np.mean(real * real))])
    return features


def _gabor_kernel(theta: float, sigma: float = 2.0, frequency: float = 0.25) -> np.ndarray:
    radius = 4
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-(xr * xr + yr * yr) / (2 * sigma * sigma)) * np.cos(2 * np.pi * frequency * xr)
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def extract_cowsrad_features(volume: np.ndarray, config: CoWSRadConfig | None = None) -> np.ndarray:
    config = config or CoWSRadConfig()
    volume = normalize_volume(volume, config.eps)
    features: list[float] = []
    features.extend(_basic_stats(volume, config.eps))
    features.extend(_wavelet_features(volume, config))
    features.extend(_directional_features(volume))
    gradients = np.gradient(volume)
    grad_mag = np.sqrt(sum(g * g for g in gradients))
    features.extend(_basic_stats(grad_mag, config.eps))
    for block in _blocks(volume, grid=(4, 4, 4)):
        features.extend([float(block.mean()), float(block.std()), float(np.mean(block * block))])
    vector = np.asarray(features, dtype=np.float32)
    vector = np.nan_to_num(vector)
    return vector


def _blocks(volume: np.ndarray, grid: tuple[int, int, int]) -> Iterable[np.ndarray]:
    splits = [np.array_split(np.arange(volume.shape[axis]), grid[axis]) for axis in range(3)]
    for xs in splits[0]:
        for ys in splits[1]:
            for zs in splits[2]:
                yield volume[np.ix_(xs, ys, zs)]


def extract_cowsrad_from_path(path: str | Path, config: CoWSRadConfig | None = None) -> np.ndarray:
    config = config or CoWSRadConfig()
    return extract_cowsrad_features(load_volume(path, config.resize_shape), config)
