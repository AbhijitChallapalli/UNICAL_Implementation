from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MRISummaryConfig:
    histogram_bins: int = 16
    eps: float = 1e-8


def load_cached_mri(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected cached 3D MRI or [1,D,H,W], got {arr.shape}: {path}")
    return np.nan_to_num(arr, copy=False)


def load_cached_mask(path: str | None, volume: np.ndarray) -> np.ndarray:
    if path:
        mask = np.load(path)
        if mask.ndim == 4:
            mask = mask[0]
        mask = mask.astype(bool)
        if mask.shape == volume.shape:
            return mask
    finite = np.isfinite(volume)
    return finite & (np.abs(volume) > 0)


def _safe_stats(values: np.ndarray, prefix: str, names: list[str], vals: list[float]) -> None:
    if values.size == 0:
        values = np.asarray([0.0], dtype=np.float32)
    values = np.nan_to_num(values.astype(np.float64), copy=False)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for name, value in stats.items():
        names.append(f"{prefix}_{name}")
        vals.append(value)
    for p in percentiles:
        names.append(f"{prefix}_p{p:02d}")
        vals.append(float(np.percentile(values, p)))


def _entropy(values: np.ndarray, bins: int, eps: float) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=False)
    prob = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob + eps)).sum())


def extract_mri_summary_features(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    config: MRISummaryConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    config = config or MRISummaryConfig()
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32), copy=False)
    mask = np.asarray(mask if mask is not None else np.abs(volume) > 0, dtype=bool)
    values = volume[mask]
    names: list[str] = []
    vals: list[float] = []

    names.extend(["foreground_voxels", "foreground_fraction"])
    vals.extend([float(mask.sum()), float(mask.mean())])
    _safe_stats(values, "fg", names, vals)
    names.append("fg_hist_entropy")
    vals.append(_entropy(values, config.histogram_bins, config.eps))

    lo, hi = (np.percentile(values, [1, 99]) if values.size else (-1.0, 1.0))
    hist, _ = np.histogram(values, bins=config.histogram_bins, range=(lo, hi), density=False)
    denom = max(float(hist.sum()), 1.0)
    for i, count in enumerate(hist):
        names.append(f"fg_hist_bin_{i:02d}")
        vals.append(float(count) / denom)

    mid = volume.shape[0] // 2
    left = volume[:mid][mask[:mid]]
    right = volume[mid:][mask[mid:]]
    names.extend(["lr_mean_difference", "lr_std_difference"])
    vals.extend([
        float(np.mean(left) - np.mean(right)) if left.size and right.size else 0.0,
        float(np.std(left) - np.std(right)) if left.size and right.size else 0.0,
    ])

    planes = {
        "axial": volume[:, :, volume.shape[2] // 2],
        "coronal": volume[:, volume.shape[1] // 2, :],
        "sagittal": volume[volume.shape[0] // 2, :, :],
    }
    masks = {
        "axial": mask[:, :, mask.shape[2] // 2],
        "coronal": mask[:, mask.shape[1] // 2, :],
        "sagittal": mask[mask.shape[0] // 2, :, :],
    }
    for plane_name in ("axial", "coronal", "sagittal"):
        plane_values = planes[plane_name][masks[plane_name]]
        _safe_stats(plane_values, f"central_{plane_name}", names, vals)

    vector = np.asarray(vals, dtype=np.float32)
    return np.nan_to_num(vector), names
