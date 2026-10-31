from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from src.full_volume import foreground_mask, load_nifti_canonical, resize_to_shape
from src.mri_ad_cn_utils import local_path


def crop_background(volume: np.ndarray, mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return volume, mask
    lo = np.maximum(coords.min(axis=0) - int(margin), 0)
    hi = np.minimum(coords.max(axis=0) + int(margin) + 1, volume.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return volume[slices], mask[slices]


def simple_threshold_brain_mask(volume: np.ndarray) -> np.ndarray:
    mask = foreground_mask(volume)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labeled, n_labels = ndimage.label(mask)
    if n_labels > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        mask = labeled == int(np.argmax(counts))
    return mask.astype(bool)


def try_synthstrip(volume: np.ndarray, enabled: bool) -> tuple[np.ndarray, str]:
    if not enabled or shutil.which("mri_synthstrip") is None:
        return simple_threshold_brain_mask(volume), "simple_threshold_fallback"
    try:
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "in.nii.gz"
            out_path = Path(tmp) / "out.nii.gz"
            mask_path = Path(tmp) / "mask.nii.gz"
            nib.Nifti1Image(volume.astype(np.float32), np.eye(4)).to_filename(str(in_path))
            subprocess.run(["mri_synthstrip", "-i", str(in_path), "-o", str(out_path), "-m", str(mask_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
            return mask, "synthstrip"
    except Exception:
        return simple_threshold_brain_mask(volume), "simple_threshold_fallback"


def try_bias_correction(volume: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    if not config.get("enabled", True):
        return volume, "disabled"
    priority = list(config.get("method_priority", ["ants_n4", "simpleitk_n4", "skip"]))
    allow_native = bool(config.get("allow_native_backend", False))
    if not allow_native:
        if config.get("fail_if_unavailable", False):
            raise RuntimeError("Native N4 backends are disabled by preprocessing_3mt_style.bias_correction.allow_native_backend=false.")
        return volume, "skipped_native_backend_disabled"
    if "skip" in priority and "ants_n4" not in priority and "simpleitk_n4" not in priority:
        return volume, "skipped_by_priority"
    if "ants_n4" in priority:
        try:
            import ants  # type: ignore

            image = ants.from_numpy(volume.astype(np.float32))
            mask_image = ants.from_numpy(mask.astype(np.uint8))
            corrected = ants.n4_bias_field_correction(image, mask=mask_image)
            return np.asarray(corrected.numpy(), dtype=np.float32), "ants_n4"
        except Exception:
            pass
    if "simpleitk_n4" in priority:
        try:
            import SimpleITK as sitk  # type: ignore

            image = sitk.GetImageFromArray(volume.astype(np.float32))
            mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
            corrected = sitk.N4BiasFieldCorrection(image, mask_image)
            return sitk.GetArrayFromImage(corrected).astype(np.float32), "simpleitk_n4"
        except Exception:
            pass
    if config.get("fail_if_unavailable", False):
        raise RuntimeError("N4 bias correction requested but ANTsPy/SimpleITK are unavailable or failed.")
    return volume, "skipped_unavailable"


def try_affine_registration(volume: np.ndarray, template_path: Path, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    if not config.get("enabled", True):
        return volume, "disabled"
    priority = list(config.get("mode_priority", ["ants_affine", "simpleitk_affine", "skip"]))
    if not template_path.is_file():
        if config.get("fail_if_template_missing", False) or config.get("strict_3mt_mode", False):
            raise FileNotFoundError(f"Registration template is missing: {template_path}")
        return volume, "skipped_missing_template"
    allow_native = bool(config.get("allow_native_backend", False))
    if not allow_native:
        if config.get("strict_3mt_mode", False):
            raise RuntimeError("Native registration backends are disabled by preprocessing_3mt_style.registration.allow_native_backend=false.")
        return volume, "skipped_native_backend_disabled"
    if "ants_affine" in priority:
        try:
            import ants  # type: ignore

            fixed = ants.image_read(str(template_path))
            moving = ants.from_numpy(volume.astype(np.float32))
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")
            return np.asarray(reg["warpedmovout"].numpy(), dtype=np.float32), "ants_affine"
        except Exception:
            pass
    if "simpleitk_affine" in priority:
        try:
            import SimpleITK as sitk  # type: ignore

            fixed = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
            moving = sitk.GetImageFromArray(volume.astype(np.float32))
            sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
            method = sitk.ImageRegistrationMethod()
            method.SetMetricAsMattesMutualInformation(32)
            method.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 100)
            method.SetInterpolator(sitk.sitkLinear)
            transform = method.Execute(fixed, moving)
            resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            return sitk.GetArrayFromImage(resampled).astype(np.float32), "simpleitk_affine"
        except Exception:
            pass
    if config.get("strict_3mt_mode", False):
        raise RuntimeError("Affine registration requested in strict mode but no backend succeeded.")
    return volume, "skipped_unavailable"


def robust_normalize(volume: np.ndarray, mask: np.ndarray, clip_percentiles: list[float] | tuple[float, float] = (1, 99), eps: float = 1e-6) -> np.ndarray:
    values = volume[mask]
    if values.size == 0:
        raise ValueError("Foreground mask is empty; cannot normalize.")
    lo, hi = np.percentile(values, clip_percentiles)
    clipped = np.clip(volume, lo, hi)
    values = clipped[mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > eps else float(values.std())
    return ((clipped - median) / (scale + eps)).astype(np.float32)


def preprocess_mri_3mt_style(path: str | Path, config: dict[str, Any], skip_registration: bool = False, debug_shape: bool = False) -> dict[str, Any]:
    prep = config["preprocessing_3mt_style"]
    target_shape = tuple(int(v) for v in (prep["debug_target_shape"] if debug_shape or prep.get("use_debug_shape") else prep["target_shape"]))
    volume = load_nifti_canonical(path, allow_4d_mean=False)
    original_shape = tuple(int(v) for v in volume.shape)
    original_mask = foreground_mask(volume)
    if not np.isfinite(volume).all() or not original_mask.any():
        raise ValueError("MRI volume has non-finite values or empty foreground.")
    if prep.get("crop_background", True):
        volume, original_mask = crop_background(volume, original_mask, int(prep.get("crop_margin_voxels", 8)))
    mask, skull_method = try_synthstrip(volume, bool(prep.get("skull_strip", {}).get("enabled", True)))
    volume, bias_method = try_bias_correction(volume, mask, prep.get("bias_correction", {}))
    reg_cfg = dict(prep.get("registration", {}))
    if skip_registration:
        reg_cfg["enabled"] = False
    template = local_path(config, reg_cfg.get("template_path", ""))
    volume, reg_method = try_affine_registration(volume, template, reg_cfg)
    mask = simple_threshold_brain_mask(volume)
    resized = resize_to_shape(volume, target_shape, order=1)
    resized_mask = resize_to_shape(mask.astype(np.float32), target_shape, order=0) > 0.5
    normalized = robust_normalize(resized, resized_mask, prep.get("normalization", {}).get("clip_percentiles", [1, 99]))
    return {
        "mri": normalized.astype(np.float32),
        "mask": resized_mask.astype(np.uint8),
        "used_skull_strip_method": skull_method,
        "used_bias_correction_method": bias_method,
        "used_registration_method": reg_method,
        "original_shape": json.dumps(original_shape),
        "final_shape": json.dumps(tuple(int(v) for v in normalized.shape)),
        "voxel_spacing": "",
    }
