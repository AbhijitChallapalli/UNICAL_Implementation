from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.full_volume import foreground_mask, load_nifti_canonical, resize_to_shape
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, local_path, output_dir, write_json


def status(config: dict, payload: dict) -> None:
    write_json(output_dir(config, "qc", "synthetic_pet_status.json"), payload)
    print(json.dumps(payload, indent=2))


def sample_paths(row: pd.Series, base_dir: Path, required_k: int) -> list[Path]:
    paths = []
    explicit = [
        c
        for c in row.index
        if str(c).startswith("sample_") and str(c) != "sample_id" and str(row[c]) not in ("", "nan", "None")
    ]
    for c in explicit:
        paths.append(Path(str(row[c])))
    if not paths:
        paths = [base_dir / str(row["sample_id"]) / f"sample_{i:02d}.nii.gz" for i in range(required_k)]
    return paths[:required_k]


def entropy(values: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(values[np.isfinite(values)], bins=bins, range=(0, 1), density=True)
    p = hist / max(hist.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def summary_features(mean_pet: np.ndarray, gated: np.ndarray, uncertainty: np.ndarray, mask: np.ndarray) -> list[float]:
    vals = []
    for arr in (mean_pet, gated, uncertainty):
        data = arr[mask]
        vals.extend([float(np.mean(data)), float(np.std(data)), float(np.median(data)), float(np.percentile(data, 10)), float(np.percentile(data, 90))])
    u = uncertainty[mask]
    vals.extend([float((u > 0.75).mean()), float((u < 0.25).mean()), entropy(u)])
    g = gated[mask]
    vals.extend([float(np.median(np.abs(g - np.median(g)))), float(mask.mean())])
    mid = gated.shape[0] // 2
    left = gated[:mid][mask[:mid]]
    right = gated[-mid:][mask[-mid:]]
    vals.append(float((left.mean() - right.mean()) / (abs(left.mean()) + abs(right.mean()) + 1e-6)) if left.size and right.size else 0.0)
    for grid in (2, 4):
        factors = [grid / s for s in gated.shape]
        coarse = ndimage.zoom(gated, factors, order=1)
        vals.extend([float(v) for v in coarse.ravel()])
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gated synthetic PET and uncertainty-summary features if K samples exist.")
    parser.add_argument("--config", default="configs/oasis3_unical_cmt_3mt_style.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    pet_cfg = config.get("synthetic_pet", {})
    if not pet_cfg.get("enabled", True):
        status(config, {"status": "disabled"})
        return
    manifest_path = cfg_path(config, "synthetic_pet_k_manifest")
    if not manifest_path.is_file():
        payload = {"status": "skipped_missing_manifest", "manifest": str(manifest_path)}
        status(config, payload)
        if pet_cfg.get("fail_if_missing", False):
            raise FileNotFoundError(manifest_path)
        return
    frame = pd.read_csv(manifest_path)
    if args.limit:
        frame = frame.head(args.limit).copy()
    required_k = int(pet_cfg.get("required_k", 4))
    base_dir = cfg_path(config, "synthetic_pet_k_output_dir")
    target_shape = tuple(int(v) for v in config["preprocessing_3mt_style"]["target_shape"])
    out_cfg = pet_cfg.get("outputs", {})
    gated_dir = local_path(config, out_cfg.get("gated_pet_dir", "data/cache/unical_cmt_3mt_pet/gated_pet"))
    mean_dir = local_path(config, out_cfg.get("mean_pet_dir", "data/cache/unical_cmt_3mt_pet/mean_pet"))
    unc_dir = local_path(config, out_cfg.get("uncertainty_dir", "data/cache/unical_cmt_3mt_pet/uncertainty"))
    for d in (gated_dir, mean_dir, unc_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    features = []
    feature_names = [f"pet_summary_{i:03d}" for i in range(97)]
    alpha = float(pet_cfg.get("uncertainty_gate", {}).get("alpha", 1.0))
    eps = float(pet_cfg.get("uncertainty_gate", {}).get("eps", 1e-8))
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        rec = {"sample_id": sample_id, "status": "failed", "error": "", "gated_pet_cache_path": str(gated_dir / f"{sample_id}_gated_pet.npy")}
        try:
            paths = sample_paths(row, base_dir, required_k)
            if len(paths) < required_k or not all(p.is_file() for p in paths):
                raise FileNotFoundError(f"Need {required_k} PET samples for {sample_id}; found {[str(p) for p in paths if p.is_file()]}")
            vols = []
            for path in paths:
                vol = load_nifti_canonical(path, allow_4d_mean=True)
                if vol.shape != target_shape:
                    vol = resize_to_shape(vol, target_shape, order=1)
                if not np.isfinite(vol).all() or np.count_nonzero(vol) == 0:
                    raise ValueError(f"Invalid PET sample: {path}")
                vols.append(vol.astype(np.float32))
            stack = np.stack(vols)
            mean_pet = stack.mean(axis=0)
            var = stack.var(axis=0)
            if float(var.max()) <= 0:
                raise ValueError("Synthetic PET samples have zero variance.")
            uncertainty = var / (float(var.max()) + eps)
            gate = np.exp(-alpha * uncertainty)
            gated = mean_pet * gate
            mask = foreground_mask(mean_pet)
            np.save(mean_dir / f"{sample_id}_mean_pet.npy", mean_pet.astype(np.float32))
            np.save(unc_dir / f"{sample_id}_uncertainty.npy", uncertainty.astype(np.float32))
            np.save(gated_dir / f"{sample_id}_gated_pet.npy", gated.astype(np.float32))
            feat = summary_features(mean_pet, gated, uncertainty, mask)
            features.append(feat)
            rec.update({"status": "passed", "mean_pet_cache_path": str(mean_dir / f"{sample_id}_mean_pet.npy"), "uncertainty_cache_path": str(unc_dir / f"{sample_id}_uncertainty.npy")})
        except Exception as exc:
            rec["error"] = str(exc)
        rows.append(rec)
    out_manifest = cfg_path(config, "gated_pet_manifest_csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    passed_ids = [r["sample_id"] for r in rows if r["status"] == "passed"]
    if features:
        X = np.asarray(features, dtype=np.float32)
        names = feature_names[: X.shape[1]]
        out_npz = cfg_path(config, "pet_summary_features_npz")
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, X=X, sample_ids=np.asarray(passed_ids), feature_names=np.asarray(names))
    payload = {"status": "completed" if features else "skipped_no_valid_pet", "samples": int(len(rows)), "passed": int(len(passed_ids)), "required_k": required_k}
    status(config, payload)
    if not features and pet_cfg.get("fail_if_missing", False):
        raise RuntimeError("No valid synthetic PET K samples were available.")


if __name__ == "__main__":
    main()
