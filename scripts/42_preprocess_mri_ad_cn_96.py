from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.full_volume import foreground_mask, load_nifti_canonical, resize_to_shape, robust_zscore
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, output_dir, write_json


def save_example(volume: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=150)
        slices = [volume[volume.shape[0] // 2], volume[:, volume.shape[1] // 2, :], volume[:, :, volume.shape[2] // 2]]
        for ax, image in zip(axes, slices):
            ax.imshow(np.rot90(image), cmap="gray")
            ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:
        print(f"[preprocess] warning: failed to save QC example {path}: {exc}")


def preprocess_one(row: pd.Series, config: dict, overwrite: bool) -> dict:
    target_shape = tuple(int(x) for x in config["preprocessing"]["target_shape"])
    cache_dir = cfg_path(config, "cache_dir")
    sample_id = str(row["sample_id"])
    mri_out = cache_dir / f"{sample_id}_mri.npy"
    mask_out = cache_dir / f"{sample_id}_mask.npy"
    rec = row.to_dict()
    rec.update({"mri_cache_96_path": str(mri_out), "brain_mask_cache_path": str(mask_out), "preprocess_status": "failed", "preprocess_error": ""})
    try:
        if mri_out.is_file() and mask_out.is_file() and not overwrite:
            rec["preprocess_status"] = "cached"
            return rec
        volume = load_nifti_canonical(row["mri_nifti_path"], allow_4d_mean=False)
        mask_native = foreground_mask(volume)
        if float(mask_native.mean()) < float(config["preprocessing"]["min_foreground_fraction"]):
            raise ValueError(f"Foreground fraction too small before resize: {mask_native.mean():.4f}")
        values = volume[mask_native]
        lo, hi = np.percentile(values, config["preprocessing"]["clip_percentiles"])
        clipped = np.clip(volume, lo, hi)
        resized = resize_to_shape(clipped, target_shape, order=1)
        resized_mask = ndimage.zoom(mask_native.astype(np.float32), [t / s for t, s in zip(target_shape, mask_native.shape)], order=0) > 0.5
        normed = robust_zscore(resized, resized_mask)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(mri_out, normed.astype(np.float32))
        np.save(mask_out, resized_mask.astype(np.uint8))
        rec["preprocess_status"] = "passed"
    except Exception as exc:
        rec["preprocess_error"] = str(exc)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AD/CN MRI volumes to 96x96x96 cache.")
    parser.add_argument("--config", default="configs/oasis3_mri_ad_cn_ablation_96.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    frame = pd.read_csv(cfg_path(config, "qc_csv"))
    frame = frame[frame["qc_status"].eq("passed")].copy()
    if args.limit:
        frame = frame.head(args.limit).copy()
    overwrite = bool(args.overwrite or config["preprocessing"].get("overwrite", False))
    rows = [preprocess_one(row, config, overwrite) for _, row in frame.iterrows()]
    out = pd.DataFrame(rows)
    out_path = cfg_path(config, "preprocessed_manifest_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    if bool(config["preprocessing"].get("save_qc_examples", True)):
        examples_per_class = int(config["preprocessing"].get("qc_examples_per_class", 12))
        for label, subset in out[out["preprocess_status"].isin(["passed", "cached"])].groupby("label"):
            for _, row in subset.head(examples_per_class).iterrows():
                save_example(np.load(row["mri_cache_96_path"]), output_dir(config, "qc", "example_slices", f"{label}_{row['sample_id']}.png"))
    summary = {
        "samples": int(len(out)),
        "passed_or_cached": int(out["preprocess_status"].isin(["passed", "cached"]).sum()),
        "failed": int((out["preprocess_status"] == "failed").sum()),
        "target_shape": config["preprocessing"]["target_shape"],
        "overwrite": overwrite,
    }
    write_json(output_dir(config, "qc", "preprocessing_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    if summary["passed_or_cached"] < 750 and not args.limit:
        raise RuntimeError(f"Too few preprocessed MRI caches: {summary['passed_or_cached']} < 750")


if __name__ == "__main__":
    main()
