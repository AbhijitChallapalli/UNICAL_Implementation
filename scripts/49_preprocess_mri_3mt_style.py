from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, local_path, output_dir, write_json
from src.mri_3mt_style import preprocess_mri_3mt_style


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
        print(f"[preprocess-3mt] warning: failed QC example {path}: {exc}")


def build_manifest(config: dict) -> pd.DataFrame:
    existing = cfg_path(config, "mri_ad_cn_manifest_csv")
    if existing.is_file():
        frame = pd.read_csv(existing)
    else:
        labels = pd.read_csv(cfg_path(config, "label_csv"))
        frame = labels.copy()
    if "label_id" not in frame.columns:
        frame["label_id"] = frame["label"].map({"CN": 0, "AD": 1}).astype(int)
    if "split_group_id" not in frame.columns:
        frame["split_group_id"] = frame["subject_id"].astype(str)
    if "mri_nifti_path" not in frame.columns:
        path_cols = [c for c in frame.columns if "path" in c.lower() and ("mri" in c.lower() or "nifti" in c.lower())]
        if path_cols:
            frame["mri_nifti_path"] = frame[path_cols[0]]
        else:
            frame["mri_nifti_path"] = frame["sample_id"].astype(str).map(lambda sid: str(local_path(config, config["paths"]["mri_root"]) / f"{sid}.nii.gz"))
    out = cfg_path(config, "cmt_manifest_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AD/CN MRI volumes with 3MT-style fallbacks.")
    parser.add_argument("--config", default="configs/oasis3_unical_cmt_3mt_style.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-registration", action="store_true")
    parser.add_argument("--debug-shape", action="store_true")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    frame = build_manifest(config)
    if args.limit:
        frame = frame.head(args.limit).copy()
    cache_dir = cfg_path(config, "cmt_cache_dir")
    overwrite = bool(args.overwrite or config["preprocessing_3mt_style"].get("overwrite", False))
    rows = []
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        mri_out = cache_dir / f"{sample_id}_mri_3mt.npy"
        mask_out = cache_dir / f"{sample_id}_mask_3mt.npy"
        rec = row.to_dict()
        rec.update({
            "mri_3mt_cache_path": str(mri_out),
            "mask_3mt_cache_path": str(mask_out),
            "preprocessing_3mt_status": "failed",
            "preprocessing_3mt_error": "",
            "used_skull_strip_method": "",
            "used_bias_correction_method": "",
            "used_registration_method": "",
            "original_shape": "",
            "final_shape": "",
            "voxel_spacing": "",
        })
        try:
            if mri_out.is_file() and mask_out.is_file() and not overwrite:
                rec["preprocessing_3mt_status"] = "cached"
                rec["final_shape"] = json.dumps(tuple(int(v) for v in np.load(mri_out, mmap_mode="r").shape))
            else:
                result = preprocess_mri_3mt_style(row["mri_nifti_path"], config, args.skip_registration, args.debug_shape)
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(mri_out, result.pop("mri"))
                np.save(mask_out, result.pop("mask"))
                rec.update(result)
                rec["preprocessing_3mt_status"] = "passed"
        except Exception as exc:
            rec["preprocessing_3mt_error"] = str(exc)
            if config.get("runtime", {}).get("fail_fast", False) and not args.limit:
                print(f"[preprocess-3mt] failed {sample_id}: {exc}")
        rows.append(rec)
    out = pd.DataFrame(rows)
    out_path = cfg_path(config, "cmt_preprocessed_manifest_csv")
    qc_path = cfg_path(config, "cmt_qc_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    out[["sample_id", "subject_id", "label", "label_id", "preprocessing_3mt_status", "preprocessing_3mt_error"]].to_csv(qc_path, index=False)
    passed = out[out["preprocessing_3mt_status"].isin(["passed", "cached"])]
    if config["preprocessing_3mt_style"].get("save_qc_examples", True):
        for label, subset in passed.groupby("label"):
            for _, ex in subset.head(int(config["preprocessing_3mt_style"].get("qc_examples_per_class", 12))).iterrows():
                save_example(np.load(ex["mri_3mt_cache_path"]), output_dir(config, "qc", "preprocessing_examples", f"{label}_{ex['sample_id']}.png"))
    summary = {"samples": int(len(out)), "passed_or_cached": int(len(passed)), "failed": int((out["preprocessing_3mt_status"] == "failed").sum())}
    write_json(output_dir(config, "qc", "preprocessing_3mt_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
