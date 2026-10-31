from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, output_dir, write_json


def inspect_nifti(path: str) -> dict:
    record = {"qc_status": "failed", "qc_error": ""}
    p = Path(path)
    try:
        if not p.is_file():
            raise FileNotFoundError(f"Missing MRI file: {p}")
        import nibabel as nib
        image = nib.load(str(p))
        canonical = nib.as_closest_canonical(image)
        data = np.asarray(canonical.get_fdata(), dtype=np.float32)
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D after squeeze, got {data.shape}")
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError("No finite voxels")
        foreground = finite & (np.abs(data) > 0)
        if not foreground.any():
            raise ValueError("Empty foreground")
        record.update({
            "qc_status": "passed",
            "file_size_bytes": int(p.stat().st_size),
            "original_shape": "x".join(str(x) for x in data.shape),
            "voxel_spacing": "x".join(f"{float(x):.5g}" for x in canonical.header.get_zooms()[:3]),
            "orientation": "".join(nib.aff2axcodes(canonical.affine)),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "foreground_fraction": float(foreground.mean()),
        })
    except Exception as exc:
        record["qc_error"] = str(exc)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="QC selected AD/CN MRI NIfTI files.")
    parser.add_argument("--config", default="configs/oasis3_mri_ad_cn_ablation_96.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    manifest = pd.read_csv(cfg_path(config, "manifest_csv"))
    manifest = manifest[manifest["manifest_status"].eq("usable")].copy()
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    rows = []
    for _, row in manifest.iterrows():
        rec = row.to_dict()
        rec.update(inspect_nifti(str(row["mri_nifti_path"])))
        rows.append(rec)
    out = pd.DataFrame(rows)
    qc_path = cfg_path(config, "qc_csv")
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(qc_path, index=False)
    failures = out[out["qc_status"] != "passed"]
    failures.to_csv(output_dir(config, "qc", "mri_qc_failures.csv"), index=False)
    summary = {
        "samples": int(len(out)),
        "passed": int((out["qc_status"] == "passed").sum()),
        "failed": int((out["qc_status"] != "passed").sum()),
        "status_counts": {k: int(v) for k, v in out["qc_status"].value_counts().to_dict().items()},
    }
    write_json(output_dir(config, "qc", "mri_qc_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    if summary["passed"] < 750 and not args.limit:
        raise RuntimeError(f"Too few QC-passed MRI files: {summary['passed']} < 750")


if __name__ == "__main__":
    main()
