from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.mri_summary import extract_mri_summary_features, load_cached_mask, load_cached_mri
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic MRI summary features.")
    parser.add_argument("--config", default="configs/oasis3_mri_ad_cn_ablation_96.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    frame = pd.read_csv(cfg_path(config, "preprocessed_manifest_csv"))
    frame = frame[frame["preprocess_status"].isin(["passed", "cached"])].copy()
    if args.limit:
        frame = frame.head(args.limit).copy()
    vectors = []
    feature_names: list[str] | None = None
    for _, row in frame.iterrows():
        volume = load_cached_mri(str(row["mri_cache_96_path"]))
        mask = load_cached_mask(str(row.get("brain_mask_cache_path", "")), volume)
        vector, names = extract_mri_summary_features(volume, mask)
        vectors.append(vector)
        feature_names = feature_names or names
    features = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    feature_dir = cfg_path(config, "feature_dir")
    feature_dir.mkdir(parents=True, exist_ok=True)
    csv_path = feature_dir / "mri_summary_features.csv"
    npz_path = feature_dir / "mri_summary_features.npz"
    manifest_path = feature_dir / "mri_summary_manifest.json"
    meta = frame[["sample_id", "subject_id", "label", "label_id"]].reset_index(drop=True)
    feature_frame = pd.DataFrame(features, columns=feature_names or [])
    pd.concat([meta, feature_frame], axis=1).to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        sample_id=meta["sample_id"].astype(str).to_numpy(),
        subject_id=meta["subject_id"].astype(str).to_numpy(),
        label_id=meta["label_id"].to_numpy(dtype=np.int64),
        feature_names=np.asarray(feature_names or [], dtype=object),
        feature_vector=features,
    )
    manifest = {"samples": int(features.shape[0]), "feature_dim": int(features.shape[1]), "feature_names": feature_names or []}
    write_json(manifest_path, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
