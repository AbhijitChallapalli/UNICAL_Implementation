from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.cowsrad import CoWSRadConfig, extract_cowsrad_from_path
from src.mri_ad_cn_utils import cfg_path, local_path, load_pipeline_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CoWS-Rad v1 features for clean MRI AD/CN manifest.")
    parser.add_argument("--config", default="configs/oasis3_mri_ad_cn_ablation_96.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    settings = config.get("cowsrad_v1", {})
    frame = pd.read_csv(cfg_path(config, "preprocessed_manifest_csv"))
    frame = frame[frame["preprocess_status"].isin(["passed", "cached"])].copy()
    if args.limit:
        frame = frame.head(args.limit).copy()

    csv_path = local_path(config, settings.get("output_csv", "features/mri_ad_cn_ablation_96/cowsrad_v1_features.csv"))
    npz_path = local_path(config, settings.get("output_npz", "features/mri_ad_cn_ablation_96/cowsrad_v1_features.npz"))
    manifest_path = local_path(config, settings.get("output_manifest", "features/mri_ad_cn_ablation_96/cowsrad_v1_manifest.json"))
    overwrite = bool(args.overwrite or settings.get("overwrite", False))
    if csv_path.is_file() and npz_path.is_file() and manifest_path.is_file() and not overwrite:
        print(f"[cowsrad-v1] outputs exist; use --overwrite to regenerate: {csv_path}")
        return

    cache_col = settings.get("input_cache_column", "mri_cache_96_path")
    resize_shape = tuple(int(x) for x in config["preprocessing"].get("target_shape", [96, 96, 96]))
    extractor_config = CoWSRadConfig(resize_shape=resize_shape)
    vectors = []
    for _, row in frame.iterrows():
        vector = extract_cowsrad_from_path(row[cache_col], extractor_config)
        vectors.append(np.nan_to_num(vector.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    x = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    feature_names = [f"cowsrad_v1_{i:04d}" for i in range(x.shape[1])]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = [c for c in ["sample_id", "subject_id", "experiment_id", "label", "label_id"] if c in frame.columns]
    out = frame[meta_cols].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(x, columns=feature_names)], axis=1)
    out.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        X=x,
        y=frame["label_id"].to_numpy(dtype=np.int64),
        sample_ids=frame["sample_id"].astype(str).to_numpy(),
        subject_ids=frame["subject_id"].astype(str).to_numpy(),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    manifest = {
        "samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "source": "src.features.cowsrad.extract_cowsrad_from_path",
        "preserve_original_feature_logic": True,
        "input_cache_column": cache_col,
        "resize_shape": list(resize_shape),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
