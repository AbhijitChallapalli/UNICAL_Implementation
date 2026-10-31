from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, write_json


def load_npz(path: Path, prefix: str) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    x_key = "X" if "X" in data.files else "features" if "features" in data.files else "feature_vector"
    id_key = "sample_ids" if "sample_ids" in data.files else "sample_id"
    name_key = "feature_names" if "feature_names" in data.files else None
    x = np.asarray(data[x_key], dtype=np.float32)
    ids = [str(v) for v in data[id_key]]
    names = [str(v) for v in data[name_key]] if name_key else [f"{prefix}_{i:04d}" for i in range(x.shape[1])]
    return x, ids, [f"{prefix}__{name}" for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine MRI summary and CoWS-Rad features for fold-local CMT selection.")
    parser.add_argument("--config", default="configs/oasis3_unical_cmt_3mt_style.yaml")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    specs = []
    c = config.get("cowsrad_selected", {})
    if c.get("include_mri_summary", True):
        specs.append(("mri_summary", cfg_path(config, "mri_summary_npz")))
    if c.get("include_cowsrad_v1", True):
        specs.append(("cowsrad_v1", cfg_path(config, "cowsrad_v1_npz")))
    specs.append(("cowsrad_v2", cfg_path(config, "cowsrad_v2_npz")))
    loaded = []
    missing = []
    for prefix, path in specs:
        if path.is_file():
            loaded.append((prefix, *load_npz(path, prefix)))
        else:
            missing.append(str(path))
    if not loaded:
        raise FileNotFoundError(f"No source feature NPZ files found. Missing: {missing}")
    common_ids = set(loaded[0][2])
    for _, _, ids, _ in loaded[1:]:
        common_ids &= set(ids)
    sample_ids = sorted(common_ids)
    if not sample_ids:
        raise RuntimeError("Feature sources have no overlapping sample_id values.")
    matrices = []
    names: list[str] = []
    source_ranges = {}
    start = 0
    for prefix, x, ids, feature_names in loaded:
        lookup = {sid: i for i, sid in enumerate(ids)}
        aligned = x[[lookup[sid] for sid in sample_ids]]
        matrices.append(aligned)
        names.extend(feature_names)
        source_ranges[prefix] = [start, start + aligned.shape[1]]
        start += aligned.shape[1]
    X = np.hstack(matrices).astype(np.float32)
    out = cfg_path(config, "cowsrad_selected_npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, sample_ids=np.asarray(sample_ids), feature_names=np.asarray(names), source_ranges=json.dumps(source_ranges))
    manifest = {
        "output_npz": str(out),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "source_ranges": source_ranges,
        "missing_sources": missing,
        "supervised_selection": "not_performed_global_selection_is_fold_local",
    }
    write_json(out.parent / "cowsrad_selected_feature_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
