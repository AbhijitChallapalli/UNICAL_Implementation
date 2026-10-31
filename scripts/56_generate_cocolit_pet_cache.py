from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(config_path: Path) -> tuple[dict, Path, Path, Path]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    paths = config.get("paths", {})
    candidates = [
        repo_path(paths[key])
        for key in ("cmt_manifest_csv", "mri_ad_cn_manifest_csv")
        if paths.get(key)
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "No AD/CN MRI manifest found. Expected one of: "
            + ", ".join(str(path) for path in candidates)
            + ". Run the MRI manifest/preprocessing stage first."
        )
    output_root = repo_path(
        paths.get(
            "synthetic_pet_k_output_dir",
            config.get("synthetic_pet", {}).get("k_output_dir", "outputs/synthetic_pet_k_ad_cn"),
        )
    )
    manifest = repo_path(
        paths.get(
            "synthetic_pet_k_manifest",
            config.get("synthetic_pet", {}).get(
                "k_manifest_csv", "data/processed/synthetic_pet_k_manifest_ad_cn.csv"
            ),
        )
    )
    return config, source, output_root, manifest


def load_rows(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, keep_default_na=False)
    if "manifest_status" in frame.columns:
        frame = frame[frame["manifest_status"].isin(["usable", "passed", "cached"])].copy()
    if "sample_id" not in frame.columns:
        raise KeyError(f"{source} does not contain sample_id")
    input_column = next(
        (name for name in ("mri_nifti_path", "mri_path") if name in frame.columns), None
    )
    if input_column is None:
        raise KeyError(f"{source} does not contain mri_nifti_path or mri_path")
    frame = frame[frame[input_column].astype(str).str.len() > 0].copy()
    frame["_cocolit_input"] = frame[input_column].map(repo_path)
    return frame.reset_index(drop=True)


def valid_nifti(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        data = np.asanyarray(nib.load(path).dataobj)
        return data.ndim >= 3 and data.size > 0 and np.isfinite(data).all() and np.count_nonzero(data) > 0
    except Exception:
        return False


def cached_sample(
    sample_id: str, sample_index: int, roots: list[Path], validator=valid_nifti
) -> Path | None:
    # The older binary pipeline used OAS30024_d0084 while the AD/CN manifest
    # uses OAS30024_MR_d0084 for the same MRI session.
    sample_aliases = list(dict.fromkeys([sample_id, sample_id.replace("_MR_", "_")]))
    for root in roots:
        for cache_id in sample_aliases:
            candidate = root / cache_id / f"sample_{sample_index:02d}.nii.gz"
            if validator(candidate):
                return candidate
    return None


def compact_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def update_manifest(sample_id: str, sample_paths: list[Path], manifest: Path, k: int) -> int:
    import fcntl  # Linux/HPC-only advisory locking for concurrent array tasks.

    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest.with_suffix(manifest.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        columns = ["sample_id", *[f"sample_{idx:02d}" for idx in range(k)]]
        current = (
            pd.read_csv(manifest, keep_default_na=False)
            if manifest.is_file() and manifest.stat().st_size > 0
            else pd.DataFrame(columns=columns)
        )
        current = current.reindex(columns=columns)
        current = current[current["sample_id"].astype(str) != sample_id]
        record = pd.DataFrame(
            [{
                "sample_id": sample_id,
                **{f"sample_{idx:02d}": str(path.resolve()) for idx, path in enumerate(sample_paths)},
            }]
        )
        current = pd.concat([current, record], ignore_index=True).sort_values("sample_id")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=manifest.stem + ".", dir=manifest.parent,
            delete=False, encoding="utf-8", newline=""
        ) as handle:
            temporary = Path(handle.name)
            current.to_csv(handle, index=False)
        os.replace(temporary, manifest)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return len(current)


def rebuild_manifest_from_cache(records: list[dict[str, str]], manifest: Path, k: int) -> None:
    import fcntl

    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest.with_suffix(manifest.suffix + ".lock")
    columns = ["sample_id", *[f"sample_{idx:02d}" for idx in range(k)]]
    frame = pd.DataFrame(records, columns=columns).sort_values("sample_id")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=manifest.stem + ".", dir=manifest.parent,
            delete=False, encoding="utf-8", newline=""
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary, manifest)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one array shard of the AD/CN CoCoLIT K-sample cache.")
    parser.add_argument("--config", default="configs/oasis3_unical_cmt_3mt_style.yaml")
    parser.add_argument("--index", type=int, help="Zero-based row selected by SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--m", type=int, default=64, help="CoCoLIT LAS samples; 64 reproduces the paper setting.")
    parser.add_argument("--ddim-n", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument(
        "--legacy-output-root", default="outputs/synthetic_pet_k",
        help="Existing K=4 cache checked before generating a new sample.",
    )
    parser.add_argument(
        "--missing-indices", action="store_true",
        help="Print a compact SLURM array expression for subjects without four cached samples.",
    )
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config, source, output_root, manifest = load_settings(config_path)
    rows = load_rows(source)
    k = args.k or int(config.get("synthetic_pet", {}).get("required_k", 4))
    cache_roots = list(dict.fromkeys([output_root.resolve(), repo_path(args.legacy_output_root).resolve()]))
    if args.missing_indices:
        missing: list[int] = []
        records: list[dict[str, str]] = []
        for row_index, sample_id in enumerate(rows["sample_id"].astype(str)):
            sample_paths = [
                cached_sample(sample_id, sample_index, cache_roots, valid_nifti)
                for sample_index in range(k)
            ]
            if any(path is None for path in sample_paths):
                missing.append(row_index)
            else:
                records.append({
                    "sample_id": sample_id,
                    **{
                        f"sample_{sample_index:02d}": str(path.resolve())
                        for sample_index, path in enumerate(sample_paths)
                        if path is not None
                    },
                })
        rebuild_manifest_from_cache(records, manifest, k)
        print(compact_indices(missing))
        return
    if args.index is None:
        parser.error("--index is required unless --missing-indices is used")
    if args.index < 0 or args.index >= len(rows):
        print(f"[cocolit-cache] index {args.index} is outside manifest rows 0..{len(rows) - 1}; nothing to do.")
        return
    row = rows.iloc[args.index]
    sample_id = str(row["sample_id"])
    input_path = Path(row["_cocolit_input"])
    if not input_path.is_file():
        raise FileNotFoundError(f"MRI input not found for {sample_id}: {input_path}")
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir = sample_dir / "cocolit_preprocess"
    preprocessed_input = preprocess_dir / "norm_mni_space.nii.gz"
    print(
        f"[cocolit-cache] row={args.index}/{len(rows) - 1} sample_id={sample_id} "
        f"input={input_path} k={k} m={args.m} ddim_n={args.ddim_n} "
        f"cache_roots={[str(root) for root in cache_roots]}",
        flush=True,
    )
    selected_paths: list[Path] = []
    for idx in range(k):
        output_path = sample_dir / f"sample_{idx:02d}.nii.gz"
        existing = None if args.force else cached_sample(sample_id, idx, cache_roots)
        if existing is not None:
            selected_paths.append(existing)
            print(f"[cocolit-cache] cached {existing}", flush=True)
            continue
        executable = shutil.which("mri2pet")
        if executable is None:
            raise FileNotFoundError("mri2pet is not on PATH; activate the environment where cocolit is installed.")
        partial_path = sample_dir / f"sample_{idx:02d}.partial.nii.gz"
        partial_path.unlink(missing_ok=True)
        synth_input = preprocessed_input if preprocessed_input.is_file() else input_path
        command = [
            executable, "--i", str(synth_input), "--o", str(partial_path), "--device", "cuda",
            "--m", str(args.m), "--ddim_n", str(args.ddim_n),
        ]
        if not args.no_preprocess and synth_input == input_path:
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            command += ["--preprocess", "--preprocess_out", str(preprocess_dir)]
        print(f"[cocolit-cache] generating sample {idx + 1}/{k}", flush=True)
        subprocess.run(command, check=True)
        if not valid_nifti(partial_path):
            raise RuntimeError(f"CoCoLIT returned an invalid output: {partial_path}")
        os.replace(partial_path, output_path)
        selected_paths.append(output_path)

    if len(selected_paths) != k or not all(valid_nifti(path) for path in selected_paths):
        raise RuntimeError(f"The K={k} cache is incomplete for {sample_id}")
    completed = update_manifest(sample_id, selected_paths, manifest, k)
    print(f"[cocolit-cache] complete={completed}/{len(rows)} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
