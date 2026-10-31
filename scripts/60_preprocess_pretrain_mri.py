#!/usr/bin/env python
"""Preprocess unlabeled OASIS-3 MRI volumes for SSL pretraining.

This script applies **exactly the same** 3MT-style preprocessing pipeline
used for the paper cohort (skull-strip → bias correction → affine registration
→ robust z-score normalisation → resize to target shape) so that the
pretrained encoder sees identically-distributed inputs.

Usage
-----
Single session (SLURM array task)::

    python scripts/60_preprocess_pretrain_mri.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml \\
        --index 42

All sessions sequentially::

    python scripts/60_preprocess_pretrain_mri.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml

Build manifest only (after all array tasks finish)::

    python scripts/60_preprocess_pretrain_mri.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml \\
        --finalize
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mri_3mt_style import preprocess_mri_3mt_style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _subject_from_session(session_id: str) -> str:
    """Return the OASIS subject identifier from an MRI experiment ID."""
    return str(session_id).split("_MR_", 1)[0]


def _resolve_path(config: dict, key: str) -> Path:
    p = Path(config["paths"][key])
    if not p.is_absolute():
        p = Path(config["paths"]["project_root"]) / p
    return p


def _assert_config_subset(expected: Any, actual: Any, path: str = "preprocessing_3mt_style") -> None:
    """Require every pretraining preprocessing option to match the paper config."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Paper compatibility mismatch at {path}: expected a mapping")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Paper config is missing {path}.{key}")
            _assert_config_subset(value, actual[key], f"{path}.{key}")
        return
    if expected != actual:
        raise ValueError(
            f"Paper compatibility mismatch at {path}: pretrain={expected!r}, paper={actual!r}"
        )


def _validate_paper_compatibility(config: dict) -> None:
    """Fail before preprocessing if registration/normalization differs from the paper run."""
    paper_path = _resolve_path(config, "paper_pipeline_config")
    if not paper_path.is_file():
        raise FileNotFoundError(f"Paper pipeline config not found: {paper_path}")
    paper = _load_config(str(paper_path))
    _assert_config_subset(
        config["preprocessing_3mt_style"],
        paper.get("preprocessing_3mt_style", {}),
    )


def _candidate_rank(path: Path) -> tuple[int, int, str]:
    """Rank T1 candidates deterministically, preferring explicitly named T1w files."""
    text = path.as_posix().lower()
    name = path.name.lower()
    if "t1w" in name:
        priority = 0
    elif "t1" in name:
        priority = 1
    elif "t1w" in text:
        priority = 2
    else:
        priority = 3
    return priority, len(path.parts), text


def _discover_sessions(config: dict) -> pd.DataFrame:
    """Discover all NIfTI files under ``pretrain_mri_root``.

    Returns a DataFrame with columns:
        session_id   – e.g. ``OAS30001_MR_d0129``
        nifti_path   – absolute path to .nii.gz
    """
    root = Path(config["paths"]["pretrain_mri_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Pretrain MRI root not found: {root}")

    rows: list[dict[str, str]] = []
    for nifti in sorted(root.rglob("*.nii.gz")):
        # Derive session_id from directory structure
        # Typical: root / OAS30001_MR_d0129 / anat1 / ... / T1w.nii.gz
        # The session directory is the first child of root
        rel = nifti.relative_to(root)
        session_id = rel.parts[0]
        rows.append({
            "session_id": session_id,
            "subject_id": _subject_from_session(session_id),
            "nifti_path": str(nifti.resolve()),
        })

    if not rows:
        raise FileNotFoundError(f"No .nii.gz files found under {root}")

    frame = pd.DataFrame(rows)

    # The download CSV is the authoritative leakage-safe session allow-list.
    allow_path = _resolve_path(config, "pretrain_csv")
    if not allow_path.is_file():
        raise FileNotFoundError(f"Pretrain session allow-list not found: {allow_path}")
    allow = pd.read_csv(allow_path)
    session_col = str(config.get("data_integrity", {}).get("allowlist_session_column", "experiment_id"))
    if session_col not in allow.columns:
        raise ValueError(f"Allow-list {allow_path} has no {session_col!r} column")
    allowed_sessions = allow[session_col].dropna().astype(str).drop_duplicates().tolist()
    allowed_set = set(allowed_sessions)
    frame = frame[frame["session_id"].isin(allowed_set)].copy()

    # Select one T1 volume per MRI session. Multiple series within a session are
    # technical repeats, whereas longitudinal MRI sessions are retained.
    frame["_rank"] = frame["nifti_path"].map(lambda value: _candidate_rank(Path(value)))
    frame = (
        frame.sort_values(["session_id", "_rank"])
        .drop_duplicates("session_id", keep="first")
        .drop(columns="_rank")
        .set_index("session_id")
        .reindex(allowed_sessions)
        .reset_index()
    )
    missing = frame[frame["nifti_path"].isna()]["session_id"].astype(str).tolist()
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} allow-listed sessions contain no NIfTI (first: {preview})"
        )

    # Defense in depth: an accidentally contaminated download directory must
    # never silently introduce a paper-cohort subject into SSL pretraining.
    paper_labels = _resolve_path(config, "paper_label_csv")
    if not paper_labels.is_file():
        raise FileNotFoundError(f"Paper label CSV not found: {paper_labels}")
    paper = pd.read_csv(paper_labels)
    if "subject_id" not in paper.columns:
        raise ValueError(f"Paper label CSV has no 'subject_id' column: {paper_labels}")
    overlap = sorted(set(frame["subject_id"].astype(str)) & set(paper["subject_id"].astype(str)))
    if overlap:
        raise RuntimeError(
            f"Leakage guard failed: {len(overlap)} pretraining subjects occur in the paper cohort "
            f"(first: {', '.join(overlap[:10])})"
        )

    if frame["session_id"].duplicated().any() or frame["nifti_path"].duplicated().any():
        raise RuntimeError("Pretraining discovery produced duplicate session or NIfTI rows")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Process one session
# ---------------------------------------------------------------------------

def process_one(
    row: pd.Series,
    config: dict,
    cache_dir: Path,
    overwrite: bool = False,
) -> dict:
    """Preprocess a single NIfTI and save to cache_dir as .npy."""
    session_id = str(row["session_id"])
    nifti_path = str(row["nifti_path"])
    out_path = cache_dir / f"{session_id}.npy"
    meta_path = cache_dir / f"{session_id}.meta.json"

    result = {
        "session_id": session_id,
        "subject_id": str(row["subject_id"]),
        "nifti_path": nifti_path,
        "cache_path": str(out_path),
        "status": "pending",
        "skull_strip_method": "",
        "bias_correction_method": "",
        "registration_method": "",
        "original_shape": "",
        "final_shape": "",
        "error": "",
    }

    if out_path.is_file() and not overwrite:
        # Reuse only a cache with provenance matching the deterministic source
        # selected by this version of the pipeline. Older race-produced caches
        # have no sidecar and are intentionally rebuilt.
        try:
            arr = np.load(out_path)
            expected_shape = tuple(int(v) for v in config["preprocessing_3mt_style"]["target_shape"])
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            same_source = Path(metadata["nifti_path"]).resolve() == Path(nifti_path).resolve()
            if same_source and tuple(arr.shape) == expected_shape and np.isfinite(arr).all():
                result["status"] = "cached"
                result["final_shape"] = json.dumps(list(arr.shape))
                result["skull_strip_method"] = metadata.get("skull_strip_method", "")
                result["bias_correction_method"] = metadata.get("bias_correction_method", "")
                result["registration_method"] = metadata.get("registration_method", "")
                return result
        except Exception:
            pass  # re-process

    try:
        output = preprocess_mri_3mt_style(nifti_path, config, skip_registration=False)
        vol = output["mri"]  # [D, H, W] float32

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vol)

        result["status"] = "passed"
        result["skull_strip_method"] = output.get("used_skull_strip_method", "")
        result["bias_correction_method"] = output.get("used_bias_correction_method", "")
        result["registration_method"] = output.get("used_registration_method", "")
        result["original_shape"] = output.get("original_shape", "")
        result["final_shape"] = output.get("final_shape", "")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "subject_id": str(row["subject_id"]),
                "nifti_path": str(Path(nifti_path).resolve()),
                "final_shape": list(vol.shape),
                "skull_strip_method": result["skull_strip_method"],
                "bias_correction_method": result["bias_correction_method"],
                "registration_method": result["registration_method"],
            }, f, indent=2)

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Finalize: collect per-task JSON → single manifest CSV
# ---------------------------------------------------------------------------

def finalize_manifest(config: dict) -> None:
    """Scan cache directory for result JSONs and build the master manifest."""
    cache_dir = _resolve_path(config, "pretrain_cache_dir")
    manifest_path = _resolve_path(config, "pretrain_manifest_csv")
    log_dir = _resolve_path(config, "pretrain_log_dir")

    sessions = _discover_sessions(config)
    rows: list[dict] = []
    for index, source in sessions.iterrows():
        jf = log_dir / f"preproc_result_{index:05d}.json"
        if jf.is_file():
            with open(jf, "r") as f:
                result = json.load(f)
            if str(result.get("session_id")) != str(source["session_id"]):
                raise RuntimeError(
                    f"Stale preprocessing result {jf}: expected {source['session_id']}, "
                    f"found {result.get('session_id')}"
                )
            rows.append(result)
            continue

        npy = cache_dir / f"{source['session_id']}.npy"
        meta = cache_dir / f"{source['session_id']}.meta.json"
        if npy.is_file() and meta.is_file():
            try:
                arr = np.load(npy)
                with open(meta, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                expected_shape = tuple(int(v) for v in config["preprocessing_3mt_style"]["target_shape"])
                if Path(metadata["nifti_path"]).resolve() != Path(source["nifti_path"]).resolve():
                    raise ValueError("cache source does not match selected NIfTI")
                if tuple(arr.shape) != expected_shape or not np.isfinite(arr).all():
                    raise ValueError(f"invalid cached array shape/values: {arr.shape}")
                rows.append({
                    "session_id": source["session_id"],
                    "subject_id": source["subject_id"],
                    "nifti_path": source["nifti_path"],
                    "cache_path": str(npy),
                    "status": "cached",
                    "final_shape": json.dumps(list(arr.shape)),
                    "error": "",
                })
                continue
            except Exception as exc:
                error = f"Failed to load cached npy: {exc}"
        else:
            error = "Missing preprocessing result or provenance-validated cache"
        rows.append({
            "session_id": source["session_id"],
            "subject_id": source["subject_id"],
            "nifti_path": source["nifti_path"],
            "cache_path": str(npy),
            "status": "missing",
            "error": error,
        })

    if not rows:
        print("ERROR: No preprocessing results found.", file=sys.stderr)
        sys.exit(1)

    frame = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)

    if frame["session_id"].duplicated().any() or frame["cache_path"].duplicated().any():
        raise RuntimeError("Final manifest contains duplicate sessions or cache paths")
    n_passed = int(frame["status"].isin(["passed", "cached"]).sum())
    n_failed = int(frame["status"].eq("failed").sum())
    print(f"Pretrain manifest saved: {manifest_path}")
    print(f"  Total rows:  {len(frame)}")
    print(f"  Passed:      {n_passed}")
    print(f"  Failed:      {n_failed}")
    for column in ("skull_strip_method", "bias_correction_method", "registration_method"):
        if column in frame.columns:
            print(f"  {column}: {frame[column].fillna('unknown').value_counts().to_dict()}")
    invalid = frame[~frame["status"].isin(["passed", "cached"])]
    if len(invalid) and bool(config.get("data_integrity", {}).get("require_all_sessions_preprocessed", True)):
        raise RuntimeError(
            f"Manifest is incomplete: {len(invalid)} of {len(frame)} allow-listed sessions are not usable. "
            "Inspect the failed/missing rows before pretraining."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess unlabeled MRI for SSL pretraining.")
    parser.add_argument("--config", default="configs/oasis3_pretrain_mri_encoder.yaml")
    parser.add_argument("--index", type=int, default=None,
                        help="Process only this row index (for SLURM array tasks).")
    parser.add_argument("--finalize", action="store_true",
                        help="Collect per-task results into a single manifest CSV.")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate paper compatibility/leakage and print the required array range.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process even if cached .npy exists.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many sessions (for testing).")
    args = parser.parse_args()

    config = _load_config(args.config)
    _validate_paper_compatibility(config)

    if args.finalize:
        finalize_manifest(config)
        return

    # Discover all NIfTI files
    sessions = _discover_sessions(config)
    print(f"Selected {len(sessions)} T1w NIfTI files across "
          f"{sessions['session_id'].nunique()} allow-listed sessions.")
    if args.preflight:
        print(f"Subjects: {sessions['subject_id'].nunique()}")
        print(f"Recommended SLURM array: 0-{len(sessions) - 1}")
        return

    if args.limit:
        sessions = sessions.head(args.limit)

    cache_dir = _resolve_path(config, "pretrain_cache_dir")
    log_dir = _resolve_path(config, "pretrain_log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    overwrite = args.overwrite or config["preprocessing_3mt_style"].get("overwrite", False)

    if args.index is not None:
        # Single array task mode
        if args.index >= len(sessions):
            print(f"Index {args.index} out of range (total {len(sessions)}). Nothing to do.")
            return
        row = sessions.iloc[args.index]
        print(f"[{args.index}/{len(sessions)}] Processing {row['session_id']} ...")
        result = process_one(row, config, cache_dir, overwrite)
        # Save per-task result
        result_path = log_dir / f"preproc_result_{args.index:05d}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Status: {result['status']}")
        if result["error"]:
            print(f"  Error:  {result['error']}")
    else:
        # Sequential mode: process all
        results: list[dict] = []
        for i, (_, row) in enumerate(sessions.iterrows()):
            print(f"[{i + 1}/{len(sessions)}] {row['session_id']} ... ", end="", flush=True)
            result = process_one(row, config, cache_dir, overwrite)
            results.append(result)
            # Save per-task result
            result_path = log_dir / f"preproc_result_{i:05d}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(result["status"])

        # Build manifest
        manifest_path = _resolve_path(config, "pretrain_manifest_csv")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(results)
        frame.to_csv(manifest_path, index=False)

        n_ok = int(frame["status"].isin(["passed", "cached"]).sum())
        n_fail = int(frame["status"].eq("failed").sum())
        print(f"\nPreprocessing complete.")
        print(f"  Total:   {len(frame)}")
        print(f"  Passed:  {n_ok}")
        print(f"  Failed:  {n_fail}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
