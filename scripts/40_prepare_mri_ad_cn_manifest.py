from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mri_ad_cn_utils import cfg_path, local_path, load_pipeline_config, output_dir, write_json


REQUIRED_ALIASES = {
    "subject_id": ["subject_id", "Subject", "subject", "OASISID", "participant_id"],
    "experiment_id": ["experiment_id", "ExperimentID", "mr_session", "session_id", "session"],
    "label": ["label", "diagnosis", "dx_label", "AD_CN_label"],
    "session_dir": ["session_dir", "mri_session_dir", "mr_session_dir", "path", "mri_dir"],
}
OPTIONAL_COLUMNS = ["sample_id", "mr_day", "cdr_day", "cdr_day_diff", "CDRTOT", "dx1", "dx1_group"]
PATTERNS = ["**/*T1w*.nii.gz", "**/*_T1w.nii.gz", "**/anat*/**/*.nii.gz"]


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    lower = {c.lower(): c for c in frame.columns}
    for canonical, aliases in REQUIRED_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                mapping[canonical] = alias
                break
            if alias.lower() in lower:
                mapping[canonical] = lower[alias.lower()]
                break
        if canonical not in mapping:
            raise ValueError(f"Could not resolve required column '{canonical}'. Available columns: {list(frame.columns)}")
    return mapping


def resolve_label_csv(config: dict) -> Path:
    local_csv = cfg_path(config, "label_csv")
    if local_csv.is_file():
        return local_csv
    hpc_csv = local_path(config, config["paths"]["hpc_label_csv"])
    if hpc_csv.is_file():
        return hpc_csv
    raise FileNotFoundError(
        "Clean AD/CN label CSV is missing. Copy it to "
        f"{local_csv} before running locally, or ensure the HPC fallback exists at {hpc_csv}."
    )


def find_t1w(session_dir: Path) -> tuple[list[Path], Path | None]:
    candidates: dict[str, Path] = {}
    for pattern in PATTERNS:
        for path in session_dir.glob(pattern):
            if path.is_file():
                candidates[str(path.resolve())] = path
    paths = list(candidates.values())
    if not paths:
        return [], None
    def score(path: Path) -> tuple[int, int, int]:
        name = path.name
        return (
            1 if name.endswith("_T1w.nii.gz") else 0,
            1 if "T1w" in name else 0,
            path.stat().st_size if path.exists() else 0,
        )
    selected = sorted(paths, key=score, reverse=True)[0]
    return sorted(paths), selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OASIS-3 clean MRI AD/CN manifest.")
    parser.add_argument("--config", default="configs/oasis3_mri_ad_cn_ablation_96.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-path-checks", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    label_csv = resolve_label_csv(config)
    frame = pd.read_csv(label_csv)
    if args.limit:
        frame = frame.head(args.limit).copy()
    columns = resolve_columns(frame)
    out = pd.DataFrame()
    for canonical, source in columns.items():
        out[canonical] = frame[source]
    for col in OPTIONAL_COLUMNS:
        if col in frame.columns:
            out[col] = frame[col]
    out["subject_id"] = out["subject_id"].astype(str)
    out["experiment_id"] = out["experiment_id"].astype(str)
    out["label"] = out["label"].astype(str).str.upper().str.strip()
    invalid_labels = sorted(set(out["label"]) - {"CN", "AD"})
    if invalid_labels:
        raise ValueError(f"Labels must be only CN or AD; found {invalid_labels}")
    out["label_id"] = out["label"].map({"CN": 0, "AD": 1}).astype(int)
    out["split_group_id"] = out["subject_id"]
    if "sample_id" not in out.columns:
        out["sample_id"] = out["experiment_id"]
    out["sample_id"] = out["sample_id"].astype(str)

    counts = out["label"].value_counts().to_dict()
    expected = config["labels"]["expected_counts"]
    print(f"[manifest] label_csv={label_csv}")
    print(f"[manifest] class_counts={json.dumps(counts, sort_keys=True)} expected={expected}")
    for label, expected_count in expected.items():
        actual = int(counts.get(label, 0))
        if abs(actual - int(expected_count)) > max(5, int(expected_count) * 0.05):
            print(f"[manifest] WARNING count mismatch for {label}: actual={actual} expected={expected_count}")

    statuses = []
    rows = []
    for _, row in out.iterrows():
        record = row.to_dict()
        session_dir = Path(str(record["session_dir"]))
        if not session_dir.is_absolute():
            session_dir = local_path(config, session_dir)
        record["session_dir"] = str(session_dir)
        candidates: list[Path] = []
        selected: Path | None = None
        error = ""
        if args.dry_run or args.skip_path_checks:
            status = "dry_run_path_not_checked"
        else:
            if not session_dir.is_dir():
                status = "missing_session_dir"
                error = f"Missing session_dir: {session_dir}"
            else:
                candidates, selected = find_t1w(session_dir)
                if selected is None:
                    status = "missing_t1w"
                    error = f"No T1w NIfTI found under {session_dir}"
                else:
                    status = "usable"
        record["t1w_candidates"] = "|".join(str(p) for p in candidates)
        record["n_t1w_candidates"] = len(candidates)
        record["mri_nifti_path"] = "" if selected is None else str(selected)
        record["selected_t1w_filename"] = "" if selected is None else selected.name
        record["manifest_status"] = status
        record["manifest_error"] = error
        statuses.append(status)
        rows.append(record)
    manifest = pd.DataFrame(rows)

    usable = manifest if args.dry_run or args.skip_path_checks else manifest[manifest["manifest_status"] == "usable"]
    summary = {
        "label_csv": str(label_csv),
        "rows": int(len(manifest)),
        "usable_rows": int(len(usable)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "status_counts": {k: int(v) for k, v in pd.Series(statuses).value_counts().to_dict().items()},
        "dry_run": bool(args.dry_run),
        "skip_path_checks": bool(args.skip_path_checks),
    }
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        if len(usable) < 750:
            raise RuntimeError(f"Too few usable rows for publishable run: {len(usable)} < 750")
        usable_counts = usable["label"].value_counts().to_dict()
        if usable_counts.get("CN", 0) < 250 or usable_counts.get("AD", 0) < 200:
            raise RuntimeError(f"Class counts too small after manifest filtering: {usable_counts}")
        out_path = cfg_path(config, "manifest_csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_path, index=False)
        excluded = manifest[manifest["manifest_status"] != "usable"]
        excluded.to_csv(output_dir(config, "qc", "manifest_excluded_rows.csv"), index=False)
        write_json(output_dir(config, "qc", "manifest_summary.json"), summary)
        print(f"[manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
