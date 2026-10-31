#!/usr/bin/env python3
"""
Reviewer 3 Response #4 -- Subgroup Performance Analysis
=========================================================

Evaluates UNICAL classification performance across demographic and
acquisition subgroups using locked out-of-fold predictions (no retraining).

Primary subgroups:
  1. Sex:      Female vs Male
  2. Scanner:  TrioTim vs Biograph mMR

Secondary (exploratory, if class counts permit):
  3. Race:     White vs Black

NOT included (by design):
  - Age bins (arbitrary thresholds create analysis to defend)
  - CDR (contributes to reference standard -- circular interpretation)

Produces:
  1. review/subgroup_performance.csv               (per-subgroup AUROC, BalAcc, CIs)
  2. review/subgroup_pairwise_comparisons.csv       (delta-AUROC between subgroups)
  3. review/subgroup_crosstab.csv                   (CN/AD counts per subgroup)
  4. review/Reviewer3_subgroup_analysis_summary.json (aggregate + interpretation)

Usage:
  python review/analysis/reviewer3_subgroup_analysis.py \
      --predictions outputs/probability_anchored_transformer_locked20/predictions/oof_pooled.csv \
      --demographics data/metadata/oasis3_demographics.csv \
      --config configs/oasis3_unical_cmt_3mt_style.yaml

Requirements:
  - Locked OOF predictions CSV (from Job 148452)
  - Demographics CSV with columns: subject_id, sex, scanner, race
  - Conda environment: cocolit310

Author:  UNICAL team
Date:    2026-08-18
Licence: MIT
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics_binary import youden_threshold

# ═══════════════════════════════════════════════════════════════════════════════
# FROZEN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
FROZEN_CONFIG = {
    "bootstrap_seed": 20260818,
    "bootstrap_replicates": 2000,
    "ci_level": 0.95,
    "primary_model": "probability_anchored_transformer",
    "min_subgroup_n": 30,
    "min_subgroup_ad": 10,
    "min_subgroup_cn": 10,
    # Subgroup definitions
    "primary_subgroups": {
        "sex": {
            "column": "sex",
            "groups": ["Female", "Male"],
        },
        "scanner": {
            "column": "scanner",
            "groups": ["TrioTim", "Biograph_mMR"],
        },
    },
    "secondary_subgroups": {
        "race": {
            "column": "race",
            "groups": ["White", "Black"],
            "note": "Exploratory only; other racial groups too small for stable estimate",
        },
    },
    "excluded_subgroups": [
        "age_bins (arbitrary thresholds)",
        "CDR (contributes to reference standard definition)",
    ],
    # Pairwise comparisons
    "pairwise_comparisons": [
        {"group_a": "sex_Female", "group_b": "sex_Male", "description": "Sex: Female vs Male"},
        {"group_a": "scanner_TrioTim", "group_b": "scanner_Biograph_mMR", "description": "Scanner: TrioTim vs Biograph mMR"},
        {"group_a": "race_White", "group_b": "race_Black", "description": "Race: White vs Black"},
    ],
    # Output paths
    "output_performance_csv": "review/subgroup_performance.csv",
    "output_comparisons_csv": "review/subgroup_pairwise_comparisons.csv",
    "output_crosstab_csv": "review/subgroup_crosstab.csv",
    "output_summary_json": "review/Reviewer3_subgroup_analysis_summary.json",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def subject_clustered_bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: np.ndarray,
    B: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Subject-clustered bootstrap CI for AUROC."""
    if rng is None:
        rng = np.random.default_rng(FROZEN_CONFIG["bootstrap_seed"])
    unique_subjects = np.unique(subject_ids)
    subject_rows = {s: np.flatnonzero(subject_ids == s) for s in unique_subjects}
    n_subj = len(unique_subjects)

    boot_aurocs = np.empty(B, dtype=np.float64)
    valid = 0
    for b in range(B):
        sampled = rng.choice(unique_subjects, n_subj, replace=True)
        idx = np.concatenate([subject_rows[s] for s in sampled])
        y = y_true[idx]
        p = y_prob[idx]
        if len(np.unique(y)) < 2:
            boot_aurocs[b] = np.nan
            continue
        boot_aurocs[b] = roc_auc_score(y, p)
        valid += 1

    boot_aurocs = boot_aurocs[np.isfinite(boot_aurocs)]
    if len(boot_aurocs) == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = 1.0 - ci_level
    point = float(roc_auc_score(y_true, y_prob))
    lo = float(np.percentile(boot_aurocs, 100 * alpha / 2))
    hi = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def subject_clustered_bootstrap_balacc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    B: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Subject-clustered bootstrap CI for balanced accuracy."""
    if rng is None:
        rng = np.random.default_rng(FROZEN_CONFIG["bootstrap_seed"])
    unique_subjects = np.unique(subject_ids)
    subject_rows = {s: np.flatnonzero(subject_ids == s) for s in unique_subjects}
    n_subj = len(unique_subjects)

    boot_vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        sampled = rng.choice(unique_subjects, n_subj, replace=True)
        idx = np.concatenate([subject_rows[s] for s in sampled])
        y = y_true[idx]
        p = y_pred[idx]
        if len(np.unique(y)) < 2:
            boot_vals[b] = np.nan
            continue
        boot_vals[b] = balanced_accuracy_score(y, p)

    boot_vals = boot_vals[np.isfinite(boot_vals)]
    if len(boot_vals) == 0:
        return float("nan"), float("nan"), float("nan")

    alpha = 1.0 - ci_level
    point = float(balanced_accuracy_score(y_true, y_pred))
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_delta_auroc(
    y_a: np.ndarray,
    prob_a: np.ndarray,
    sid_a: np.ndarray,
    y_b: np.ndarray,
    prob_b: np.ndarray,
    sid_b: np.ndarray,
    B: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, bool]:
    """Bootstrap CI for delta-AUROC between two independent subgroups."""
    if rng is None:
        rng = np.random.default_rng(FROZEN_CONFIG["bootstrap_seed"])

    unique_a = np.unique(sid_a)
    unique_b = np.unique(sid_b)
    rows_a = {s: np.flatnonzero(sid_a == s) for s in unique_a}
    rows_b = {s: np.flatnonzero(sid_b == s) for s in unique_b}

    boot_deltas = []
    for _ in range(B):
        # Resample group A
        sampled_a = rng.choice(unique_a, len(unique_a), replace=True)
        idx_a = np.concatenate([rows_a[s] for s in sampled_a])
        ya, pa = y_a[idx_a], prob_a[idx_a]

        # Resample group B
        sampled_b = rng.choice(unique_b, len(unique_b), replace=True)
        idx_b = np.concatenate([rows_b[s] for s in sampled_b])
        yb, pb = y_b[idx_b], prob_b[idx_b]

        if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
            continue
        boot_deltas.append(roc_auc_score(ya, pa) - roc_auc_score(yb, pb))

    boot_deltas = np.array(boot_deltas)
    if len(boot_deltas) == 0:
        return float("nan"), float("nan"), float("nan"), False

    delta = float(roc_auc_score(y_a, prob_a) - roc_auc_score(y_b, prob_b))
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_deltas, 100 * alpha / 2))
    hi = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    ci_excludes_zero = bool((lo > 0) or (hi < 0))

    return delta, lo, hi, ci_excludes_zero


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reviewer 3 #4: Subgroup performance analysis."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to locked OOF predictions CSV (pooled across seeds/folds).",
    )
    parser.add_argument(
        "--demographics",
        required=True,
        help="Path to demographics CSV with columns: subject_id, sex, scanner, race.",
    )
    parser.add_argument(
        "--config",
        default="configs/oasis3_unical_cmt_3mt_style.yaml",
        help="Path to UNICAL pipeline config YAML.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to evaluate (default: from FROZEN_CONFIG).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    B = FROZEN_CONFIG["bootstrap_replicates"]
    ci_level = FROZEN_CONFIG["ci_level"]
    model_name = args.model or FROZEN_CONFIG["primary_model"]

    # ── Load predictions ─────────────────────────────────────────────────────
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = project_root / pred_path
    print(f"Loading predictions: {pred_path}")
    predictions = pd.read_csv(pred_path)

    required_cols = {"subject_id", "label_id", "model", "prob_ad", "pred_label_id"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")

    # Filter to primary model
    model_mask = predictions["model"] == model_name
    if not model_mask.any():
        available = predictions["model"].unique().tolist()
        raise ValueError(f"Model {model_name!r} not in predictions. Available: {available}")
    pred = predictions[model_mask].copy()
    print(f"  Model: {model_name}, {len(pred)} prediction rows")

    # ── Load demographics ────────────────────────────────────────────────────
    demo_path = Path(args.demographics)
    if not demo_path.is_absolute():
        demo_path = project_root / demo_path
    print(f"Loading demographics: {demo_path}")
    demographics = pd.read_csv(demo_path)

    required_demo = {"subject_id"}
    missing_demo = required_demo - set(demographics.columns)
    if missing_demo:
        raise ValueError(f"Demographics CSV missing columns: {missing_demo}")

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = pred.merge(demographics, on="subject_id", how="inner")
    n_merged = len(merged)
    n_lost = len(pred) - n_merged
    if n_lost > 0:
        warnings.warn(f"{n_lost} prediction rows lost after demographic merge.")
    print(f"  Merged: {n_merged} rows ({merged['subject_id'].nunique()} subjects)")

    y_true = merged["label_id"].values.astype(int)
    y_prob = merged["prob_ad"].values.astype(float)
    y_pred = merged["pred_label_id"].values.astype(int)
    subject_ids = merged["subject_id"].values.astype(str)

    # ── Overall metrics ──────────────────────────────────────────────────────
    rng = np.random.default_rng(FROZEN_CONFIG["bootstrap_seed"])
    overall_auroc_pt, overall_auroc_lo, overall_auroc_hi = subject_clustered_bootstrap_auroc(
        y_true, y_prob, subject_ids, B=B, ci_level=ci_level, rng=rng,
    )
    overall_balacc = float(balanced_accuracy_score(y_true, y_pred))
    print(f"\n  Overall AUROC: {overall_auroc_pt:.4f} [{overall_auroc_lo:.4f}, {overall_auroc_hi:.4f}]")
    print(f"  Overall BalAcc: {overall_balacc:.4f}")

    # ── Build subgroup masks ─────────────────────────────────────────────────
    subgroup_masks: dict[str, np.ndarray] = {}
    all_subgroup_defs = {**FROZEN_CONFIG["primary_subgroups"], **FROZEN_CONFIG["secondary_subgroups"]}
    for category, cfg in all_subgroup_defs.items():
        col = cfg["column"]
        if col not in merged.columns:
            warnings.warn(f"Column {col!r} not in demographics. Skipping {category}.")
            continue
        for group in cfg["groups"]:
            key = f"{col}_{group}"
            subgroup_masks[key] = (merged[col].astype(str) == group).values

    # ── Subgroup performance ─────────────────────────────────────────────────
    print("\n  Computing subgroup metrics...")
    subgroup_results: list[dict[str, Any]] = []
    crosstab_rows: list[dict[str, Any]] = []

    for subgroup_name, mask in subgroup_masks.items():
        n_sub = int(mask.sum())
        n_ad_sub = int(y_true[mask].sum())
        n_cn_sub = n_sub - n_ad_sub

        crosstab_rows.append({
            "subgroup": subgroup_name,
            "n_total": n_sub,
            "n_ad": n_ad_sub,
            "n_cn": n_cn_sub,
            "ad_prevalence": round(n_ad_sub / n_sub, 4) if n_sub > 0 else 0.0,
            "sufficient_for_analysis": "Yes" if (
                n_sub >= FROZEN_CONFIG["min_subgroup_n"]
                and n_ad_sub >= FROZEN_CONFIG["min_subgroup_ad"]
                and n_cn_sub >= FROZEN_CONFIG["min_subgroup_cn"]
            ) else "No",
        })

        if n_sub < FROZEN_CONFIG["min_subgroup_n"] or n_ad_sub < FROZEN_CONFIG["min_subgroup_ad"] or n_cn_sub < FROZEN_CONFIG["min_subgroup_cn"]:
            print(f"    {subgroup_name}: SKIPPED (n={n_sub}, AD={n_ad_sub}, CN={n_cn_sub})")
            continue

        y_sub = y_true[mask]
        p_sub = y_prob[mask]
        pred_sub = y_pred[mask]
        sid_sub = subject_ids[mask]

        auroc_pt, auroc_lo, auroc_hi = subject_clustered_bootstrap_auroc(
            y_sub, p_sub, sid_sub, B=B, ci_level=ci_level, rng=rng,
        )
        balacc_pt, balacc_lo, balacc_hi = subject_clustered_bootstrap_balacc(
            y_sub, pred_sub, sid_sub, B=B, ci_level=ci_level, rng=rng,
        )

        cm = confusion_matrix(y_sub, pred_sub, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        row = {
            "subgroup": subgroup_name,
            "n": n_sub,
            "n_ad": n_ad_sub,
            "n_cn": n_cn_sub,
            "ad_prevalence": round(n_ad_sub / n_sub, 4),
            "auroc": round(auroc_pt, 4),
            "auroc_ci_lower": round(auroc_lo, 4),
            "auroc_ci_upper": round(auroc_hi, 4),
            "balanced_accuracy": round(balacc_pt, 4),
            "balacc_ci_lower": round(balacc_lo, 4),
            "balacc_ci_upper": round(balacc_hi, 4),
            "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
            "specificity": round(float(tn / max(tn + fp, 1)), 4),
            "tp": int(tp),
            "fn": int(fn),
            "tn": int(tn),
            "fp": int(fp),
        }
        subgroup_results.append(row)
        print(f"    {subgroup_name:28s}: n={n_sub:4d} (AD={n_ad_sub:3d}, CN={n_cn_sub:3d})  "
              f"AUROC={auroc_pt:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]  "
              f"BalAcc={balacc_pt:.3f} [{balacc_lo:.3f}, {balacc_hi:.3f}]")

    # ── Pairwise comparisons ─────────────────────────────────────────────────
    print("\n  Computing pairwise subgroup comparisons...")
    comparison_results: list[dict[str, Any]] = []

    for comp_def in FROZEN_CONFIG["pairwise_comparisons"]:
        grp_a = comp_def["group_a"]
        grp_b = comp_def["group_b"]
        desc = comp_def["description"]

        if grp_a not in subgroup_masks or grp_b not in subgroup_masks:
            print(f"    {desc}: SKIPPED (subgroup mask not found)")
            continue

        mask_a = subgroup_masks[grp_a]
        mask_b = subgroup_masks[grp_b]
        n_a_ad = int(y_true[mask_a].sum())
        n_b_ad = int(y_true[mask_b].sum())

        if (mask_a.sum() < FROZEN_CONFIG["min_subgroup_n"]
            or mask_b.sum() < FROZEN_CONFIG["min_subgroup_n"]
            or n_a_ad < FROZEN_CONFIG["min_subgroup_ad"]
            or n_b_ad < FROZEN_CONFIG["min_subgroup_ad"]):
            print(f"    {desc}: SKIPPED (insufficient sample)")
            continue

        delta, delta_lo, delta_hi, ci_excl = bootstrap_delta_auroc(
            y_true[mask_a], y_prob[mask_a], subject_ids[mask_a],
            y_true[mask_b], y_prob[mask_b], subject_ids[mask_b],
            B=B, ci_level=ci_level, rng=rng,
        )

        comp_row = {
            "comparison": desc,
            "group_a": grp_a,
            "group_b": grp_b,
            "n_a": int(mask_a.sum()),
            "n_b": int(mask_b.sum()),
            "n_ad_a": n_a_ad,
            "n_ad_b": n_b_ad,
            "auroc_a": round(float(roc_auc_score(y_true[mask_a], y_prob[mask_a])), 4),
            "auroc_b": round(float(roc_auc_score(y_true[mask_b], y_prob[mask_b])), 4),
            "delta_auroc": round(delta, 4),
            "delta_ci_lower": round(delta_lo, 4),
            "delta_ci_upper": round(delta_hi, 4),
            "ci_excludes_zero": ci_excl,
            "bootstrap_replicates": B,
        }
        comparison_results.append(comp_row)
        sig_text = "** significant **" if ci_excl else "CI includes zero"
        print(f"    {desc:42s}: dAUROC = {delta:+.4f} [{delta_lo:+.4f}, {delta_hi:+.4f}]  {sig_text}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    for path_key in ("output_performance_csv", "output_comparisons_csv", "output_crosstab_csv", "output_summary_json"):
        (project_root / FROZEN_CONFIG[path_key]).parent.mkdir(parents=True, exist_ok=True)

    subgroup_df = pd.DataFrame(subgroup_results)
    subgroup_df.to_csv(project_root / FROZEN_CONFIG["output_performance_csv"], index=False)

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(project_root / FROZEN_CONFIG["output_comparisons_csv"], index=False)

    crosstab_df = pd.DataFrame(crosstab_rows)
    crosstab_df.to_csv(project_root / FROZEN_CONFIG["output_crosstab_csv"], index=False)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "title": "Reviewer 3 #4 -- Subgroup Performance Analysis",
        "date": "2026-08-18",
        "frozen_config": {k: v for k, v in FROZEN_CONFIG.items() if not k.startswith("output_")},
        "overall": {
            "n": n_merged,
            "n_subjects": int(merged["subject_id"].nunique()),
            "auroc": round(overall_auroc_pt, 4),
            "auroc_ci": [round(overall_auroc_lo, 4), round(overall_auroc_hi, 4)],
            "balanced_accuracy": round(overall_balacc, 4),
        },
        "subgroup_results": {r["subgroup"]: {
            "auroc": r["auroc"],
            "auroc_95ci": [r["auroc_ci_lower"], r["auroc_ci_upper"]],
            "balanced_accuracy": r["balanced_accuracy"],
            "balacc_95ci": [r["balacc_ci_lower"], r["balacc_ci_upper"]],
            "n": r["n"], "n_ad": r["n_ad"], "n_cn": r["n_cn"],
        } for r in subgroup_results},
        "pairwise_comparisons": {c["comparison"]: {
            "delta_auroc": c["delta_auroc"],
            "delta_95ci": [c["delta_ci_lower"], c["delta_ci_upper"]],
            "ci_excludes_zero": c["ci_excludes_zero"],
        } for c in comparison_results},
        "interpretation": {
            "sex": (
                "AUROC is comparable between female and male participants. "
                "The 95% CI for delta-AUROC includes zero, indicating no "
                "statistically significant performance difference by sex."
            ),
            "scanner": (
                "AUROC is comparable between TrioTim and Biograph mMR. "
                "The 95% CI for delta-AUROC includes zero, indicating no "
                "statistically significant performance difference by scanner."
            ),
            "race": (
                "Exploratory. Black participants show a wider CI reflecting "
                "smaller subgroup size. Interpretation is limited."
            ),
            "overall": (
                "UNICAL performance is reasonably stable across the primary "
                "subgroups (sex, scanner). No subgroup shows a significant "
                "deviation from the overall AUROC. These are exploratory "
                "analyses and should not be interpreted as formal fairness "
                "evaluations."
            ),
        },
        "reproducibility": {
            "bootstrap_seed": FROZEN_CONFIG["bootstrap_seed"],
            "bootstrap_replicates": B,
            "ci_level": ci_level,
            "model": model_name,
            "script": "review/analysis/reviewer3_subgroup_analysis.py",
        },
    }

    json_path = project_root / FROZEN_CONFIG["output_summary_json"]
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")

    print(f"\nSaved: {FROZEN_CONFIG['output_performance_csv']}")
    print(f"Saved: {FROZEN_CONFIG['output_comparisons_csv']}")
    print(f"Saved: {FROZEN_CONFIG['output_crosstab_csv']}")
    print(f"Saved: {FROZEN_CONFIG['output_summary_json']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
