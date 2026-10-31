from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


PRIMARY_MODEL = "pet_anchored_transformer"
DECISION_BASELINES = ("pet_logistic", "concat_mlp", "average_all")
METRIC_NAMES = ("balanced_accuracy", "auroc", "auprc")
PAPER_NAMES = {
    "probability_anchored_transformer": "Probability-anchored adaptive transformer",
    "probability_anchored_transformer_no_uncertainty": "Anchored transformer without uncertainty gate",
    "probability_anchored_transformer_uniform_weights": "Anchored transformer with uniform weights",
    "pet_anchored_transformer": "PET-anchored residual transformer",
    "pet_anchored_transformer_no_uncertainty": "Transformer without uncertainty gate",
    "pet_anchored_transformer_no_modality_dropout": "Transformer without modality dropout",
    "average_all": "MRI + CoWS-Rad + PET average",
    "average_cowsrad_pet": "CoWS-Rad + PET average",
    "average_mri_pet": "MRI + PET average",
    "average_mri_cowsrad": "MRI + CoWS-Rad average",
    "concat_mlp": "Concatenation MLP",
    "pet_logistic": "PET logistic regression",
    "cowsrad_logistic": "CoWS-Rad logistic regression",
    "mri_logistic": "MRI logistic regression",
}


def metric_vector(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Same metric vector used by scripts/64_analyze_pet_anchored_fusion.py."""
    return np.asarray(
        [
            balanced_accuracy_score(y, prediction),
            roc_auc_score(y, probability),
            average_precision_score(y, probability),
        ],
        dtype=float,
    )


def read_seed(seed_dir: Path) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_path = seed_dir / "tables" / "pet_anchored_fusion_results.csv"
    fold_path = seed_dir / "tables" / "pet_anchored_fusion_fold_metrics.csv"
    prediction_path = seed_dir / "predictions" / "pet_anchored_fusion_oof.csv"
    manifest_path = seed_dir / "metrics" / "run_manifest.json"
    for path in (result_path, fold_path, prediction_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required per-seed result is missing or empty: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_seeds = [int(value) for value in manifest.get("split_seeds", [])]
    if len(manifest_seeds) != 1:
        raise ValueError(f"Expected exactly one split seed in {manifest_path}; got {manifest_seeds}")
    seed = manifest_seeds[0]
    results = pd.read_csv(result_path)
    folds = pd.read_csv(fold_path)
    predictions = pd.read_csv(prediction_path)
    if results["model"].duplicated().any():
        raise ValueError(f"Duplicate model summaries in {result_path}")
    folds["seed"] = seed
    predictions["seed"] = seed
    return seed, results, folds, predictions


def aggregate_summary(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in per_seed_results.groupby("model", sort=False):
        n_seeds = int(group["seed"].nunique())
        ddof = 1 if n_seeds > 1 else 0
        row: dict[str, Any] = {"model": model, "n_seeds": n_seeds}
        for source, prefix in (
            ("auroc_mean", "auroc"),
            ("balanced_accuracy_mean", "balanced_accuracy"),
            ("auprc_mean", "auprc"),
            ("ece_mean", "ece"),
            ("sensitivity_mean", "sensitivity"),
            ("specificity_mean", "specificity"),
        ):
            values = group[source].astype(float).to_numpy()
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_std"] = float(values.std(ddof=ddof))
            if prefix == "auroc":
                row["auroc_min"] = float(values.min())
                row["auroc_max"] = float(values.max())
        rows.append(row)
    columns = [
        "model",
        "n_seeds",
        "auroc_mean",
        "auroc_std",
        "auroc_min",
        "auroc_max",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "auprc_mean",
        "auprc_std",
        "ece_mean",
        "ece_std",
        "sensitivity_mean",
        "sensitivity_std",
        "specificity_mean",
        "specificity_std",
    ]
    return pd.DataFrame(rows)[columns].sort_values("auroc_mean", ascending=False).reset_index(drop=True)


def paired_wilcoxon(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    pivot = per_seed_results.pivot(index="seed", columns="model", values="auroc_mean")
    if PRIMARY_MODEL not in pivot:
        raise ValueError(f"Primary model {PRIMARY_MODEL!r} is absent from per-seed summaries")
    rows = []
    for baseline in sorted(model for model in pivot.columns if model != PRIMARY_MODEL):
        paired = pivot[[PRIMARY_MODEL, baseline]].dropna()
        n_seeds = int(len(paired))
        if n_seeds < 5:
            warnings.warn(
                f"Wilcoxon comparison {PRIMARY_MODEL} vs {baseline} has only {n_seeds} seeds; "
                "the test is underpowered and should be interpreted descriptively.",
                RuntimeWarning,
            )
        differences = paired[PRIMARY_MODEL].to_numpy() - paired[baseline].to_numpy()
        if n_seeds == 0:
            statistic, p_value, mean_difference = np.nan, np.nan, np.nan
        elif np.allclose(differences, 0.0):
            statistic, p_value, mean_difference = 0.0, 1.0, 0.0
        else:
            try:
                result = wilcoxon(differences, alternative="two-sided")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError as exc:
                warnings.warn(f"Wilcoxon test failed for {baseline}: {exc}", RuntimeWarning)
                statistic, p_value = np.nan, np.nan
            mean_difference = float(differences.mean())
        rows.append(
            {
                "primary_model": PRIMARY_MODEL,
                "baseline_model": baseline,
                "metric": "auroc",
                "n_seeds": n_seeds,
                "mean_difference": mean_difference,
                "statistic": statistic,
                "p_value": p_value,
                "significant_at_0.05": bool(np.isfinite(p_value) and p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def pooled_subject_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int = 20260723) -> pd.DataFrame:
    required = {
        "seed",
        "sample_id",
        "subject_id",
        "label_id",
        "model",
        "prob_ad",
        "pred_label_id",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Pooled predictions are missing columns: {sorted(missing)}")
    models = list(predictions["model"].drop_duplicates())
    if PRIMARY_MODEL not in models:
        raise ValueError(f"Primary model {PRIMARY_MODEL!r} is absent from pooled predictions")
    matrices: dict[str, pd.DataFrame] = {}
    reference_keys: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    reference_subjects: np.ndarray | None = None
    for model in models:
        subset = (
            predictions[predictions["model"].eq(model)]
            .sort_values(["seed", "sample_id"])
            .reset_index(drop=True)
        )
        if subset.duplicated(["seed", "sample_id"]).any():
            raise ValueError(f"Duplicate seed/sample predictions for model {model}")
        keys = (subset["seed"].astype(str) + "::" + subset["sample_id"].astype(str)).to_numpy()
        labels = subset["label_id"].to_numpy(dtype=int)
        subjects = subset["subject_id"].astype(str).to_numpy()
        if reference_keys is None:
            reference_keys, reference_labels, reference_subjects = keys, labels, subjects
        elif (
            not np.array_equal(keys, reference_keys)
            or not np.array_equal(labels, reference_labels)
            or not np.array_equal(subjects, reference_subjects)
        ):
            raise ValueError(f"Predictions for {model} are not paired with the primary prediction rows")
        matrices[model] = subset
    assert reference_labels is not None and reference_subjects is not None
    unique_subjects = np.unique(reference_subjects)
    subject_rows = {
        subject: np.flatnonzero(reference_subjects == subject) for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    primary = matrices[PRIMARY_MODEL]
    rows: list[dict[str, Any]] = []
    for baseline_name in sorted(model for model in models if model != PRIMARY_MODEL):
        baseline = matrices[baseline_name]
        samples: list[np.ndarray] = []
        for _ in range(replicates):
            sampled_subjects = rng.choice(unique_subjects, len(unique_subjects), replace=True)
            indices = np.concatenate([subject_rows[subject] for subject in sampled_subjects])
            y = reference_labels[indices]
            if len(np.unique(y)) < 2:
                continue
            primary_values = metric_vector(
                y,
                primary["prob_ad"].to_numpy()[indices],
                primary["pred_label_id"].to_numpy(dtype=int)[indices],
            )
            baseline_values = metric_vector(
                y,
                baseline["prob_ad"].to_numpy()[indices],
                baseline["pred_label_id"].to_numpy(dtype=int)[indices],
            )
            samples.append(primary_values - baseline_values)
        distribution = np.asarray(samples)
        if not len(distribution):
            raise RuntimeError(f"No valid bootstrap replicates for comparison with {baseline_name}")
        for metric_index, metric in enumerate(METRIC_NAMES):
            values = distribution[:, metric_index]
            rows.append(
                {
                    "primary_model": PRIMARY_MODEL,
                    "baseline_model": baseline_name,
                    "metric": metric,
                    "mean_difference": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "probability_difference_positive": float(np.mean(values > 0)),
                    "bootstrap_replicates": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci_excludes_zero(
    bootstrap: pd.DataFrame, baseline: str, metric: str = "auroc"
) -> bool:
    row = bootstrap[
        bootstrap["baseline_model"].eq(baseline) & bootstrap["metric"].eq(metric)
    ]
    return bool(len(row) and float(row.iloc[0]["ci95_low"]) > 0.0)


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    top = summary.head(6).copy()

    def formatted(mean: float, std: float) -> str:
        return f"{mean:.3f} ± {std:.3f}"

    return pd.DataFrame(
        {
            "Model": top["model"].map(lambda value: PAPER_NAMES.get(value, value)),
            "AUROC (mean±std)": [
                formatted(mean, std) for mean, std in zip(top["auroc_mean"], top["auroc_std"])
            ],
            "Bal. Acc. (mean±std)": [
                formatted(mean, std)
                for mean, std in zip(
                    top["balanced_accuracy_mean"], top["balanced_accuracy_std"]
                )
            ],
            "Sensitivity (mean±std)": [
                formatted(mean, std)
                for mean, std in zip(top["sensitivity_mean"], top["sensitivity_std"])
            ],
            "Specificity (mean±std)": [
                formatted(mean, std)
                for mean, std in zip(top["specificity_mean"], top["specificity_std"])
            ],
            "ECE (mean±std)": [
                formatted(mean, std) for mean, std in zip(top["ece_mean"], top["ece_std"])
            ],
        }
    )


def main() -> None:
    global PRIMARY_MODEL
    parser = argparse.ArgumentParser(description="Aggregate PET-anchored fusion results across seeds.")
    parser.add_argument("--seed-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/pet_anchored_fusion_multiseed")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    args = parser.parse_args()
    PRIMARY_MODEL = str(args.primary_model)
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")

    seed_dirs = [Path(value).expanduser() for value in args.seed_dirs]
    loaded = [read_seed(path) for path in seed_dirs]
    seeds = [seed for seed, _, _, _ in loaded]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seed directories were supplied: {seeds}")
    model_sets = [set(results["model"]) for _, results, _, _ in loaded]
    if any(models != model_sets[0] for models in model_sets[1:]):
        raise ValueError("Per-seed result directories do not contain identical model sets")
    if len(seeds) < 5:
        warnings.warn(
            f"Only {len(seeds)} seeds were supplied; publication inference is underpowered.",
            RuntimeWarning,
        )

    result_frames = []
    fold_frames = []
    prediction_frames = []
    for seed, results, folds, predictions in loaded:
        results = results.copy()
        results["seed"] = seed
        result_frames.append(results)
        fold_frames.append(folds)
        prediction_frames.append(predictions)
    per_seed_results = pd.concat(result_frames, ignore_index=True)
    fold_detail = pd.concat(fold_frames, ignore_index=True)
    pooled_predictions = pd.concat(prediction_frames, ignore_index=True)

    summary = aggregate_summary(per_seed_results)
    wilcoxon_frame = paired_wilcoxon(per_seed_results)
    bootstrap = pooled_subject_bootstrap(
        pooled_predictions, replicates=args.bootstrap_replicates
    )
    table2 = paper_table(summary)

    primary_row = summary[summary["model"].eq(PRIMARY_MODEL)]
    if primary_row.empty:
        raise ValueError(f"Primary model {PRIMARY_MODEL!r} is absent from multiseed summary")
    primary_row = primary_row.iloc[0]
    decisions = {
        baseline: bootstrap_ci_excludes_zero(bootstrap, baseline)
        for baseline in DECISION_BASELINES
    }
    if all(decisions.values()):
        interpretation = (
            "The PET-anchored transformer has subject-clustered paired AUROC evidence over "
            "PET logistic regression, concatenation MLP, and equal three-modality averaging "
            "across the prespecified seeds."
        )
    elif decisions["pet_logistic"] and decisions["concat_mlp"]:
        interpretation = (
            "The PET-anchored transformer improves over learned unimodal/concatenation baselines, "
            "but not conclusively over equal three-modality averaging. Report the averaging model "
            "as a strong baseline and avoid a transformer-superiority claim."
        )
    else:
        interpretation = (
            "The PET-anchored transformer does not show robust paired AUROC evidence over all key "
            "baselines. Use the strongest simpler model as the predictive model and report the "
            "transformer as an ablation."
        )
    decision = {
        "primary_model": PRIMARY_MODEL,
        "n_seeds": len(seeds),
        "seeds_used": seeds,
        "multiseed_auroc_mean": float(primary_row["auroc_mean"]),
        "multiseed_auroc_std": float(primary_row["auroc_std"]),
        "transformer_vs_pet_logistic_auroc_ci_excludes_zero": decisions["pet_logistic"],
        "transformer_vs_concat_mlp_auroc_ci_excludes_zero": decisions["concat_mlp"],
        "transformer_vs_average_all_auroc_ci_excludes_zero": decisions["average_all"],
        "interpretation": interpretation,
    }

    output_dir = Path(args.output_dir).expanduser()
    tables_dir = output_dir / "tables"
    analysis_dir = output_dir / "analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "multiseed_summary.csv", index=False)
    fold_detail.to_csv(tables_dir / "multiseed_fold_detail.csv", index=False)
    table2.to_csv(tables_dir / "paper_table2.csv", index=False)
    wilcoxon_frame.to_csv(analysis_dir / "multiseed_wilcoxon.csv", index=False)
    bootstrap.to_csv(analysis_dir / "multiseed_bootstrap.csv", index=False)
    (analysis_dir / "publication_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("Multiseed summary")
    print(summary.to_string(index=False))
    print(f"\nPublication decision: {interpretation}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
