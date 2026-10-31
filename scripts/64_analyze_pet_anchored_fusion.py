from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


PRIMARY_MODEL = "pet_anchored_transformer"
KEY_BASELINES = ("pet_logistic", "average_all", "concat_mlp")


def metric_vector(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([
        balanced_accuracy_score(y, prediction),
        roc_auc_score(y, probability),
        average_precision_score(y, probability),
    ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject-clustered paired analysis for PET-anchored fusion.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    parser.add_argument(
        "--key-baselines",
        default=",".join(KEY_BASELINES),
        help="Comma-separated models used for paired bootstrap comparisons",
    )
    args = parser.parse_args()
    primary_model = str(args.primary_model)
    key_baselines = tuple(
        value.strip() for value in args.key_baselines.split(",") if value.strip()
    )

    frame = pd.read_csv(args.predictions)
    required = {"sample_id", "subject_id", "label_id", "model", "prob_ad", "pred_label_id", "fold"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    models = list(frame["model"].drop_duplicates())
    if primary_model not in models:
        raise ValueError(f"Primary model {primary_model!r} is absent; models={models}")
    matrices: dict[str, pd.DataFrame] = {}
    reference_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    reference_subjects: np.ndarray | None = None
    for model in models:
        subset = frame[frame["model"].eq(model)].sort_values("sample_id").reset_index(drop=True)
        if subset["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample predictions for {model}")
        ids = subset["sample_id"].astype(str).to_numpy()
        labels = subset["label_id"].to_numpy(dtype=int)
        subjects = subset["subject_id"].astype(str).to_numpy()
        if reference_ids is None:
            reference_ids, reference_labels, reference_subjects = ids, labels, subjects
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(labels, reference_labels) or not np.array_equal(subjects, reference_subjects):
            raise ValueError(f"Predictions for {model} are not paired with the reference model")
        matrices[model] = subset
    assert reference_labels is not None and reference_subjects is not None

    pooled_rows = []
    for model, subset in matrices.items():
        values = metric_vector(reference_labels, subset["prob_ad"].to_numpy(), subset["pred_label_id"].to_numpy(dtype=int))
        pooled_rows.append({"model": model, "balanced_accuracy": values[0], "auroc": values[1], "auprc": values[2]})
    pooled = pd.DataFrame(pooled_rows).sort_values(["auroc", "balanced_accuracy"], ascending=False)

    unique_subjects = np.unique(reference_subjects)
    subject_rows = {subject: np.flatnonzero(reference_subjects == subject) for subject in unique_subjects}
    rng = np.random.default_rng(args.seed)
    comparisons = []
    primary = matrices[primary_model]
    baselines = [model for model in key_baselines if model in matrices and model != primary_model]
    for baseline_name in baselines:
        baseline = matrices[baseline_name]
        samples: list[np.ndarray] = []
        for _ in range(args.bootstrap_replicates):
            selected = rng.choice(unique_subjects, len(unique_subjects), replace=True)
            indices = np.concatenate([subject_rows[subject] for subject in selected])
            y = reference_labels[indices]
            if len(np.unique(y)) < 2:
                continue
            primary_values = metric_vector(y, primary["prob_ad"].to_numpy()[indices], primary["pred_label_id"].to_numpy(dtype=int)[indices])
            baseline_values = metric_vector(y, baseline["prob_ad"].to_numpy()[indices], baseline["pred_label_id"].to_numpy(dtype=int)[indices])
            samples.append(primary_values - baseline_values)
        distribution = np.asarray(samples)
        if not len(distribution):
            raise RuntimeError(f"No valid bootstrap replicates for comparison with {baseline_name}")
        for metric_index, metric in enumerate(("balanced_accuracy", "auroc", "auprc")):
            values = distribution[:, metric_index]
            comparisons.append({
                "primary_model": primary_model,
                "baseline_model": baseline_name,
                "metric": metric,
                "mean_difference": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_difference_positive": float(np.mean(values > 0)),
                "bootstrap_replicates": int(len(values)),
            })
    comparison_frame = pd.DataFrame(comparisons)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_dir / "pooled_oof_metrics.csv", index=False)
    comparison_frame.to_csv(output_dir / "subject_clustered_paired_bootstrap.csv", index=False)

    pooled_lookup = pooled.set_index("model")
    best_auroc_model = str(pooled.iloc[0]["model"])
    transformer_vs_pet = comparison_frame[
        comparison_frame["baseline_model"].eq("pet_logistic") & comparison_frame["metric"].eq("auroc")
    ]
    transformer_auroc_ci_excludes_zero = bool(len(transformer_vs_pet) and float(transformer_vs_pet.iloc[0]["ci95_low"]) > 0)
    decision = {
        "primary_model": primary_model,
        "best_pooled_auroc_model": best_auroc_model,
        "primary_pooled_metrics": pooled_lookup.loc[primary_model].to_dict(),
        "transformer_auroc_ci_excludes_zero_vs_pet_logistic": transformer_auroc_ci_excludes_zero,
        "interpretation": (
            "Transformer has paired AUROC evidence over PET logistic. Confirm on locked additional seeds before a robustness claim."
            if transformer_auroc_ci_excludes_zero
            else "Transformer does not have paired AUROC evidence over PET logistic. Use the strongest simpler model as the predictive model and report the transformer as an ablation."
        ),
    }
    (output_dir / "publication_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("Pooled OOF metrics")
    print(pooled.to_string(index=False))
    print("\nSubject-clustered paired bootstrap")
    print(comparison_frame.to_string(index=False))
    print("\n" + decision["interpretation"])


if __name__ == "__main__":
    main()
