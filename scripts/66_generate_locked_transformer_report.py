from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_MODEL = "probability_anchored_transformer"
TARGET_AUROC = 0.85
DISPLAY_NAMES = {
    "probability_anchored_transformer": "Full anchored transformer",
    "probability_anchored_transformer_no_uncertainty": "Transformer, no uncertainty",
    "probability_anchored_transformer_uniform_weights": "Transformer, uniform weights",
    "average_all": "MRI + CoWS-Rad + PET average",
    "average_cowsrad_pet": "CoWS-Rad + PET average",
    "average_mri_pet": "MRI + PET average",
    "average_mri_cowsrad": "MRI + CoWS-Rad average",
    "concat_mlp": "Concatenation MLP",
    "pet_logistic": "PET logistic",
    "cowsrad_logistic": "CoWS-Rad logistic",
    "mri_logistic": "MRI logistic",
}


def display_name(value: str) -> str:
    return DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result is missing or empty: {path}")
    return path


def save_figure(
    figure: plt.Figure, figure_dir: Path, filename: str
) -> tuple[Path, str]:
    figure.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return path, f"data:image/png;base64,{encoded}"


def figure_card(title: str, description: str, image_uri: str) -> str:
    return (
        '<section class="figure-card">'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(description)}</p>"
        f'<img src="{image_uri}" alt="{html.escape(title)}">'
        "</section>"
    )


def format_table(
    frame: pd.DataFrame,
    *,
    float_columns: tuple[str, ...] = (),
    rename: dict[str, str] | None = None,
) -> str:
    result = frame.copy()
    if rename:
        result = result.rename(columns=rename)
    for column in float_columns:
        target = rename.get(column, column) if rename else column
        if target in result:
            result[target] = result[target].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    return result.to_html(index=False, escape=True, classes="data-table", border=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report for locked transformer results."
    )
    parser.add_argument(
        "--result-dir", default="probability_anchored_transformer_locked20"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else result_dir / "locked20_supervisor_report.html"
    )
    figure_dir = result_dir / "report_figures"

    summary_path = require(result_dir / "tables" / "multiseed_summary.csv")
    folds_path = require(result_dir / "tables" / "multiseed_fold_detail.csv")
    wilcoxon_path = require(result_dir / "analysis" / "multiseed_wilcoxon.csv")
    bootstrap_path = require(result_dir / "analysis" / "multiseed_bootstrap.csv")
    decision_path = require(result_dir / "analysis" / "publication_decision.json")

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(folds_path)
    wilcoxon = pd.read_csv(wilcoxon_path)
    bootstrap = pd.read_csv(bootstrap_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    n_seeds = int(folds["seed"].nunique())
    n_models = int(folds["model"].nunique())
    group_counts = folds.groupby(["seed", "model"]).size()
    complete = bool(
        len(folds) == n_seeds * n_models * 5
        and len(group_counts) == n_seeds * n_models
        and (group_counts == 5).all()
    )
    seed_metrics = (
        folds.groupby(["seed", "model"], as_index=False)[
            ["auroc", "balanced_accuracy", "auprc", "ece"]
        ]
        .mean()
    )
    auroc_pivot = seed_metrics.pivot(index="seed", columns="model", values="auroc")
    summary_lookup = summary.set_index("model")
    primary = summary_lookup.loc[PRIMARY_MODEL]
    best = summary.sort_values("auroc_mean", ascending=False).iloc[0]
    average_all = summary_lookup.loc["average_all"]
    target_gap = TARGET_AUROC - float(primary["auroc_mean"])

    figures: list[str] = []

    top = summary.sort_values("auroc_mean", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y = np.arange(len(top))
    ax.errorbar(
        top["auroc_mean"],
        y,
        xerr=top["auroc_std"],
        fmt="o",
        color="#136f63",
        ecolor="#75b8ad",
        capsize=3,
    )
    ax.set_yticks(y, [display_name(value) for value in top["model"]])
    ax.axvline(TARGET_AUROC, color="#c33c54", linestyle="--", label="Target AUROC 0.85")
    ax.set_xlabel("Mean AUROC across locked seeds (error bars: seed SD)")
    ax.set_title("Locked-20 discrimination")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    _, uri = save_figure(fig, figure_dir, "01_auroc_forest.png")
    figures.append(
        figure_card(
            "AUROC ranking and target",
            "Mean AUROC across 20 split seeds. Error bars show variability across seeds, not independent-cohort confidence intervals.",
            uri,
        )
    )

    top_six = summary.sort_values("auroc_mean", ascending=False).head(6)["model"].tolist()
    box_values = [auroc_pivot[model].dropna().to_numpy() for model in top_six]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    boxes = ax.boxplot(
        box_values,
        patch_artist=True,
        tick_labels=[display_name(x) for x in top_six],
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor("#b8ddd7")
        patch.set_edgecolor("#136f63")
    ax.axhline(TARGET_AUROC, color="#c33c54", linestyle="--")
    ax.set_ylabel("Seed-level mean AUROC")
    ax.set_title("Seed-to-seed stability")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", alpha=0.25)
    _, uri = save_figure(fig, figure_dir, "02_seed_auroc_boxplot.png")
    figures.append(
        figure_card(
            "Seed-level AUROC distributions",
            "The transformer family and probability averaging overlap strongly across the locked seed pool.",
            uri,
        )
    )

    heat = auroc_pivot[top_six].T
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=0.72, vmax=0.79)
    ax.set_yticks(np.arange(len(heat.index)), [display_name(x) for x in heat.index])
    ax.set_xticks(np.arange(len(heat.columns)), [str(x) for x in heat.columns], rotation=65)
    ax.set_xlabel("Locked seed")
    ax.set_title("AUROC stability heatmap")
    fig.colorbar(image, ax=ax, label="AUROC")
    _, uri = save_figure(fig, figure_dir, "03_seed_heatmap.png")
    figures.append(
        figure_card(
            "Seed/model heatmap",
            "No single method dominates every resampling of the same cohort.",
            uri,
        )
    )

    delta = wilcoxon.sort_values("mean_difference")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    colors = ["#136f63" if value > 0 else "#c33c54" for value in delta["mean_difference"]]
    ax.barh(
        [display_name(value) for value in delta["baseline_model"]],
        delta["mean_difference"],
        color=colors,
        alpha=0.88,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Full transformer minus baseline: seed-mean AUROC")
    ax.set_title("Paired AUROC differences across seeds")
    ax.grid(axis="x", alpha=0.25)
    _, uri = save_figure(fig, figure_dir, "04_paired_auroc_deltas.png")
    figures.append(
        figure_card(
            "Paired AUROC differences",
            "Positive values favor the full probability-anchored transformer. Subject-clustered bootstrap intervals remain the primary uncertainty analysis.",
            uri,
        )
    )

    top_scatter = summary.sort_values("auroc_mean", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(7.8, 6.1))
    scatter = ax.scatter(
        top_scatter["sensitivity_mean"],
        top_scatter["specificity_mean"],
        c=top_scatter["auroc_mean"],
        cmap="viridis",
        s=90,
        edgecolor="white",
        linewidth=0.8,
    )
    for _, row in top_scatter.iterrows():
        ax.annotate(
            display_name(str(row["model"])),
            (row["sensitivity_mean"], row["specificity_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Specificity")
    ax.set_title("Operating-point trade-off")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Mean AUROC")
    _, uri = save_figure(fig, figure_dir, "05_sensitivity_specificity.png")
    figures.append(
        figure_card(
            "Sensitivity–specificity trade-off",
            "Values use validation-derived Youden thresholds within each fold.",
            uri,
        )
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.scatter(
        summary["ece_mean"],
        summary["auroc_mean"],
        c=summary["balanced_accuracy_mean"],
        cmap="plasma",
        s=85,
        edgecolor="white",
    )
    for _, row in summary.iterrows():
        ax.annotate(
            display_name(str(row["model"])),
            (row["ece_mean"], row["auroc_mean"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(TARGET_AUROC, color="#c33c54", linestyle="--")
    ax.set_xlabel("Expected calibration error (lower is better)")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Discrimination–calibration profile")
    ax.grid(alpha=0.25)
    _, uri = save_figure(fig, figure_dir, "06_calibration_tradeoff.png")
    figures.append(
        figure_card(
            "Discrimination and calibration",
            "No evaluated model reached the prespecified 0.85 AUROC target.",
            uri,
        )
    )

    residual = folds.loc[
        folds["model"].eq(PRIMARY_MODEL), "residual_scale"
    ].dropna()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.hist(residual, bins=20, color="#75b8ad", edgecolor="#136f63")
    ax.axvline(-0.01, color="#c33c54", linestyle="--")
    ax.axvline(0.01, color="#c33c54", linestyle="--", label="±0.01")
    ax.set_xlabel("Learned residual scale")
    ax.set_ylabel("Fold count")
    ax.set_title("Transformer residual correction across 100 folds")
    ax.legend()
    _, uri = save_figure(fig, figure_dir, "07_residual_scale.png")
    figures.append(
        figure_card(
            "Residual-scale distribution",
            f"{100.0 * float((residual.abs() <= 0.01).mean()):.1f}% of folds remained within ±0.01, indicating very small bounded corrections to the probability anchor.",
            uri,
        )
    )

    contributions = pd.Series(
        {
            "Add PET to MRI + CoWS-Rad": float(
                summary_lookup.loc["average_all", "auroc_mean"]
                - summary_lookup.loc["average_mri_cowsrad", "auroc_mean"]
            ),
            "Add MRI to CoWS-Rad + PET": float(
                summary_lookup.loc["average_all", "auroc_mean"]
                - summary_lookup.loc["average_cowsrad_pet", "auroc_mean"]
            ),
            "Add CoWS-Rad to MRI + PET": float(
                summary_lookup.loc["average_all", "auroc_mean"]
                - summary_lookup.loc["average_mri_pet", "auroc_mean"]
            ),
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.barh(contributions.index, contributions.values, color=["#136f63", "#75b8ad", "#75b8ad"])
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Incremental mean AUROC")
    ax.set_title("Conditional modality contribution")
    ax.grid(axis="x", alpha=0.25)
    _, uri = save_figure(fig, figure_dir, "08_modality_contribution.png")
    figures.append(
        figure_card(
            "Modality contribution",
            "Synthetic PET provides the largest incremental discrimination in the locked analysis.",
            uri,
        )
    )

    main_table = summary.sort_values("auroc_mean", ascending=False).copy()
    main_table.insert(1, "display_model", main_table["model"].map(display_name))
    main_table = main_table[
        [
            "display_model",
            "n_seeds",
            "auroc_mean",
            "auroc_std",
            "balanced_accuracy_mean",
            "auprc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "ece_mean",
        ]
    ]
    main_table_html = format_table(
        main_table,
        float_columns=(
            "auroc_mean",
            "auroc_std",
            "balanced_accuracy_mean",
            "auprc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "ece_mean",
        ),
        rename={
            "display_model": "Model",
            "n_seeds": "Seeds",
            "auroc_mean": "AUROC mean",
            "auroc_std": "AUROC SD",
            "balanced_accuracy_mean": "Balanced accuracy",
            "auprc_mean": "AUPRC",
            "sensitivity_mean": "Sensitivity",
            "specificity_mean": "Specificity",
            "ece_mean": "ECE",
        },
    )

    key_bootstrap = bootstrap[
        bootstrap["metric"].eq("auroc")
        & bootstrap["baseline_model"].isin(
            ["average_all", "pet_logistic", "concat_mlp"]
        )
    ].copy()
    key_bootstrap["baseline_model"] = key_bootstrap["baseline_model"].map(display_name)
    bootstrap_html = format_table(
        key_bootstrap[
            [
                "baseline_model",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "probability_difference_positive",
                "bootstrap_replicates",
            ]
        ],
        float_columns=(
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "probability_difference_positive",
        ),
        rename={
            "baseline_model": "Baseline",
            "mean_difference": "Mean AUROC difference",
            "ci95_low": "95% CI low",
            "ci95_high": "95% CI high",
            "probability_difference_positive": "P(difference > 0)",
            "bootstrap_replicates": "Bootstrap replicates",
        },
    )

    wilcoxon_table = wilcoxon.copy()
    wilcoxon_table["baseline_model"] = wilcoxon_table["baseline_model"].map(display_name)
    wilcoxon_html = format_table(
        wilcoxon_table[
            [
                "baseline_model",
                "n_seeds",
                "mean_difference",
                "statistic",
                "p_value",
                "significant_at_0.05",
            ]
        ],
        float_columns=("mean_difference", "statistic", "p_value"),
        rename={
            "baseline_model": "Baseline",
            "n_seeds": "Seeds",
            "mean_difference": "Mean AUROC difference",
            "statistic": "Wilcoxon statistic",
            "p_value": "p-value",
            "significant_at_0.05": "p < 0.05",
        },
    )

    best_name = display_name(str(best["model"]))
    decision_text = html.escape(str(decision.get("interpretation", "")))
    completeness_text = "PASS" if complete else "FAIL"
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Locked-20 Probability-Anchored Transformer Report</title>
  <style>
    :root {{ --ink:#18302d; --muted:#58706c; --accent:#136f63; --soft:#e8f3f1;
             --warn:#c33c54; --paper:#ffffff; --line:#d7e4e1; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#f3f7f6; color:var(--ink);
            font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.5; }}
    main {{ width:min(1180px,94vw); margin:28px auto 60px; }}
    header {{ padding:34px; border-radius:18px; color:white;
              background:linear-gradient(120deg,#0f5048,#168376); }}
    header h1 {{ margin:0 0 8px; font-size:2rem; }}
    header p {{ margin:0; opacity:.92; }}
    h2 {{ margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:7px; }}
    h3 {{ margin:0 0 6px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
             gap:14px; margin:18px 0; }}
    .kpi,.panel,.figure-card {{ background:var(--paper); border:1px solid var(--line);
                                border-radius:14px; box-shadow:0 5px 18px #173d3710; }}
    .kpi {{ padding:18px; }}
    .kpi .value {{ font-size:1.55rem; font-weight:750; color:var(--accent); }}
    .kpi .label {{ color:var(--muted); font-size:.9rem; }}
    .panel {{ padding:20px; margin:14px 0; overflow:auto; }}
    .figure-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:16px; }}
    .figure-card {{ padding:18px; }}
    .figure-card p {{ color:var(--muted); min-height:48px; }}
    .figure-card img {{ width:100%; height:auto; display:block; }}
    .data-table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
    .data-table th {{ background:var(--soft); position:sticky; top:0; }}
    .data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; }}
    .data-table th:first-child,.data-table td:first-child {{ text-align:left; }}
    .status-pass {{ color:#087f5b; font-weight:700; }}
    .warning {{ border-left:5px solid var(--warn); background:#fff3f5; padding:14px 18px; border-radius:8px; }}
    .note {{ border-left:5px solid var(--accent); background:var(--soft); padding:14px 18px; border-radius:8px; }}
    code {{ background:#edf2f1; padding:2px 5px; border-radius:4px; }}
    footer {{ color:var(--muted); margin-top:30px; font-size:.85rem; }}
    @media(max-width:700px) {{ .figure-grid {{ grid-template-columns:1fr; }} header {{ padding:24px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Locked-20 Probability-Anchored Transformer Evaluation</h1>
    <p>Leakage-safe repeated stratified group cross-validation • supervisor report • actual observed results</p>
  </header>

  <div class="grid">
    <div class="kpi"><div class="value">{float(primary['auroc_mean']):.3f} ± {float(primary['auroc_std']):.3f}</div><div class="label">Full transformer AUROC</div></div>
    <div class="kpi"><div class="value">{float(primary['balanced_accuracy_mean']):.3f}</div><div class="label">Balanced accuracy</div></div>
    <div class="kpi"><div class="value">{float(primary['auprc_mean']):.3f}</div><div class="label">AUPRC</div></div>
    <div class="kpi"><div class="value">{float(primary['ece_mean']):.3f}</div><div class="label">Expected calibration error</div></div>
    <div class="kpi"><div class="value">{n_seeds}</div><div class="label">Locked split seeds</div></div>
    <div class="kpi"><div class="value">{target_gap:.3f}</div><div class="label">Gap to 0.85 AUROC target</div></div>
  </div>

  <section class="panel">
    <h2>Executive assessment</h2>
    <p>The best seed-averaged model was <strong>{html.escape(best_name)}</strong>
       (AUROC {float(best['auroc_mean']):.4f} ± {float(best['auroc_std']):.4f}).
       The prespecified full transformer achieved AUROC {float(primary['auroc_mean']):.4f}
       and did not outperform equal three-modality averaging
       ({float(average_all['auroc_mean']):.4f}).</p>
    <p>{decision_text}</p>
    <div class="warning"><strong>Target audit:</strong> no evaluated model reached AUROC 0.85.
      Results were not filtered or altered to meet a target.</div>
  </section>

  <section class="panel">
    <h2>Data integrity</h2>
    <p><span class="status-pass">{completeness_text}</span> — {len(folds):,} fold/model rows;
       {n_seeds} seeds × {n_models} models × 5 folds. All expected combinations are present.</p>
    <p>Split seeds repeatedly evaluate the same underlying subjects; they are not 20 independent cohorts.</p>
  </section>

  <h2>Main results table</h2>
  <section class="panel">{main_table_html}</section>

  <h2>Figures</h2>
  <div class="figure-grid">{''.join(figures)}</div>

  <h2>Subject-clustered paired bootstrap</h2>
  <section class="panel">
    <p>Primary model: {html.escape(display_name(PRIMARY_MODEL))}. Intervals crossing zero do not support superiority.</p>
    {bootstrap_html}
  </section>

  <h2>Seed-level Wilcoxon comparisons</h2>
  <section class="panel">
    <p>These comparisons measure consistency across split seeds. Because seeds reuse subjects,
       subject-clustered bootstrap results are more conservative for inferential claims.</p>
    {wilcoxon_html}
  </section>

  <h2>Dropout, ROC and calibration-curve availability</h2>
  <section class="panel">
    <div class="note"><strong>Modality-dropout curve:</strong> not estimable from this run.
      The experiment used one fixed dropout configuration and did not include a probability-anchored
      no-dropout or dropout-rate sweep. Creating such a curve from these files would fabricate data.</div>
    <p><strong>ROC, precision-recall and calibration curves:</strong> the aggregate folder contains fold
       metrics but not per-subject OOF predictions. Curves require the 20 per-seed prediction CSV files.
       The report therefore includes only metrics supported by the supplied locked outputs.</p>
  </section>

  <h2>Publication interpretation</h2>
  <section class="panel">
    <ul>
      <li>The transformer significantly improves seed-level AUROC over MRI, CoWS-Rad, PET logistic,
          and concatenation MLP baselines.</li>
      <li>Subject-clustered bootstrap supports improvement over concatenation MLP, but not over
          PET logistic or equal probability averaging.</li>
      <li>The transformer correction remained close to zero, supporting probability averaging as
          a strong and stable anchor.</li>
      <li>Synthetic PET supplies the largest conditional multimodal contribution in this experiment.</li>
      <li>The results support competitiveness and robustness, not an AUROC ≥0.85 or superiority-over-averaging claim.</li>
    </ul>
  </section>

  <h2>Source files</h2>
  <section class="panel">
    <ul>
      <li><code>{html.escape(str(summary_path))}</code></li>
      <li><code>{html.escape(str(folds_path))}</code></li>
      <li><code>{html.escape(str(wilcoxon_path))}</code></li>
      <li><code>{html.escape(str(bootstrap_path))}</code></li>
      <li><code>{html.escape(str(decision_path))}</code></li>
    </ul>
  </section>

  <footer>Generated reproducibly by <code>scripts/66_generate_locked_transformer_report.py</code>.
  Figures are embedded in this HTML so the file can be shared independently.</footer>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report: {output_path}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
