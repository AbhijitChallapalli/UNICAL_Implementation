from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_binary import binary_metrics, youden_threshold


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "cuda_if_available"))
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def class_weights(labels: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    weights = np.array([len(labels) / max(2 * counts[i], 1) for i in range(2)], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.stale = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, value: float, model: torch.nn.Module) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.stale = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.stale += 1
        return self.stale >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload


def fit_cowsrad_fold_transform(
    x: np.ndarray,
    train_idx: np.ndarray,
    config: dict[str, Any],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, Pipeline]:
    topk = int(config.get("cowsrad_selected", {}).get("cowsrad_v2_topk", 50))
    k = min(max(topk, 1), x.shape[1])
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("variance", VarianceThreshold(0.0)),
        ("select", SelectKBest(f_classif, k=k)),
    ])
    pipe.fit(x[train_idx], np.asarray(config["_fold_labels"])[train_idx])
    return pipe.transform(x).astype(np.float32), pipe


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold_source_y: np.ndarray | None = None, threshold_source_prob: np.ndarray | None = None, ece_bins: int = 15) -> dict[str, Any]:
    threshold = youden_threshold(threshold_source_y, threshold_source_prob) if threshold_source_y is not None and threshold_source_prob is not None else 0.5
    return binary_metrics(y_true, y_prob, threshold, ece_bins)


def summarize_fold_metrics(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {"model": model_name, "n_folds": int(len(frame))}
    for col in ["balanced_accuracy", "accuracy", "auroc", "auprc", "macro_f1", "sensitivity", "specificity", "precision", "ece", "brier"]:
        if col in frame:
            out[f"{col}_mean"] = float(np.nanmean(frame[col]))
            out[f"{col}_std"] = float(np.nanstd(frame[col]))
    return out
