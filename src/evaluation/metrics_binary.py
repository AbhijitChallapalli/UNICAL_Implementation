from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if not idx.any():
            continue
        ece += float(idx.mean()) * abs(float(y_prob[idx].mean()) - float(y_true[idx].mean()))
    return float(ece)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.asarray(y_prob, dtype=float))
    if candidates.size == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        score = sens + spec - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, ece_bins: int = 15) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ece": expected_calibration_error(y_true, y_prob, ece_bins),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        warnings.warn(f"AUROC undefined for this fold: {exc}")
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError as exc:
        warnings.warn(f"AUPRC undefined for this fold: {exc}")
        out["auprc"] = float("nan")
    return out
