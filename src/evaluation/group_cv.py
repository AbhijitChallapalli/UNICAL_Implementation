from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def repeated_stratified_group_kfold(
    frame: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    n_splits: int = 5,
    seeds: list[int] | tuple[int, ...] = (42,),
) -> list[dict[str, np.ndarray | int]]:
    groups = frame[group_column].astype(str).to_numpy()
    labels = frame[label_column].to_numpy()
    indices = np.arange(len(frame))
    folds: list[dict[str, np.ndarray | int]] = []
    for repeat_idx, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(indices, labels, groups)):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise RuntimeError(f"Subject overlap detected in repeat {repeat_idx} fold {fold_idx}: {sorted(overlap)[:5]}")
            folds.append({
                "repeat": repeat_idx,
                "seed": int(seed),
                "fold": fold_idx,
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
    return folds


def save_fold_assignments(
    frame: pd.DataFrame,
    folds: list[dict[str, np.ndarray | int]],
    out_dir: str | Path,
    sample_id_column: str = "sample_id",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        rows = []
        for split_name, key in (("train", "train_idx"), ("test", "test_idx")):
            idx = np.asarray(fold[key], dtype=int)
            split = frame.iloc[idx][[sample_id_column, "subject_id", "label", "label_id"]].copy()
            split["split"] = split_name
            split["repeat"] = int(fold["repeat"])
            split["seed"] = int(fold["seed"])
            split["fold"] = int(fold["fold"])
            rows.append(split)
        pd.concat(rows, ignore_index=True).to_csv(out / f"repeat{fold['repeat']}_seed{fold['seed']}_fold{fold['fold']}.csv", index=False)
