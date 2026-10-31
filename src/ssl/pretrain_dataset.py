"""Unlabeled MRI dataset for self-supervised pretraining.

Loads preprocessed ``.npy`` volumes produced by ``scripts/60_preprocess_pretrain_mri.py``
and applies on-the-fly 3D augmentations from :mod:`src.ssl.augmentations3d`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.ssl.augmentations3d import Compose3D, build_ssl_augmentation


class PretrainMRIDataset(Dataset):
    """Dataset of preprocessed, unlabeled 3D MRI volumes for SSL.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain a ``cache_path`` column pointing to ``.npy`` files and a
        ``status`` column. Rows marked ``passed`` or provenance-validated
        ``cached`` are used.
    config : dict
        Full YAML config; augmentation is read from ``config['ssl']['augmentation']``.
    transform : Compose3D | None
        Explicit override.  If *None*, one is built from *config*.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        config: dict[str, Any],
        transform: Compose3D | None = None,
    ) -> None:
        passed = manifest[manifest["status"].isin(["passed", "cached"])].reset_index(drop=True)
        self.paths: list[str] = passed["cache_path"].tolist()
        if len(self.paths) == 0:
            raise ValueError("No usable passed/cached volumes found in pretrain manifest.")
        self.transform = transform or build_ssl_augmentation(config)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.paths[idx]).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None]  # [1, D, H, W]
        vol = torch.from_numpy(arr)
        vol = self.transform(vol)
        return vol


class PretrainMRIDatasetDual(Dataset):
    """Returns *two* independently-augmented views of each volume.

    Useful if the SSL objective needs a contrastive pair (not used by MAE but
    provided for future flexibility / ablation with VICReg / BYOL).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        config: dict[str, Any],
        transform: Compose3D | None = None,
    ) -> None:
        passed = manifest[manifest["status"].isin(["passed", "cached"])].reset_index(drop=True)
        self.paths: list[str] = passed["cache_path"].tolist()
        if len(self.paths) == 0:
            raise ValueError("No usable passed/cached volumes found in pretrain manifest.")
        self.transform = transform or build_ssl_augmentation(config)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = np.load(self.paths[idx]).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None]
        vol = torch.from_numpy(arr)
        view1 = self.transform(vol.clone())
        view2 = self.transform(vol.clone())
        return view1, view2
