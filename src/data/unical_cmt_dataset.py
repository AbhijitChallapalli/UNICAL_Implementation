from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_npz_matrix(path: str | Path | None) -> tuple[np.ndarray | None, list[str], list[str]]:
    if not path or not Path(path).is_file():
        return None, [], []
    data = np.load(path, allow_pickle=True)
    x_key = "X" if "X" in data.files else "features"
    id_key = "sample_ids" if "sample_ids" in data.files else "sample_id"
    names_key = "feature_names" if "feature_names" in data.files else None
    names = [str(v) for v in data[names_key]] if names_key else [f"feature_{i}" for i in range(data[x_key].shape[1])]
    return np.asarray(data[x_key], dtype=np.float32), [str(v) for v in data[id_key]], names


class UNICALCMTDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cowsrad_npz: str | Path | None,
        pet_summary_npz: str | Path | None = None,
        gated_pet_manifest: str | Path | None = None,
        indices: np.ndarray | list[int] | None = None,
        cowsrad_matrix: np.ndarray | None = None,
    ) -> None:
        self.frame = manifest.iloc[indices].reset_index(drop=True) if indices is not None else manifest.reset_index(drop=True)
        self.cowsrad_x, cowsrad_ids, self.cowsrad_names = _load_npz_matrix(cowsrad_npz)
        if cowsrad_matrix is not None:
            self.cowsrad_x = np.asarray(cowsrad_matrix, dtype=np.float32)
            cowsrad_ids = self.frame["sample_id"].astype(str).tolist()
        self.cowsrad_lookup = {sid: i for i, sid in enumerate(cowsrad_ids)}
        self.pet_x, pet_ids, self.pet_names = _load_npz_matrix(pet_summary_npz)
        self.pet_lookup = {sid: i for i, sid in enumerate(pet_ids)}
        self.gated_lookup: dict[str, str] = {}
        if gated_pet_manifest and Path(gated_pet_manifest).is_file():
            pet_frame = pd.read_csv(gated_pet_manifest)
            if "gated_pet_cache_path" in pet_frame.columns:
                self.gated_lookup = dict(zip(pet_frame["sample_id"].astype(str), pet_frame["gated_pet_cache_path"].astype(str)))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        sample_id = str(row["sample_id"])
        mri_path = row.get("mri_3mt_cache_path", row.get("mri_cache_96_path", ""))
        mri = np.load(mri_path).astype(np.float32)
        if mri.ndim == 3:
            mri = mri[None]
        if self.cowsrad_x is None or sample_id not in self.cowsrad_lookup:
            raise KeyError(f"Missing CoWS-Rad features for sample_id={sample_id}")
        item: dict[str, torch.Tensor | str] = {
            "mri": torch.from_numpy(mri),
            "cowsrad": torch.from_numpy(self.cowsrad_x[self.cowsrad_lookup[sample_id]].astype(np.float32)),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": sample_id,
            "subject_id": str(row["subject_id"]),
        }
        if self.pet_x is not None and sample_id in self.pet_lookup:
            item["pet_summary"] = torch.from_numpy(self.pet_x[self.pet_lookup[sample_id]].astype(np.float32))
            item["pet_summary_available"] = torch.tensor(True)
        else:
            width = self.pet_x.shape[1] if self.pet_x is not None else 1
            item["pet_summary"] = torch.zeros(width, dtype=torch.float32)
            item["pet_summary_available"] = torch.tensor(False)
        gated_path = self.gated_lookup.get(sample_id)
        if gated_path and Path(gated_path).is_file():
            gated = np.load(gated_path).astype(np.float32)
            if gated.ndim == 3:
                gated = gated[None]
            item["gated_pet"] = torch.from_numpy(gated)
            item["gated_pet_available"] = torch.tensor(True)
        else:
            item["gated_pet"] = torch.zeros_like(item["mri"])
            item["gated_pet_available"] = torch.tensor(False)
        return item
