#!/usr/bin/env python
"""Validate the pretrained MRI encoder via linear probing on the paper cohort.

Freezes the pretrained ``MRICNNTransformerEncoder``, trains a single linear
head on top, and reports AD/CN balanced accuracy across stratified group
k-fold CV.  This confirms the encoder learned useful representations before
plugging it into the full UNICAL CMT pipeline.

Usage
-----
::

    python scripts/62_validate_pretrained_encoder.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml \\
        --checkpoint outputs/pretrain_mri_encoder/checkpoints/mri_encoder_pretrained_best.pt

Compare against a random-init baseline::

    python scripts/62_validate_pretrained_encoder.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml \\
        --random-init
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mri_cnn_transformer_encoder import MRICNNTransformerEncoder
from src.evaluation.metrics_binary import binary_metrics, youden_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(config: dict, section: str, key: str) -> Path:
    p = Path(config[section][key])
    if not p.is_absolute():
        p = Path(config["paths"]["project_root"]) / p
    return p


class VolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: np.ndarray, path_col: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.labels = labels.astype(np.int64)
        self.path_col = path_col

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        arr = np.load(self.frame.loc[idx, self.path_col]).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None]
        return torch.from_numpy(arr), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def build_encoder(config: dict) -> MRICNNTransformerEncoder:
    enc_cfg = config["encoder"]
    return MRICNNTransformerEncoder(
        input_channels=int(enc_cfg.get("input_channels", 1)),
        stem_channels=int(enc_cfg.get("stem_channels", 16)),
        block_channels=list(enc_cfg.get("block_channels", [32, 64, 128, 256])),
        downsample_stride=int(enc_cfg.get("downsample_stride", 2)),
        dropout3d=float(enc_cfg.get("dropout3d", 0.0)),
        d_model=int(enc_cfg.get("d_model", 512)),
        transformer_layers=int(enc_cfg.get("transformer_layers", 2)),
        transformer_heads=int(enc_cfg.get("transformer_heads", 8)),
        transformer_ff_dim=int(enc_cfg.get("transformer_ff_dim", 1024)),
        transformer_dropout=float(enc_cfg.get("transformer_dropout", 0.0)),
        output_dim=int(enc_cfg.get("output_dim", 512)),
    )


def load_pretrained(encoder: MRICNNTransformerEncoder, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    print(f"Loaded pretrained encoder from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract [B, output_dim] embeddings for all samples."""
    encoder.eval()
    all_emb, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            emb = encoder(xb)
            if isinstance(emb, dict):
                emb = emb["embedding"]
        all_emb.append(emb.float().cpu().numpy())
        all_y.append(yb.numpy())
    return np.concatenate(all_emb), np.concatenate(all_y)


def train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Linear, float]:
    """Train a linear head on frozen embeddings and return it with val threshold."""
    dim = x_train.shape[1]
    head = nn.Linear(dim, 2).to(device)

    # Class weights
    counts = np.bincount(y_train.astype(int), minlength=2)
    weights = torch.tensor([len(y_train) / max(2 * c, 1) for c in counts], dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train.astype(np.int64)).to(device)
    xv = torch.from_numpy(x_val).to(device)

    best_bacc, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        logits = head(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_prob = torch.softmax(head(xv), dim=1)[:, 1].cpu().numpy()
        thr = youden_threshold(y_val, val_prob)
        m = binary_metrics(y_val, val_prob, thr)
        if m["balanced_accuracy"] > best_bacc:
            best_bacc = m["balanced_accuracy"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_state:
        head.load_state_dict(best_state)
    return head, best_bacc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pretrained MRI encoder via linear probing.")
    parser.add_argument("--config", default="configs/oasis3_pretrain_mri_encoder.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained encoder checkpoint.")
    parser.add_argument("--random-init", action="store_true",
                        help="Run with random-init encoder (baseline comparison).")
    args = parser.parse_args()

    config = _load_config(args.config)
    val_cfg = config.get("validation", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(config["paths"]["project_root"]) / config["paths"]["pretrain_checkpoint_dir"] / "mri_encoder_pretrained_best.pt"

    # Load paper cohort manifest
    manifest_path = _resolve(config, "validation", "probe_manifest_csv")
    label_path = _resolve(config, "validation", "probe_label_csv")

    if not manifest_path.is_file():
        print(f"ERROR: Probe manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    frame = pd.read_csv(manifest_path)
    # Filter to passed preprocessing
    if "preprocessing_3mt_status" in frame.columns:
        frame = frame[frame["preprocessing_3mt_status"].isin(["passed", "cached"])].copy()

    # Determine path column
    path_col = "mri_3mt_cache_path" if "mri_3mt_cache_path" in frame.columns else "mri_cache_96_path"
    if path_col not in frame.columns:
        print(f"ERROR: No MRI cache path column found in {manifest_path}", file=sys.stderr)
        sys.exit(1)

    y = frame["label_id"].to_numpy(dtype=int)
    groups = frame["subject_id"].astype(str).to_numpy()

    print(f"Probe dataset: {len(frame)} samples, {np.bincount(y, minlength=2)} class distribution")

    # Build encoder
    encoder = build_encoder(config)

    if args.random_init:
        print("Using RANDOM-INIT encoder (baseline)")
        tag = "random_init"
    else:
        if not ckpt_path.is_file():
            print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        load_pretrained(encoder, ckpt_path)
        tag = "pretrained"

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    # Extract embeddings (once — encoder is frozen)
    print("Extracting embeddings...")
    full_ds = VolumeDataset(frame, y, path_col)
    full_loader = DataLoader(full_ds, batch_size=int(val_cfg.get("probe_batch_size", 16)),
                             shuffle=False, num_workers=4)
    embeddings, labels = extract_embeddings(encoder, full_loader, device)
    print(f"Embedding shape: {embeddings.shape}")

    # K-fold linear probing
    n_folds = int(val_cfg.get("probe_folds", 5))
    probe_seed = int(val_cfg.get("probe_seed", 42))
    probe_epochs = int(val_cfg.get("probe_epochs", 50))
    probe_lr = float(val_cfg.get("probe_lr", 1e-3))

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=probe_seed)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y)), y, groups)):
        # Inner split for early stopping
        inner_skf = StratifiedGroupKFold(n_splits=min(5, max(2, np.bincount(y[train_idx], minlength=2).min())),
                                         shuffle=True, random_state=probe_seed + fold_i)
        fit_rel, val_rel = next(inner_skf.split(np.arange(len(train_idx)), y[train_idx], groups[train_idx]))
        fit_idx = train_idx[fit_rel]
        val_idx = train_idx[val_rel]

        head, val_bacc = train_linear_probe(
            embeddings[fit_idx], y[fit_idx],
            embeddings[val_idx], y[val_idx],
            epochs=probe_epochs, lr=probe_lr, device=device,
        )

        # Test
        head.eval()
        with torch.no_grad():
            test_prob = torch.softmax(
                head(torch.from_numpy(embeddings[test_idx]).to(device)),
                dim=1,
            )[:, 1].cpu().numpy()

        with torch.no_grad():
            val_prob = torch.softmax(
                head(torch.from_numpy(embeddings[val_idx]).to(device)),
                dim=1,
            )[:, 1].cpu().numpy()
        threshold = youden_threshold(y[val_idx], val_prob)
        metrics = binary_metrics(y[test_idx], test_prob, threshold)
        metrics["fold"] = fold_i
        fold_results.append(metrics)
        print(f"  Fold {fold_i}: bal_acc={metrics['balanced_accuracy']:.4f}  "
              f"auroc={metrics['auroc']:.4f}  f1={metrics['macro_f1']:.4f}")

    # Summary
    df = pd.DataFrame(fold_results)
    summary = {}
    for col in ["balanced_accuracy", "accuracy", "auroc", "auprc", "macro_f1", "sensitivity", "specificity"]:
        if col in df:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_std"] = float(df[col].std())
    summary["encoder"] = tag
    summary["n_folds"] = n_folds
    summary["n_samples"] = len(frame)

    print(f"\n{'='*60}")
    print(f"Linear Probe Results ({tag})")
    print(f"{'='*60}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    log_dir = Path(config["paths"]["project_root"]) / config["paths"]["pretrain_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"linear_probe_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
