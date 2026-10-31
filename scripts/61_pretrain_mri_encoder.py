#!/usr/bin/env python
"""Self-supervised pretraining of MRICNNTransformerEncoder via 3D MAE.

Usage
-----
::

    python scripts/61_pretrain_mri_encoder.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml

Resume from checkpoint::

    python scripts/61_pretrain_mri_encoder.py \\
        --config configs/oasis3_pretrain_mri_encoder.yaml \\
        --resume outputs/pretrain_mri_encoder/checkpoints/mae_epoch_100.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ssl.mae3d import build_mae3d
from src.ssl.augmentations3d import build_ssl_validation_transform
from src.ssl.pretrain_dataset import PretrainMRIDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(config: dict, key: str) -> Path:
    p = Path(config["paths"][key])
    if not p.is_absolute():
        p = Path(config["paths"]["project_root"]) / p
    return p


def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_kwargs: dict,
    grad_clip: float,
    accum_steps: int,
    channels_last_3d: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        x = batch.to(device, non_blocking=True)
        if channels_last_3d:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast(**amp_kwargs):
            out = model(x)
            loss = out["loss"] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += out["loss"].item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_kwargs: dict,
    channels_last_3d: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        if channels_last_3d:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast(**amp_kwargs):
            losses.append(float(model(x)["loss"].item()))
    return float(np.mean(losses)) if losses else float("inf")


def subject_disjoint_split(
    manifest: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a deterministic subject-disjoint SSL train/validation split."""
    usable = manifest[manifest["status"].isin(["passed", "cached"])].copy()
    if "subject_id" not in usable.columns:
        usable["subject_id"] = usable["session_id"].astype(str).str.split("_MR_", n=1).str[0]
    groups = usable["subject_id"].dropna().astype(str).unique()
    if len(groups) < 2:
        raise ValueError("At least two subjects are required for an SSL validation split")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_val = max(1, min(len(groups) - 1, int(round(len(groups) * validation_fraction))))
    val_groups = set(groups[:n_val].tolist())
    is_val = usable["subject_id"].astype(str).isin(val_groups)
    train = usable[~is_val].reset_index(drop=True)
    val = usable[is_val].reset_index(drop=True)
    if set(train["subject_id"].astype(str)) & set(val["subject_id"].astype(str)):
        raise RuntimeError("Subject leakage detected in the SSL validation split")
    return train, val


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }, path)


def save_encoder_only(path: Path, model: torch.nn.Module, epoch: int, config: dict) -> None:
    """Save only the encoder weights (what downstream uses)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # model.encoder is the MRICNNTransformerEncoder
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": model.encoder.state_dict(),
        "encoder_config": config["encoder"],
    }, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def prune_checkpoints(ckpt_dir: Path, keep_last_k: int) -> None:
    """Remove old periodic checkpoints, keeping the last *k*."""
    periodic = sorted(ckpt_dir.glob("mae_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    # Never delete *_best.pt or *_final.pt
    periodic = [p for p in periodic if "_best" not in p.name and "_final" not in p.name]
    for p in periodic[:-keep_last_k]:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised 3D MAE pretraining of MRI encoder.")
    parser.add_argument("--config", default="configs/oasis3_pretrain_mri_encoder.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    config = _load_config(args.config)
    ssl_cfg = config["ssl"]
    if os.environ.get("UNICAL_SSL_BATCH_SIZE"):
        ssl_cfg["batch_size"] = int(os.environ["UNICAL_SSL_BATCH_SIZE"])
    if os.environ.get("UNICAL_SSL_ACCUMULATION_STEPS"):
        ssl_cfg["accumulation_steps"] = int(os.environ["UNICAL_SSL_ACCUMULATION_STEPS"])
    seed = int(config["project"].get("seed", 42))

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matmul_prec = str(ssl_cfg.get("matmul_precision", "high")).lower()
    if matmul_prec in ("high", "highest", "medium"):
        torch.set_float32_matmul_precision(matmul_prec)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Paths
    ckpt_dir = _resolve(config, "pretrain_checkpoint_dir")
    log_dir = _resolve(config, "pretrain_log_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    manifest_path = _resolve(config, "pretrain_manifest_csv")
    if not manifest_path.is_file():
        print(f"ERROR: Pretrain manifest not found: {manifest_path}", file=sys.stderr)
        print("Run scripts/60_preprocess_pretrain_mri.py first (or --finalize).", file=sys.stderr)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    if manifest["session_id"].duplicated().any() or manifest["cache_path"].duplicated().any():
        raise RuntimeError("Pretraining manifest contains duplicate sessions or cache paths")
    missing_cache = [p for p in manifest.loc[manifest["status"].isin(["passed", "cached"]), "cache_path"] if not Path(p).is_file()]
    if missing_cache:
        raise FileNotFoundError(
            f"{len(missing_cache)} usable manifest rows have missing cache files (first: {missing_cache[0]})"
        )
    train_frame, val_frame = subject_disjoint_split(
        manifest,
        float(ssl_cfg.get("validation_fraction", 0.10)),
        seed,
    )
    dataset = PretrainMRIDataset(train_frame, config)
    val_dataset = PretrainMRIDataset(
        val_frame,
        config,
        transform=build_ssl_validation_transform(config),
    )
    print(
        f"Pretrain dataset: train={len(dataset)} volumes/{train_frame['subject_id'].nunique()} subjects, "
        f"validation={len(val_dataset)} volumes/{val_frame['subject_id'].nunique()} subjects"
    )

    loader = DataLoader(
        dataset,
        batch_size=int(ssl_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(ssl_cfg.get("num_workers", 8)),
        pin_memory=bool(ssl_cfg.get("pin_memory", True)),
        persistent_workers=bool(ssl_cfg.get("persistent_workers", True)) and int(ssl_cfg.get("num_workers", 8)) > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(ssl_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(ssl_cfg.get("num_workers", 8)),
        pin_memory=bool(ssl_cfg.get("pin_memory", True)),
        persistent_workers=bool(ssl_cfg.get("persistent_workers", True)) and int(ssl_cfg.get("num_workers", 8)) > 0,
        drop_last=False,
    )

    # Model
    channels_last_3d = bool(ssl_cfg.get("channels_last_3d", False)) and device.type == "cuda"
    model = build_mae3d(config).to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    n_params_dec = sum(p.numel() for p in model.decoder.parameters())
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"MAE parameters: encoder={n_params_enc:,} decoder={n_params_dec:,} total={n_params_total:,}")
    print(f"Spatial grid: {model.grid}  tokens/volume: {model.n_tokens}  "
          f"patch_voxels: {model.patch_voxels}  mask_ratio: {model.mask_ratio}")
    print(f"Channels-last 3D: {channels_last_3d}")

    # Optimizer
    total_epochs = int(ssl_cfg.get("epochs", 300))
    warmup_epochs = int(ssl_cfg.get("warmup_epochs", 20))
    base_lr = float(ssl_cfg.get("base_lr", 1.5e-4))
    min_lr = float(ssl_cfg.get("min_lr", 1e-6))
    weight_decay = float(ssl_cfg.get("weight_decay", 0.05))
    grad_clip = float(ssl_cfg.get("gradient_clip_norm", 1.0))
    accum_steps = int(ssl_cfg.get("accumulation_steps", 1))

    # AdamW with separate decay for normalization/bias
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or "pos" in name or "mask_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=base_lr, betas=(0.9, 0.95))

    # AMP
    amp_enabled = bool(ssl_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_str = str(ssl_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    amp_kwargs = {"device_type": device.type, "enabled": amp_enabled, "dtype": amp_dtype}
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, scaler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")

    # Logging
    save_every = int(ssl_cfg.get("save_every_epochs", 25))
    keep_last_k = int(ssl_cfg.get("keep_last_k", 3))
    log_path = log_dir / "pretrain_log.jsonl"

    print(f"\n{'='*60}")
    print(f"Starting MAE pretraining: epochs {start_epoch}→{total_epochs}")
    print(f"Effective batch size: {int(ssl_cfg.get('batch_size', 4))} × {accum_steps} = "
          f"{int(ssl_cfg.get('batch_size', 4)) * accum_steps}")
    print(f"LR schedule: warmup {warmup_epochs} → cosine → {min_lr}")
    print(f"AMP: {amp_enabled}  dtype: {amp_dtype}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Set LR
        lr = cosine_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr)
        set_lr(optimizer, lr)

        # Train
        metrics = train_one_epoch(
            model, loader, optimizer, scaler, device, amp_kwargs,
            grad_clip, accum_steps, channels_last_3d,
        )
        metrics["val_loss"] = validate_one_epoch(
            model, val_loader, device, amp_kwargs, channels_last_3d,
        )
        if device.type == "cuda":
            metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024**3)
        elapsed = time.time() - t0
        metrics["epoch"] = epoch
        metrics["lr"] = lr
        metrics["elapsed_s"] = elapsed

        # Log
        line = (f"Epoch {epoch:4d}/{total_epochs}  "
                f"loss={metrics['train_loss']:.6f}  "
                f"val_loss={metrics['val_loss']:.6f}  "
                f"lr={lr:.2e}  "
                f"time={elapsed:.1f}s"
                + (f"  peak_gpu={metrics['peak_allocated_gib']:.1f}GiB" if device.type == "cuda" else ""))
        print(line)

        with open(log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Best checkpoint (encoder-only)
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            save_encoder_only(
                ckpt_dir / "mri_encoder_pretrained_best.pt",
                model, epoch, config,
            )

        # Periodic checkpoint (full MAE for resuming)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"mae_epoch_{epoch:04d}.pt",
                model, optimizer, scaler, epoch, best_loss, config,
            )
            prune_checkpoints(ckpt_dir, keep_last_k)

    # Final save
    save_checkpoint(
        ckpt_dir / "mae_final.pt",
        model, optimizer, scaler, total_epochs - 1, best_loss, config,
    )
    save_encoder_only(
        ckpt_dir / "mri_encoder_pretrained_final.pt",
        model, total_epochs - 1, config,
    )

    print(f"\n{'='*60}")
    print(f"Pretraining complete.  Best validation loss: {best_loss:.6f}")
    print(f"Encoder checkpoint: {ckpt_dir / 'mri_encoder_pretrained_best.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
