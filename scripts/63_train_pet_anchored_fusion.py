from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.group_cv import repeated_stratified_group_kfold, save_fold_assignments
from src.evaluation.metrics_binary import binary_metrics, youden_threshold
from src.mri_ad_cn_utils import cfg_path, load_pipeline_config, write_json
from src.models.mri_cnn_transformer_encoder import MRICNNTransformerEncoder
from src.models.pet_anchored_residual_transformer import (
    ConcatenationMLP,
    PETAnchoredResidualTransformer,
    ProbabilityAnchoredTransformer,
)
from src.training.unical_cmt_train_utils import summarize_fold_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = (
    "mri_logistic",
    "cowsrad_logistic",
    "pet_logistic",
    "average_mri_cowsrad",
    "average_mri_pet",
    "average_cowsrad_pet",
    "average_all",
    "concat_mlp",
    "pet_anchored_transformer",
    "pet_anchored_transformer_no_uncertainty",
    "pet_anchored_transformer_no_modality_dropout",
)
PROBABILITY_ANCHORED_VARIANTS = (
    "probability_anchored_transformer",
    "probability_anchored_transformer_no_uncertainty",
    "probability_anchored_transformer_uniform_weights",
)
AVAILABLE_VARIANTS = DEFAULT_VARIANTS + PROBABILITY_ANCHORED_VARIANTS


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aligned_npz(path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    x_key = "X" if "X" in data.files else "features"
    id_key = "sample_ids" if "sample_ids" in data.files else "sample_id"
    ids = [str(value) for value in data[id_key]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    missing = [str(value) for value in frame["sample_id"].astype(str) if str(value) not in lookup]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} cohort samples; examples={missing[:5]}")
    names = [str(value) for value in data["feature_names"]] if "feature_names" in data.files else [f"feature_{i}" for i in range(data[x_key].shape[1])]
    order = [lookup[str(value)] for value in frame["sample_id"].astype(str)]
    matrix = np.asarray(data[x_key], dtype=np.float32)[order]
    if not np.isfinite(matrix).all():
        matrix = matrix.copy()
    return matrix, names


class VolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.frame.iloc[index]
        volume = np.load(str(row["mri_3mt_cache_path"])).astype(np.float32)
        if volume.ndim == 3:
            volume = volume[None]
        return torch.from_numpy(volume), str(row["sample_id"])


@torch.no_grad()
def extract_frozen_mri_embeddings(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> np.ndarray:
    cache = output_root / "features" / "frozen_mri_embeddings.npz"
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    if cache.is_file():
        payload = np.load(cache, allow_pickle=True)
        cached_ids = np.asarray(payload["sample_ids"]).astype(str)
        embeddings = np.asarray(payload["X"], dtype=np.float32)
        if np.array_equal(cached_ids, sample_ids) and embeddings.shape[0] == len(frame):
            print(f"Loaded cached frozen MRI embeddings: {cache} {embeddings.shape}")
            return embeddings
        raise RuntimeError(f"MRI embedding cache does not match the current cohort: {cache}")

    checkpoint_value = config.get("pretrain", {}).get("mri_encoder_checkpoint")
    if not checkpoint_value:
        raise ValueError("pretrain.mri_encoder_checkpoint is required")
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_absolute():
        project_root = Path(str(config.get("paths", {}).get("project_root", REPO_ROOT)))
        checkpoint = project_root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    encoder = MRICNNTransformerEncoder.from_config(config["model"]["mri_cnn_transformer"])
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("encoder_state_dict")
    if not state:
        raise KeyError(f"Checkpoint has no encoder_state_dict: {checkpoint}")
    encoder.load_state_dict(state, strict=True)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    batch_size = int(config.get("pet_anchored_fusion", {}).get("embedding_batch_size", 4))
    loader = DataLoader(VolumeDataset(frame), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[np.ndarray] = []
    seen_ids: list[str] = []
    amp_enabled = device.type == "cuda"
    for volumes, ids in loader:
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
            encoded = encoder(volumes.to(device, non_blocking=True))
        chunks.append(encoded.float().cpu().numpy())
        seen_ids.extend([str(value) for value in ids])
    if seen_ids != sample_ids.tolist():
        raise RuntimeError("MRI embedding extraction changed sample order")
    embeddings = np.concatenate(chunks).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=embeddings, sample_ids=sample_ids, checkpoint=str(checkpoint))
    print(f"Saved frozen MRI embeddings: {cache} {embeddings.shape}")
    return embeddings


def inner_group_split(frame: pd.DataFrame, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = frame.iloc[train_idx]["label_id"].to_numpy(dtype=int)
    groups = frame.iloc[train_idx]["subject_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_rel, val_rel = next(splitter.split(np.arange(len(train_idx)), labels, groups))
    fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]
    if set(frame.iloc[fit_idx]["subject_id"]).intersection(set(frame.iloc[val_idx]["subject_id"])):
        raise RuntimeError("Subject leakage in inner split")
    return fit_idx, val_idx


def fold_transform_cowsrad(raw: np.ndarray, y: np.ndarray, fit_idx: np.ndarray, top_k: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    variance = VarianceThreshold()
    fit = imputer.fit_transform(raw[fit_idx])
    fit = scaler.fit_transform(fit)
    fit = variance.fit_transform(fit)
    selector = SelectKBest(f_classif, k=min(max(1, top_k), fit.shape[1]))
    selector.fit(fit, y[fit_idx])
    all_values = imputer.transform(raw)
    all_values = scaler.transform(all_values)
    all_values = variance.transform(all_values)
    return selector.transform(all_values).astype(np.float32)


def fold_standardize(raw: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit = imputer.fit_transform(raw[fit_idx])
    scaler.fit(fit)
    return scaler.transform(imputer.transform(raw)).astype(np.float32)


def fit_logistic(x: np.ndarray, y: np.ndarray, fit_idx: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, solver="liblinear", random_state=0)
    model.fit(x[fit_idx], y[fit_idx])
    return model


def build_stacking_probabilities(
    features: dict[str, np.ndarray],
    models: dict[str, LogisticRegression],
    y: np.ndarray,
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create leakage-safe base probabilities for the anchored transformer.

    Validation and test rows use classifiers fitted only on the outer-fold fit
    partition. Fit rows receive group-aware cross-fitted predictions so the
    transformer cannot learn from optimistically in-sample base scores.
    """

    modality_order = ("mri", "cowsrad", "pet")
    probabilities = np.column_stack(
        [models[name].predict_proba(features[name])[:, 1] for name in modality_order]
    )
    fit_labels = y[fit_idx]
    fit_groups = frame.iloc[fit_idx]["subject_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    covered = np.zeros(len(fit_idx), dtype=bool)
    for train_rel, holdout_rel in splitter.split(
        np.arange(len(fit_idx)), fit_labels, fit_groups
    ):
        train_abs = fit_idx[train_rel]
        holdout_abs = fit_idx[holdout_rel]
        if set(fit_groups[train_rel]).intersection(set(fit_groups[holdout_rel])):
            raise RuntimeError("Subject leakage in stacking probability cross-fit")
        for modality_index, name in enumerate(modality_order):
            crossfit_model = fit_logistic(features[name], y, train_abs)
            probabilities[holdout_abs, modality_index] = crossfit_model.predict_proba(
                features[name][holdout_abs]
            )[:, 1]
        covered[holdout_rel] = True
    if not covered.all():
        raise RuntimeError("Cross-fitted stacking probabilities did not cover every fit row")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite stacking probabilities were produced")
    return probabilities.astype(np.float32)


def add_prediction_rows(
    rows: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    name: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    val_prob: np.ndarray,
    test_prob: np.ndarray,
    fold: dict[str, Any],
    ece_bins: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    threshold = youden_threshold(y[val_idx], val_prob)
    metrics = binary_metrics(y[test_idx], test_prob, threshold, ece_bins)
    metrics.update({"model": name, "repeat": int(fold["repeat"]), "seed": int(fold["seed"]), "fold": int(fold["fold"])})
    if metadata:
        metrics.update(metadata)
    rows.append(metrics)
    pred = frame.iloc[test_idx][["sample_id", "subject_id", "label", "label_id"]].copy()
    pred["model"] = name
    pred["prob_ad"] = test_prob
    pred["threshold"] = threshold
    pred["pred_label_id"] = (test_prob >= threshold).astype(int)
    pred["repeat"] = int(fold["repeat"])
    pred["seed"] = int(fold["seed"])
    pred["fold"] = int(fold["fold"])
    predictions.append(pred)


def train_neural_model(
    model: nn.Module,
    arrays: tuple[np.ndarray, ...],
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    auxiliary_weight: float,
    residual_penalty: float,
    weight_anchor_penalty: float,
) -> nn.Module:
    model = model.to(device)
    tensors = [torch.from_numpy(value) for value in arrays]
    fit_ds = TensorDataset(*(tensor[fit_idx] for tensor in tensors), torch.from_numpy(y[fit_idx].astype(np.float32)))
    val_infer_ds = TensorDataset(*(tensor[val_idx] for tensor in tensors))
    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_infer_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    counts = np.bincount(y[fit_idx], minlength=2)
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    initial_val_prob = infer_neural(model, val_loader, device)
    best_score = float(binary_metrics(y[val_idx], initial_val_prob, 0.5)["auroc"])
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch in fit_loader:
            *feature_batch, labels = batch
            feature_batch = [value.to(device) for value in feature_batch]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(*feature_batch)
            if hasattr(output, "logits"):
                loss = F.binary_cross_entropy_with_logits(output.logits, labels, pos_weight=pos_weight)
                if isinstance(model, ProbabilityAnchoredTransformer):
                    uniform = output.keep_mask.to(output.convex_weights.dtype)
                    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
                    weight_penalty = (output.convex_weights - uniform).square().mean()
                    loss = (
                        loss
                        + residual_penalty * output.residual_scale.square()
                        + weight_anchor_penalty * weight_penalty
                    )
                else:
                    auxiliary_labels = labels.unsqueeze(1).expand_as(output.auxiliary_logits)
                    auxiliary_loss = F.binary_cross_entropy_with_logits(
                        output.auxiliary_logits, auxiliary_labels, pos_weight=pos_weight
                    )
                    loss = loss + auxiliary_weight * auxiliary_loss + residual_penalty * output.residual_scale.square()
            else:
                loss = F.binary_cross_entropy_with_logits(output, labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_prob = infer_neural(model, val_loader, device)
        try:
            score = float(binary_metrics(y[val_idx], val_prob, 0.5)["auroc"])
        except (ValueError, KeyError):
            score = -np.inf
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def infer_neural(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in loader:
        output = model(*(value.to(device) for value in batch))
        logits = output.logits if hasattr(output, "logits") else output
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def infer_arrays(model: nn.Module, arrays: tuple[np.ndarray, ...], indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    tensors = [torch.from_numpy(value[indices]) for value in arrays]
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False, num_workers=0)
    return infer_neural(model, loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe PET-anchored residual transformer fusion.")
    parser.add_argument("--config", default="configs/oasis3_unical_cmt_3mt_style.yaml")
    parser.add_argument("--output-root", default="outputs/pet_anchored_fusion_seed42")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, help="Override config splits.seeds for this run")
    parser.add_argument("--batch-size", type=int, help="Override pet_anchored_fusion.batch_size")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.seeds:
        config["splits"]["seeds"] = list(args.seeds)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        config.setdefault("pet_anchored_fusion", {})["batch_size"] = args.batch_size
    output_root = resolve_output(args.output_root)
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    unknown = sorted(set(variants).difference(AVAILABLE_VARIANTS))
    if unknown:
        parser.error(f"Unknown variants: {unknown}; choices={AVAILABLE_VARIANTS}")
    frame = pd.read_csv(cfg_path(config, "cmt_preprocessed_manifest_csv"))
    frame = frame[frame["preprocessing_3mt_status"].isin(["passed", "cached"])].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit).copy()
    y = frame["label_id"].to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Binary AD/CN labels are required")
    raw_cows, _ = load_aligned_npz(cfg_path(config, "cowsrad_selected_npz"), frame)
    raw_pet, pet_names = load_aligned_npz(cfg_path(config, "pet_summary_features_npz"), frame)
    device_name = str(config.get("training", {}).get("device", "cuda_if_available"))
    if device_name == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    embeddings = extract_frozen_mri_embeddings(frame, config, output_root, device)
    fusion_cfg = config.get("pet_anchored_fusion", {})
    folds = repeated_stratified_group_kfold(
        frame,
        group_column="subject_id",
        label_column="label_id",
        n_splits=int(config["splits"]["n_splits"]),
        seeds=list(config["splits"]["seeds"]),
    )
    save_fold_assignments(frame, folds, output_root / "splits")
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    ece_bins = int(config.get("evaluation", {}).get("ece_bins", 15))

    for fold in folds:
        fold_seed = int(fold["seed"]) + int(fold["fold"])
        seed_everything(fold_seed)
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        fit_idx, val_idx = inner_group_split(frame, train_idx, fold_seed)
        train_subjects = set(frame.iloc[train_idx]["subject_id"].astype(str))
        test_subjects = set(frame.iloc[test_idx]["subject_id"].astype(str))
        if train_subjects.intersection(test_subjects):
            raise RuntimeError("Outer-fold subject leakage detected")
        mri = fold_standardize(embeddings, fit_idx)
        cows = fold_transform_cowsrad(raw_cows, y, fit_idx, int(config.get("cowsrad_selected", {}).get("cowsrad_v2_topk", 50)))
        pet = fold_standardize(raw_pet, fit_idx)
        arrays = (mri, cows, pet)
        logistic_features = {"mri": mri, "cowsrad": cows, "pet": pet}
        logistic_models = {name: fit_logistic(values, y, fit_idx) for name, values in logistic_features.items()}
        val_probs = {name: model.predict_proba(logistic_features[name][val_idx])[:, 1] for name, model in logistic_models.items()}
        test_probs = {name: model.predict_proba(logistic_features[name][test_idx])[:, 1] for name, model in logistic_models.items()}
        needs_probability_anchor = any(
            value.startswith("probability_anchored_") for value in variants
        )
        probability_anchored_arrays: tuple[np.ndarray, ...] | None = None
        if needs_probability_anchor:
            stacking_probabilities = build_stacking_probabilities(
                logistic_features,
                logistic_models,
                y,
                frame,
                fit_idx,
                fold_seed,
            )
            probability_anchored_arrays = (*arrays, stacking_probabilities)

        for modality, variant in (("mri", "mri_logistic"), ("cowsrad", "cowsrad_logistic"), ("pet", "pet_logistic")):
            if variant in variants:
                add_prediction_rows(rows, prediction_rows, name=variant, frame=frame, y=y, val_idx=val_idx, test_idx=test_idx, val_prob=val_probs[modality], test_prob=test_probs[modality], fold=fold, ece_bins=ece_bins)
        averages = {
            "average_mri_cowsrad": ("mri", "cowsrad"),
            "average_mri_pet": ("mri", "pet"),
            "average_cowsrad_pet": ("cowsrad", "pet"),
            "average_all": ("mri", "cowsrad", "pet"),
        }
        for name, members in averages.items():
            if name not in variants:
                continue
            add_prediction_rows(
                rows,
                prediction_rows,
                name=name,
                frame=frame,
                y=y,
                val_idx=val_idx,
                test_idx=test_idx,
                val_prob=np.mean([val_probs[value] for value in members], axis=0),
                test_prob=np.mean([test_probs[value] for value in members], axis=0),
                fold=fold,
                ece_bins=ece_bins,
                metadata={"members": "+".join(members)},
            )

        neural_variants = [
            value
            for value in variants
            if value.startswith(("concat_", "pet_anchored_", "probability_anchored_"))
        ]
        for variant_index, variant in enumerate(neural_variants):
            seed_everything(fold_seed + 100 * (variant_index + 1))
            if variant == "concat_mlp":
                model: nn.Module = ConcatenationMLP(mri.shape[1], cows.shape[1], pet.shape[1], hidden_dim=int(fusion_cfg.get("concat_hidden_dim", 128)), dropout=float(fusion_cfg.get("dropout", 0.15)))
                model_arrays = arrays
            elif variant.startswith("probability_anchored_"):
                if probability_anchored_arrays is None:
                    raise RuntimeError("Probability-anchored arrays were not initialized")
                model = ProbabilityAnchoredTransformer(
                    mri.shape[1],
                    cows.shape[1],
                    pet.shape[1],
                    d_model=int(fusion_cfg.get("probability_anchor_d_model", 64)),
                    heads=int(fusion_cfg.get("probability_anchor_heads", 4)),
                    ff_dim=int(fusion_cfg.get("probability_anchor_ff_dim", 128)),
                    layers=int(fusion_cfg.get("probability_anchor_layers", 1)),
                    dropout=float(fusion_cfg.get("probability_anchor_dropout", 0.15)),
                    modality_dropout=tuple(
                        fusion_cfg.get("probability_anchor_modality_dropout", [0.10, 0.10, 0.05])
                    ),
                    use_uncertainty_gate=variant != "probability_anchored_transformer_no_uncertainty",
                    use_adaptive_weights=variant != "probability_anchored_transformer_uniform_weights",
                    pet_uncertainty_indices=tuple(fusion_cfg.get("pet_uncertainty_indices", list(range(10, 18)))),
                    max_residual_scale=float(fusion_cfg.get("probability_anchor_max_residual_scale", 0.50)),
                )
                model_arrays = probability_anchored_arrays
            else:
                use_gate = variant != "pet_anchored_transformer_no_uncertainty"
                modality_dropout = (0.0, 0.0, 0.0) if variant == "pet_anchored_transformer_no_modality_dropout" else tuple(fusion_cfg.get("modality_dropout", [0.10, 0.10, 0.05]))
                model = PETAnchoredResidualTransformer(
                    mri.shape[1],
                    cows.shape[1],
                    pet.shape[1],
                    d_model=int(fusion_cfg.get("d_model", 128)),
                    heads=int(fusion_cfg.get("heads", 4)),
                    ff_dim=int(fusion_cfg.get("ff_dim", 256)),
                    layers=int(fusion_cfg.get("layers", 1)),
                    dropout=float(fusion_cfg.get("dropout", 0.15)),
                    modality_dropout=modality_dropout,
                    use_uncertainty_gate=use_gate,
                    pet_uncertainty_indices=tuple(fusion_cfg.get("pet_uncertainty_indices", list(range(10, 18)))),
                )
                model_arrays = arrays
            model = train_neural_model(
                model,
                model_arrays,
                y,
                fit_idx,
                val_idx,
                device=device,
                epochs=int(fusion_cfg.get("epochs", 100)),
                batch_size=int(fusion_cfg.get("batch_size", 64)),
                learning_rate=float(fusion_cfg.get("learning_rate", 1e-4)),
                weight_decay=float(fusion_cfg.get("weight_decay", 1e-4)),
                patience=int(fusion_cfg.get("patience", 12)),
                auxiliary_weight=float(fusion_cfg.get("auxiliary_weight", 0.1)),
                residual_penalty=float(fusion_cfg.get("residual_penalty", 0.01)),
                weight_anchor_penalty=float(fusion_cfg.get("probability_anchor_weight_penalty", 0.10)),
            )
            val_prob = infer_arrays(model, model_arrays, val_idx, device, int(fusion_cfg.get("batch_size", 64)))
            test_prob = infer_arrays(model, model_arrays, test_idx, device, int(fusion_cfg.get("batch_size", 64)))
            metadata: dict[str, Any] = {"trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))}
            if isinstance(model, PETAnchoredResidualTransformer):
                metadata.update({
                    "residual_scale": float(model.residual_scale.detach().cpu()),
                    "convex_weights": json.dumps(torch.softmax(model.convex_weight_logits.detach().cpu(), dim=0).tolist()),
                    "uncertainty_gate": bool(model.use_uncertainty_gate),
                    "modality_dropout": json.dumps(model.modality_dropout.detach().cpu().tolist()),
                })
            elif isinstance(model, ProbabilityAnchoredTransformer):
                metadata.update({
                    "residual_scale": float(model.residual_scale.detach().cpu()),
                    "uncertainty_gate": bool(model.use_uncertainty_gate),
                    "adaptive_weights": bool(model.use_adaptive_weights),
                    "max_residual_scale": float(model.max_residual_scale),
                    "modality_dropout": json.dumps(model.modality_dropout.detach().cpu().tolist()),
                })
            add_prediction_rows(rows, prediction_rows, name=variant, frame=frame, y=y, val_idx=val_idx, test_idx=test_idx, val_prob=val_prob, test_prob=test_prob, fold=fold, ece_bins=ece_bins, metadata=metadata)
            checkpoint = output_root / "checkpoints" / f"{variant}_seed{fold['seed']}_fold{fold['fold']}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "variant": variant, "fold": int(fold["fold"]), "seed": int(fold["seed"]), "pet_feature_names": pet_names}, checkpoint)

    fold_frame = pd.DataFrame(rows)
    summaries = []
    for name, group in fold_frame.groupby("model", sort=False):
        summary = summarize_fold_metrics(group.to_dict("records"), name)
        summary["trainable_parameters"] = int(group["trainable_parameters"].iloc[0]) if "trainable_parameters" in group and group["trainable_parameters"].notna().any() else 0
        summaries.append(summary)
    tables = output_root / "tables"
    predictions = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(tables / "pet_anchored_fusion_fold_metrics.csv", index=False)
    pd.DataFrame(summaries).sort_values(["balanced_accuracy_mean", "auroc_mean"], ascending=False).to_csv(tables / "pet_anchored_fusion_results.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions / "pet_anchored_fusion_oof.csv", index=False)
    write_json(metrics_dir / "run_manifest.json", {
        "config": str(args.config),
        "output_root": str(output_root),
        "variants": list(variants),
        "samples": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "folds": int(len(folds)),
        "split_seeds": list(config["splits"]["seeds"]),
        "mri_embedding_dim": int(embeddings.shape[1]),
        "cowsrad_input_dim": int(raw_cows.shape[1]),
        "pet_input_dim": int(raw_pet.shape[1]),
        "fusion_batch_size": int(fusion_cfg.get("batch_size", 64)),
    })
    print(pd.DataFrame(summaries).sort_values(["balanced_accuracy_mean", "auroc_mean"], ascending=False).to_string(index=False))
    print(f"Results: {tables / 'pet_anchored_fusion_results.csv'}")


if __name__ == "__main__":
    main()
