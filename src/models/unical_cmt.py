from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.models.modality_encoders import CoWSRadEncoder, PETSummaryEncoder
from src.models.mri_cnn_transformer_encoder import MRICNNTransformerEncoder
from src.models.pet_gated_encoder import GatedPETImageEncoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 512, heads: int = 8, ff_dim: int = 1024, dropout: float = 0.15, norm_first: bool = True) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))

    def forward(self, query: torch.Tensor, modality_tokens: torch.Tensor) -> torch.Tensor:
        if modality_tokens.ndim == 2:
            modality_tokens = modality_tokens.unsqueeze(1)
        q = self.q_norm(query) if self.norm_first else query
        kv = self.kv_norm(modality_tokens) if self.norm_first else modality_tokens
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + self.drop(attended)
        if not self.norm_first:
            query = self.q_norm(query)
        ff_in = self.ff_norm(query) if self.norm_first else query
        query = query + self.drop(self.ff(ff_in))
        if not self.norm_first:
            query = self.ff_norm(query)
        return query


class CascadedModalityTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        query_tokens: int = 1,
        modality_order: list[str] | None = None,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.15,
        norm_first: bool = True,
        modality_dropout: dict[str, float] | None = None,
        never_drop_all_modalities: bool = True,
    ) -> None:
        super().__init__()
        self.modality_order = modality_order or ["raw_mri", "cowsrad", "pet_summary", "gated_pet_image"]
        self.modality_dropout = modality_dropout or {}
        self.never_drop_all_modalities = never_drop_all_modalities
        self.query = nn.Parameter(torch.zeros(1, query_tokens, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleDict({name: CrossAttentionBlock(d_model, heads, ff_dim, dropout, norm_first) for name in self.modality_order})

    def _drop_decisions(self, embeddings: dict[str, torch.Tensor]) -> dict[str, bool]:
        keep = {name: name in embeddings and embeddings[name] is not None for name in self.modality_order}
        if self.training:
            for name in list(keep):
                if keep[name] and torch.rand((), device=embeddings[name].device).item() < float(self.modality_dropout.get(name, 0.0)):
                    keep[name] = False
            if self.never_drop_all_modalities and not any(keep.values()):
                available = [name for name in self.modality_order if name in embeddings and embeddings[name] is not None]
                if available:
                    keep[available[0]] = True
        return keep

    def forward(self, embeddings: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, bool]]:
        first = next(v for v in embeddings.values() if v is not None)
        query = self.query.expand(first.shape[0], -1, -1)
        states: dict[str, torch.Tensor] = {}
        keep = self._drop_decisions(embeddings)
        active: list[str] = []
        for name in self.modality_order:
            if not keep.get(name, False):
                continue
            query = self.blocks[name](query, embeddings[name])
            active.append(name)
            key = "_".join(active)
            states[key] = query[:, 0]
        states["final"] = query[:, 0]
        return query[:, 0], states, keep


class UNICALCMTClassifier(nn.Module):
    def __init__(
        self,
        cowsrad_input_dim: int,
        pet_summary_input_dim: int | None = None,
        cfg: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
        use_auxiliary_heads: bool = True,
        use_modality_dropout: bool = True,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        model_cfg = cfg.get("model", cfg)
        d_model = int(model_cfg.get("d_model", 512))
        self.modalities = modalities or ["raw_mri", "cowsrad", "pet_summary", "gated_pet_image"]
        self.use_auxiliary_heads = use_auxiliary_heads
        self.raw_mri_encoder = MRICNNTransformerEncoder.from_config(model_cfg.get("mri_cnn_transformer", {}))
        self.cowsrad_encoder = CoWSRadEncoder(cowsrad_input_dim, **model_cfg.get("cowsrad_encoder", {}))
        self.pet_summary_encoder = PETSummaryEncoder(pet_summary_input_dim or 1, **model_cfg.get("pet_summary_encoder", {}))
        self.pet_image_encoder = GatedPETImageEncoder.from_config(model_cfg.get("pet_image_encoder", {}))
        drop_cfg = model_cfg.get("modality_dropout", {})
        dropout = {
            "raw_mri": float(drop_cfg.get("raw_mri_p", 0.0)) if use_modality_dropout else 0.0,
            "cowsrad": float(drop_cfg.get("cowsrad_p", 0.0)) if use_modality_dropout else 0.0,
            "pet_summary": float(drop_cfg.get("pet_summary_p", 0.0)) if use_modality_dropout else 0.0,
            "gated_pet_image": float(drop_cfg.get("gated_pet_image_p", 0.0)) if use_modality_dropout else 0.0,
        }
        fusion_cfg = model_cfg.get("cmt_fusion", {})
        self.fusion = CascadedModalityTransformer(
            d_model=d_model,
            query_tokens=int(fusion_cfg.get("query_tokens", 1)),
            modality_order=list(fusion_cfg.get("modality_order", ["raw_mri", "cowsrad", "pet_summary", "gated_pet_image"])),
            heads=int(fusion_cfg.get("heads", 8)),
            ff_dim=int(fusion_cfg.get("ff_dim", 1024)),
            dropout=float(fusion_cfg.get("dropout", 0.15)),
            norm_first=bool(fusion_cfg.get("norm_first", True)),
            modality_dropout=dropout,
            never_drop_all_modalities=bool(drop_cfg.get("never_drop_all_modalities", True)),
        )
        num_classes = int(model_cfg.get("num_classes", 2))
        self.final_head = nn.Linear(d_model, num_classes)
        self.aux_heads = nn.ModuleDict({
            "raw_mri": nn.Linear(d_model, num_classes),
            "raw_mri_cowsrad": nn.Linear(d_model, num_classes),
            "raw_mri_cowsrad_pet_summary": nn.Linear(d_model, num_classes),
        })

    def encode_modalities(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        embeddings: dict[str, torch.Tensor] = {}
        availability: dict[str, torch.Tensor] = {}
        if "raw_mri" in self.modalities and batch.get("mri") is not None:
            embeddings["raw_mri"] = self.raw_mri_encoder(batch["mri"])
            availability["raw_mri"] = torch.ones(batch["mri"].shape[0], dtype=torch.bool, device=batch["mri"].device)
        if "cowsrad" in self.modalities and batch.get("cowsrad") is not None:
            embeddings["cowsrad"] = self.cowsrad_encoder(batch["cowsrad"])
            availability["cowsrad"] = torch.ones(batch["cowsrad"].shape[0], dtype=torch.bool, device=batch["cowsrad"].device)
        if "pet_summary" in self.modalities and batch.get("pet_summary") is not None:
            mask = batch.get("pet_summary_available")
            if mask is None or bool(mask.any().item()):
                embeddings["pet_summary"] = self.pet_summary_encoder(batch["pet_summary"])
                availability["pet_summary"] = mask.bool() if mask is not None else torch.ones(batch["pet_summary"].shape[0], dtype=torch.bool, device=batch["pet_summary"].device)
        if "gated_pet_image" in self.modalities and batch.get("gated_pet") is not None:
            mask = batch.get("gated_pet_available")
            if mask is None or bool(mask.any().item()):
                embeddings["gated_pet_image"] = self.pet_image_encoder(batch["gated_pet"])
                availability["gated_pet_image"] = mask.bool() if mask is not None else torch.ones(batch["gated_pet"].shape[0], dtype=torch.bool, device=batch["gated_pet"].device)
        return embeddings, availability

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        embeddings, availability = self.encode_modalities(batch)
        if not embeddings:
            raise ValueError("No modalities are available for UNICAL CMT forward pass.")
        final_state, states, keep = self.fusion(embeddings)
        aux = {}
        if self.use_auxiliary_heads:
            for key, head in self.aux_heads.items():
                if key in states:
                    aux[key] = head(states[key])
        return {"logits": self.final_head(final_state), "auxiliary_logits": aux, "availability": availability, "modality_keep": keep}


def unical_cmt_loss(outputs: dict[str, Any], labels: torch.Tensor, class_weights: torch.Tensor | None = None, aux_weights: dict[str, float] | None = None, label_smoothing: float = 0.0) -> torch.Tensor:
    loss = F.cross_entropy(outputs["logits"], labels, weight=class_weights, label_smoothing=label_smoothing)
    aux_weights = aux_weights or {}
    for name, logits in outputs.get("auxiliary_logits", {}).items():
        loss = loss + float(aux_weights.get(name, 0.0)) * F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
    return loss
