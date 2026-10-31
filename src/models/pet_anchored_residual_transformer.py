from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ResidualFusionOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    correction_logits: torch.Tensor
    auxiliary_logits: torch.Tensor
    convex_weights: torch.Tensor
    pet_reliability: torch.Tensor
    keep_mask: torch.Tensor
    residual_scale: torch.Tensor


@dataclass
class ProbabilityAnchoredFusionOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    correction_logits: torch.Tensor
    auxiliary_logits: torch.Tensor
    convex_weights: torch.Tensor
    pet_reliability: torch.Tensor
    keep_mask: torch.Tensor
    residual_scale: torch.Tensor


class PETAnchoredResidualTransformer(nn.Module):
    """Small residual transformer over MRI, CoWS-Rad, and PET feature tokens.

    The convex combination of three auxiliary logits is the safe baseline. The
    transformer learns only a residual interaction correction anchored on the
    PET token. Initialising ``residual_scale`` to zero makes the initial model
    exactly equal to its learned convex late-fusion baseline.
    """

    modality_order = ("mri", "cowsrad", "pet")

    def __init__(
        self,
        mri_dim: int,
        cowsrad_dim: int,
        pet_dim: int,
        *,
        d_model: int = 128,
        heads: int = 4,
        ff_dim: int = 256,
        layers: int = 1,
        dropout: float = 0.15,
        modality_dropout: tuple[float, float, float] = (0.10, 0.10, 0.05),
        use_uncertainty_gate: bool = True,
        pet_uncertainty_indices: tuple[int, ...] = tuple(range(10, 18)),
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        self.projectors = nn.ModuleList([
            FeatureProjector(mri_dim, d_model, dropout),
            FeatureProjector(cowsrad_dim, d_model, dropout),
            FeatureProjector(pet_dim, d_model, dropout),
        ])
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.auxiliary_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(3)])
        self.convex_weight_logits = nn.Parameter(torch.zeros(3))
        self.correction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 1),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("modality_dropout", torch.tensor(modality_dropout, dtype=torch.float32))
        valid_uncertainty = tuple(i for i in pet_uncertainty_indices if 0 <= i < pet_dim)
        self.pet_uncertainty_indices = valid_uncertainty
        self.use_uncertainty_gate = bool(use_uncertainty_gate and valid_uncertainty)
        if self.use_uncertainty_gate:
            self.pet_gate = nn.Sequential(
                nn.Linear(len(valid_uncertainty), 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.pet_gate[-1].weight)
            nn.init.zeros_(self.pet_gate[-1].bias)
        else:
            self.pet_gate = None

    def _keep_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        keep = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        if self.training and bool((self.modality_dropout > 0).any()):
            probs = self.modality_dropout.to(device).unsqueeze(0).expand(batch_size, -1)
            keep = torch.rand(batch_size, 3, device=device) >= probs
            empty = ~keep.any(dim=1)
            if bool(empty.any()):
                # PET is the anchor and the strongest unimodal branch.
                keep[empty, 2] = True
        return keep

    def forward(self, mri: torch.Tensor, cowsrad: torch.Tensor, pet: torch.Tensor) -> ResidualFusionOutput:
        mri_token = self.projectors[0](mri)
        cowsrad_token = self.projectors[1](cowsrad)
        pet_token = self.projectors[2](pet)
        if self.pet_gate is not None:
            uncertainty = pet[:, self.pet_uncertainty_indices]
            # The factor is initialised to one and is constrained to (0, 2).
            reliability = 2.0 * torch.sigmoid(self.pet_gate(uncertainty))
        else:
            reliability = torch.ones(pet.shape[0], 1, device=pet.device, dtype=pet.dtype)
        pet_token = pet_token * reliability
        tokens = torch.stack([mri_token, cowsrad_token, pet_token], dim=1)
        tokens = tokens + self.modality_embedding.to(tokens.dtype)
        keep = self._keep_mask(tokens.shape[0], tokens.device)
        transformed = self.transformer(tokens, src_key_padding_mask=~keep)
        auxiliary = torch.cat([head(tokens[:, i]) for i, head in enumerate(self.auxiliary_heads)], dim=1)
        base_weights = torch.softmax(self.convex_weight_logits, dim=0).unsqueeze(0).expand_as(auxiliary)
        masked_weights = base_weights * keep.to(base_weights.dtype)
        masked_weights = masked_weights / masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        base_logits = (masked_weights * auxiliary).sum(dim=1)
        correction = self.correction_head(transformed[:, 2]).squeeze(1)
        logits = base_logits + self.residual_scale * correction
        return ResidualFusionOutput(
            logits=logits,
            base_logits=base_logits,
            correction_logits=correction,
            auxiliary_logits=auxiliary,
            convex_weights=torch.softmax(self.convex_weight_logits, dim=0),
            pet_reliability=reliability.squeeze(1),
            keep_mask=keep,
            residual_scale=self.residual_scale,
        )


class ProbabilityAnchoredTransformer(nn.Module):
    """Adaptive transformer that starts exactly from late probability averaging.

    The three input probabilities are produced by leakage-safe unimodal
    classifiers fitted inside the outer training fold. At initialization, the
    model is exactly their arithmetic mean. The transformer can subsequently
    learn sample-specific modality weights and a bounded residual correction.
    """

    modality_order = ("mri", "cowsrad", "pet")

    def __init__(
        self,
        mri_dim: int,
        cowsrad_dim: int,
        pet_dim: int,
        *,
        d_model: int = 64,
        heads: int = 4,
        ff_dim: int = 128,
        layers: int = 1,
        dropout: float = 0.15,
        modality_dropout: tuple[float, float, float] = (0.10, 0.10, 0.05),
        use_uncertainty_gate: bool = True,
        use_adaptive_weights: bool = True,
        pet_uncertainty_indices: tuple[int, ...] = tuple(range(10, 18)),
        max_residual_scale: float = 0.50,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        if max_residual_scale <= 0:
            raise ValueError("max_residual_scale must be positive")
        self.projectors = nn.ModuleList([
            FeatureProjector(mri_dim, d_model, dropout),
            FeatureProjector(cowsrad_dim, d_model, dropout),
            FeatureProjector(pet_dim, d_model, dropout),
        ])
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.use_adaptive_weights = bool(use_adaptive_weights)
        self.weight_head = nn.Linear(d_model, 1)
        # Zero initialization makes the initial softmax exactly uniform.
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self.correction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 1),
        )
        self.raw_residual_scale = nn.Parameter(torch.zeros(()))
        self.max_residual_scale = float(max_residual_scale)
        self.register_buffer(
            "modality_dropout", torch.tensor(modality_dropout, dtype=torch.float32)
        )
        valid_uncertainty = tuple(
            index for index in pet_uncertainty_indices if 0 <= index < pet_dim
        )
        self.pet_uncertainty_indices = valid_uncertainty
        self.use_uncertainty_gate = bool(
            use_uncertainty_gate and use_adaptive_weights and valid_uncertainty
        )
        if self.use_uncertainty_gate:
            self.pet_gate = nn.Sequential(
                nn.Linear(len(valid_uncertainty), 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            # Reliability starts at one, preserving exact uniform averaging.
            nn.init.zeros_(self.pet_gate[-1].weight)
            nn.init.zeros_(self.pet_gate[-1].bias)
        else:
            self.pet_gate = None

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.max_residual_scale * torch.tanh(self.raw_residual_scale)

    def _keep_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        keep = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
        if self.training and bool((self.modality_dropout > 0).any()):
            probabilities = self.modality_dropout.to(device).unsqueeze(0).expand(batch_size, -1)
            keep = torch.rand(batch_size, 3, device=device) >= probabilities
            empty = ~keep.any(dim=1)
            if bool(empty.any()):
                keep[empty, 2] = True
        return keep

    def forward(
        self,
        mri: torch.Tensor,
        cowsrad: torch.Tensor,
        pet: torch.Tensor,
        modality_probabilities: torch.Tensor,
    ) -> ProbabilityAnchoredFusionOutput:
        if modality_probabilities.ndim != 2 or modality_probabilities.shape[1] != 3:
            raise ValueError(
                "modality_probabilities must have shape [batch, 3] in MRI/CoWS-Rad/PET order"
            )
        tokens = torch.stack(
            [
                self.projectors[0](mri),
                self.projectors[1](cowsrad),
                self.projectors[2](pet),
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embedding.to(tokens.dtype)
        keep = self._keep_mask(tokens.shape[0], tokens.device)
        transformed = self.transformer(tokens, src_key_padding_mask=~keep)

        if self.pet_gate is not None:
            uncertainty = pet[:, self.pet_uncertainty_indices]
            reliability = 2.0 * torch.sigmoid(self.pet_gate(uncertainty))
        else:
            reliability = torch.ones(
                pet.shape[0], 1, device=pet.device, dtype=pet.dtype
            )

        if self.use_adaptive_weights:
            weight_logits = self.weight_head(transformed).squeeze(-1)
            if self.pet_gate is not None:
                weight_logits = weight_logits.clone()
                weight_logits[:, 2] = (
                    weight_logits[:, 2] + reliability.squeeze(1).clamp_min(1e-6).log()
                )
        else:
            weight_logits = torch.zeros(
                tokens.shape[:2], device=tokens.device, dtype=tokens.dtype
            )
        weight_logits = weight_logits.masked_fill(~keep, torch.finfo(weight_logits.dtype).min)
        weights = torch.softmax(weight_logits, dim=1)

        probabilities = modality_probabilities.to(tokens.dtype).clamp(1e-5, 1.0 - 1e-5)
        base_probability = (weights * probabilities).sum(dim=1).clamp(1e-5, 1.0 - 1e-5)
        base_logits = torch.logit(base_probability)
        pooled = (weights.unsqueeze(-1) * transformed).sum(dim=1)
        correction = self.correction_head(pooled).squeeze(1)
        logits = base_logits + self.residual_scale * torch.tanh(correction)
        auxiliary_logits = torch.logit(probabilities)
        return ProbabilityAnchoredFusionOutput(
            logits=logits,
            base_logits=base_logits,
            correction_logits=correction,
            auxiliary_logits=auxiliary_logits,
            convex_weights=weights,
            pet_reliability=reliability.squeeze(1),
            keep_mask=keep,
            residual_scale=self.residual_scale,
        )


class ConcatenationMLP(nn.Module):
    def __init__(self, mri_dim: int, cowsrad_dim: int, pet_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        total = mri_dim + cowsrad_dim + pet_dim
        self.net = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mri: torch.Tensor, cowsrad: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([mri, cowsrad, pet], dim=1)).squeeze(1)
