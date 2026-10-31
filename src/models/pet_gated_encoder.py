from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.mri_cnn_transformer_encoder import ResidualDownsample3DBlock, _group_norm


class GatedPETImageEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        stem_channels: int = 16,
        block_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        output_dim: int = 512,
        dropout3d: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, stem_channels, 3, padding=1, bias=False),
            _group_norm(stem_channels),
            nn.GELU(),
        )
        channels = [stem_channels, *list(block_channels)]
        self.blocks = nn.Sequential(*[ResidualDownsample3DBlock(channels[i], channels[i + 1], 2, dropout3d) for i in range(len(channels) - 1)])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], output_dim), nn.LayerNorm(output_dim), nn.GELU())

    @classmethod
    def from_config(cls, cfg: dict) -> "GatedPETImageEncoder":
        return cls(**cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.blocks(self.stem(x))))
