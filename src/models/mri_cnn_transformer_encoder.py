from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualDownsample3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.proj = (
            nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), _group_norm(out_channels))
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class MRICNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        stem_channels: int = 16,
        block_channels: list[int] | tuple[int, ...] = (32, 64, 128, 256),
        downsample_stride: int = 2,
        dropout3d: float = 0.1,
        d_model: int = 512,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        output_dim: int = 512,
        return_tokens: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.return_tokens = return_tokens
        self.d_model = int(d_model)
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, stem_channels, 3, padding=1, bias=False),
            _group_norm(stem_channels),
            nn.GELU(),
            nn.Conv3d(stem_channels, stem_channels, 3, padding=1, bias=False),
            _group_norm(stem_channels),
            nn.GELU(),
        )
        channels = [stem_channels, *list(block_channels)]
        self.blocks = nn.Sequential(
            *[
                ResidualDownsample3DBlock(channels[i], channels[i + 1], stride=downsample_stride, dropout=dropout3d)
                for i in range(len(channels) - 1)
            ]
        )
        self.patch_projection = nn.Linear(channels[-1], self.d_model)
        self.base_grid = (9, 11, 9)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.d_model, *self.base_grid))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.proj = nn.Sequential(nn.Linear(self.d_model, output_dim), nn.LayerNorm(output_dim))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MRICNNTransformerEncoder":
        return cls(
            input_channels=int(cfg.get("input_channels", 1)),
            stem_channels=int(cfg.get("stem_channels", 16)),
            block_channels=list(cfg.get("block_channels", [32, 64, 128, 256])),
            downsample_stride=int(cfg.get("downsample_stride", 2)),
            dropout3d=float(cfg.get("dropout3d", 0.1)),
            d_model=int(cfg.get("patch_projection_dim", cfg.get("d_model", 512))),
            transformer_layers=int(cfg.get("transformer_layers", 2)),
            transformer_heads=int(cfg.get("transformer_heads", 8)),
            transformer_ff_dim=int(cfg.get("transformer_ff_dim", 1024)),
            transformer_dropout=float(cfg.get("transformer_dropout", 0.1)),
            output_dim=int(cfg.get("output_dim", 512)),
        )

    def _positional_tokens(self, grid: tuple[int, int, int]) -> torch.Tensor:
        pos = self.positional_embedding
        if tuple(pos.shape[-3:]) != grid:
            pos = F.interpolate(pos, size=grid, mode="trilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_tokens: bool | None = None) -> dict[str, torch.Tensor] | torch.Tensor:
        x = self.blocks(self.stem(x))
        grid = tuple(int(v) for v in x.shape[-3:])
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.patch_projection(tokens) + self._positional_tokens(grid).to(tokens.dtype)
        tokens = self.transformer(tokens)
        embedding = self.proj(self.norm(tokens.mean(dim=1)))
        if return_tokens if return_tokens is not None else self.return_tokens:
            return {"embedding": embedding, "patch_tokens": tokens}
        return embedding
