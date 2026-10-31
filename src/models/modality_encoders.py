from __future__ import annotations

from torch import nn


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    dims = [int(input_dim), *[int(v) for v in hidden_dims]]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(dropout)])
    layers.extend([nn.Linear(dims[-1], int(output_dim)), nn.LayerNorm(int(output_dim))])
    return nn.Sequential(*layers)


class CoWSRadEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.2, output_dim: int = 512, **_: object) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims or [512, 512], output_dim, dropout)

    def forward(self, x):
        return self.net(x)


class PETSummaryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.2, output_dim: int = 512, **_: object) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims or [256, 512], output_dim, dropout)

    def forward(self, x):
        return self.net(x)
