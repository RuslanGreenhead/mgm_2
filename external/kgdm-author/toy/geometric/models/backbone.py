from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBackbone(nn.Module):
    """Minimal MLP: vector noise -> flat output."""

    def __init__(
        self, dim_noise: int = 2, dim_data: int = 2, hidden: int = 64, depth: int = 3    ):
        super().__init__()
        layers = []
        in_dim = dim_noise
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim_data))
        self.net = nn.Sequential(*layers)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)

