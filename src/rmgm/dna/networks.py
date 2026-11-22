"""Sequence networks for conditional R-MGM promoter generation."""

from __future__ import annotations

import math
import torch
from torch import nn

from .geometry import from_logits, tangent


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.conv = nn.Conv1d(width, width, 9, padding=4 * dilation, dilation=dilation)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.conv(self.norm(x)))


class SequenceCNN(nn.Module):
    def __init__(self, input_channels: int, width: int = 256, blocks: int = 20):
        super().__init__()
        if blocks < 1 or width < 4:
            raise ValueError("width and blocks must be positive")
        self.stem = nn.Conv1d(input_channels, width, 9, padding=4)
        dilations = (1, 1, 4, 16, 64)
        self.blocks = nn.Sequential(*(ResidualBlock(width, dilations[i % 5]) for i in range(blocks)))
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 4, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.stem(features.transpose(1, 2))
        return self.head(self.blocks(h)).transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, width: int = 256, blocks: int = 20):
        super().__init__()
        self.net = SequenceCNN(5, width, blocks)

    def forward(self, noise: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        return from_logits(self.net(torch.cat((noise, signal), dim=-1)))


class Critic(nn.Module):
    def __init__(self, width: int = 256, blocks: int = 20, frequencies: int = 8):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(frequencies))
        self.net = SequenceCNN(5 + 2 * frequencies, width, blocks)

    def forward(self, time: torch.Tensor, point: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        phase = 2 * math.pi * time * self.frequencies.to(time)
        features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        features = features.expand(-1, point.shape[1], -1)
        raw = self.net(torch.cat((point, signal, features), dim=-1))
        return tangent(point, raw)
