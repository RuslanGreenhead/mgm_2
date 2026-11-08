from typing import Sequence

import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (32, 32),
        activation: str = "silu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation '{activation}', expected one of {list(_ACTIVATIONS)}"
            )
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
