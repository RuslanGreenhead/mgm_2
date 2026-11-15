"""Tangent-valued critics for Riemannian MGM."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class TimeFourier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    @property
    def out_dim(self) -> int:
        return 2 * self.n_features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        freqs = torch.arange(
            1, self.n_features + 1, dtype=t.dtype, device=t.device
        ).reshape(1, -1)
        phases = 2.0 * math.pi * t * freqs
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "silu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        act = _activation(activation)
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.extend([nn.Linear(d_in, hidden_dim), act()])
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), act()])
            layers.append(nn.Linear(hidden_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TangentCritic(nn.Module):
    """Ambient MLP followed by projection to ``T_x M``."""

    def __init__(
        self,
        manifold: Any,
        dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "silu",
        fourier: int = 8,
        metric_normalize: bool = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.metric_normalize = metric_normalize
        self.time_features = TimeFourier(fourier) if fourier > 0 else None
        time_dim = self.time_features.out_dim if self.time_features else 1
        self.net = MLP(
            dim + time_dim,
            dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        t_features = self.time_features(t) if self.time_features else t
        raw = self.net(torch.cat([x, t_features.to(x)], dim=-1))
        v = self.manifold.proju(x, raw)
        if self.metric_normalize and hasattr(self.manifold, "metric_normalized"):
            v = self.manifold.metric_normalized(x, v)
        return v


def build_critic(cfg: Optional[Any], manifold: Any, dim: int) -> TangentCritic:
    cfg = _critic_cfg(cfg)
    return TangentCritic(
        manifold=manifold,
        dim=dim,
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", _cfg_get(cfg, "d_model", 256))),
        num_layers=int(_cfg_get(cfg, "num_layers", 4)),
        activation=str(_cfg_get(cfg, "activation", _cfg_get(cfg, "actfn", "silu"))),
        fourier=int(_cfg_get(cfg, "fourier", 8) or 0),
        metric_normalize=bool(_cfg_get(cfg, "metric_normalize", False)),
    )


def _activation(name: str):
    normalized = str(name).lower()
    if normalized in {"silu", "swish"}:
        return nn.SiLU
    if normalized == "relu":
        return nn.ReLU
    if normalized == "gelu":
        return nn.GELU
    if normalized in {"srelu", "softplus"}:
        return nn.Softplus
    if normalized == "sine":
        return Sine
    if normalized == "tanh":
        return nn.Tanh
    raise ValueError(f"Unknown activation: {name}")


def _critic_cfg(cfg: Optional[Any]) -> Any:
    if cfg is None:
        return {}
    nested = _cfg_get(cfg, "critic", None)
    if nested is not None:
        return nested
    rmgm = _cfg_get(cfg, "rmgm", None)
    nested = _cfg_get(rmgm, "critic", None)
    return cfg if nested is None else nested


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(cfg, key, default)
