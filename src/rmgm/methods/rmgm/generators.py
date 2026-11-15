"""Generator choices for the maintained sphere/torus R-MGM implementation.

The GMM is a local chart mixture, not a separate training objective. The
component selector is discrete, as in the historical implementation, so its
logits receive no pathwise gradient through generated samples.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rmgm.geometry import PointGenerator


class LocalGMMGenerator(nn.Module):
    """Trainable local Gaussian mixture mapped onto a sphere or flat torus."""

    def __init__(self, ops, latent_dim=16, components=4, scale=0.25):
        super().__init__()
        if components < 1 or scale <= 0:
            raise ValueError("GMM components and scale must be positive")
        if latent_dim < ops.dim + 1:
            raise ValueError("GMM latent_dim must allow a selector and tangent noise")
        self.ops = ops
        self.latent_dim = latent_dim
        self.components = components
        centers = torch.randn(components, ops.dim) * 0.1 if ops.kind == "sphere" else torch.rand(components, ops.dim) * (2 * math.pi)
        self.centers = nn.Parameter(centers)
        self.log_scale = nn.Parameter(torch.log(torch.expm1(torch.full((components, ops.dim), scale))))
        self.component_logits = nn.Parameter(torch.zeros(components))

    def forward(self, z):
        if z.shape[-1] < self.latent_dim:
            raise ValueError("Latent input is narrower than configured latent_dim")
        selector = 0.5 * (1 + torch.erf(z[..., 0] / math.sqrt(2)))
        edges = torch.softmax(self.component_logits, 0).cumsum(0)[:-1]
        index = torch.searchsorted(edges.to(selector), selector).clamp(max=self.components - 1)
        center = self.ops.proj(self.centers)[index]
        tangent = self.ops.tangent(center, z[..., 1:1 + self.ops.dim] * F.softplus(self.log_scale)[index])
        return self.ops.proj(self.ops.exp(center, tangent))


def make_generator(ops, cfg):
    kind = cfg.get("generator", "mlp")
    if kind == "mlp":
        return PointGenerator(ops, cfg.get("latent_dim", 16), cfg.get("width", 128), cfg.get("depth", 3))
    if kind == "gmm":
        return LocalGMMGenerator(ops, cfg.get("latent_dim", 16), cfg.get("gmm_components", 4), cfg.get("gmm_scale", 0.25))
    raise ValueError(f"Unknown R-MGM generator: {kind}")
