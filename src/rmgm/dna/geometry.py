"""Fisher--Rao square-root geometry on a product of four-state simplices."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def target_sphere(labels: torch.Tensor) -> torch.Tensor:
    """Map A/C/G/T labels to boundary points; label 4 denotes an unknown base."""
    if labels.ndim != 2 or torch.any((labels < 0) | (labels > 4)):
        raise ValueError("Expected labels [B,L] with values 0..4")
    one_hot = F.one_hot(labels.long().clamp_max(3), 4).to(torch.float32)
    return torch.where((labels == 4)[..., None], torch.full_like(one_hot, 0.5), one_hot)


def from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Square-root probabilities with a stable derivative for sharp logits."""
    # sqrt(softmax) differentiates through sqrt(0) when softmax underflows.
    # exp(log_softmax / 2) is mathematically identical without that singularity.
    return torch.exp(0.5 * torch.log_softmax(logits.float(), dim=-1))


def tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (x * v).sum(dim=-1, keepdim=True) * x


def sphere_log(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    orth = y - dot * x
    radius = torch.linalg.vector_norm(orth, dim=-1, keepdim=True)
    angle = torch.atan2(radius, dot)
    scale = torch.where(radius < 1e-5, 1.0 + angle.square() / 6.0,
                        angle / radius.clamp_min(1e-8))
    return tangent(x, orth * scale)


def sphere_exp(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v = tangent(x, v)
    r2 = v.square().sum(dim=-1, keepdim=True)
    r = r2.clamp_min(1e-16).sqrt()
    sinc = torch.where(r2 < 1e-8, 1.0 - r2 / 6.0 + r2.square() / 120.0,
                       torch.sin(r) / r)
    return x * torch.cos(r) + v * sinc


def path(x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest product geodesic and its velocity, both [B,L,4]."""
    delta = sphere_log(x0, x1)
    r = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    phase = time * r
    point = torch.cos(phase) * x0 + torch.where(
        r < 1e-5, time - time.pow(3) * r.square() / 6.0,
        torch.sin(phase) / r.clamp_min(1e-8),
    ) * delta
    velocity = -r * torch.sin(phase) * x0 + torch.cos(phase) * delta
    return point, tangent(point, velocity)


def mean_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Batch and position mean of the unit-sphere inner product."""
    return (u * v).sum(dim=-1).mean()


def fisher_rao_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Distance on the product manifold, with Fisher--Rao factor two."""
    angles = torch.atan2(torch.linalg.vector_norm(x - y, dim=-1),
                         torch.linalg.vector_norm(x + y, dim=-1)) * 2.0
    return 2.0 * torch.linalg.vector_norm(angles, dim=-1)
