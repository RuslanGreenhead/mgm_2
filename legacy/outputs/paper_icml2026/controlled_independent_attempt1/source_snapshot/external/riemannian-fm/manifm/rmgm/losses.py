"""Loss functions for Riemannian Midpoint Generative Models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .geometry import sample_midpoint_path


def metric_inner(
    manifold: Any, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return one scalar inner product per sample.

    Some RFM manifold implementations return structured metric components
    rather than a flat ``(N,)`` tensor. MGM needs a scalar fiberwise Hilbert
    norm, so all non-batch dimensions are summed.
    """

    val = manifold.inner(x, u, v)
    if val.ndim == 0:
        return val.reshape(1)
    if val.ndim == 1:
        return val
    return val.reshape(val.shape[0], -1).sum(dim=-1)


def critic_loss(
    manifold: Any,
    generator,
    critic,
    z: torch.Tensor,
    x1: torch.Tensor,
    *,
    normalize_dim: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MGM tangent-regression update for the critic."""

    with torch.no_grad():
        x0 = generator(z)
        x_tilde, a_t, times = sample_midpoint_path(manifold, x0, x1)
    pred = critic(times.t, x_tilde)
    diff = pred - a_t
    loss_per_sample = metric_inner(manifold, x_tilde, diff, diff)
    loss = loss_per_sample.mean()
    if normalize_dim:
        loss = loss / x1.shape[-1]
    metrics = {
        "critic_loss": loss.detach(),
        "target_norm": metric_inner(manifold, x_tilde, a_t, a_t).mean().detach(),
        "pred_norm": metric_inner(manifold, x_tilde, pred, pred).mean().detach(),
    }
    return loss, metrics


def generator_loss(
    manifold: Any,
    generator,
    critic,
    z: torch.Tensor,
    x1: torch.Tensor,
    *,
    normalize_dim: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Variational MGM update for the one-step generator.

    For fixed critic, the generator descends
    ``E[2 <f, A>_g - ||f||_g^2]``.
    """

    x0 = generator(z)
    x_tilde, a_t, times = sample_midpoint_path(manifold, x0, x1)
    pred = critic(times.t, x_tilde)
    value = 2.0 * metric_inner(manifold, x_tilde, pred, a_t)
    value = value - metric_inner(manifold, x_tilde, pred, pred)
    loss = value.mean()
    if normalize_dim:
        loss = loss / x1.shape[-1]
    metrics = {
        "generator_loss": loss.detach(),
        "generator_value": value.mean().detach(),
        "sample_norm": metric_inner(manifold, x_tilde, a_t, a_t).mean().detach(),
    }
    return loss, metrics
