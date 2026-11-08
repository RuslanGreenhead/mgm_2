"""Checkerboard tile accuracy on sphere or hyperboloid.

Unwraps on-manifold samples back to the 2D chart at the manifold origin and
returns the fraction that land in a black square of the source checkerboard.
Only meaningful for the *_checkerboard datasets.
"""
from __future__ import annotations

import torch
from manifolds.base import BaseManifold


@torch.no_grad()
def tile_acc(
    samples_ambient: torch.Tensor,
    manifold: BaseManifold,
    num_square: int = 4,
) -> float:
    uv = manifold.unwrap(samples_ambient)              # [N, 2] in [-1, 1]^2
    x_raw = (uv + 1.0) * 0.5 * num_square              # [N, 2] in [0, num_square]
    in_range = (
        (x_raw[:, 0] >= 0)
        & (x_raw[:, 0] < num_square)
        & (x_raw[:, 1] >= 0)
        & (x_raw[:, 1] < num_square)
    )
    cell_x = torch.floor(x_raw[:, 0]).long()
    cell_y = torch.floor(x_raw[:, 1]).long()
    is_black = ((cell_x + cell_y) % 2 == 1)
    return float((in_range & is_black).float().mean().item())
