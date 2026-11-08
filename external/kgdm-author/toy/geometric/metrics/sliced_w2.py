"""Sliced Wasserstein-2 between two empirical distributions in R^D.

Project both samples onto random unit directions, compute the 1D W2 (sort + L2
diff) per direction, and average. Works in the ambient space, so it is valid
for both sphere and hyperboloid samples even when one set drifts off-manifold
(useful for comparing the euclidean baseline).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def sliced_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 128,
    generator: torch.Generator | None = None,
) -> float:
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"dim mismatch: {x.shape} vs {y.shape}")
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]

    D = x.shape[1]
    proj = torch.randn(n_proj, D, device=x.device, dtype=x.dtype, generator=generator)
    proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    x_proj = x @ proj.t()                                # [n, n_proj]
    y_proj = y @ proj.t()
    x_sorted, _ = x_proj.sort(dim=0)
    y_sorted, _ = y_proj.sort(dim=0)
    w2_per_dir = ((x_sorted - y_sorted) ** 2).mean(dim=0).sqrt()
    return float(w2_per_dir.mean().item())
