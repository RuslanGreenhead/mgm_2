"""Unified drift model on a Riemannian manifold (sphere or hyperboloid).

Two orthogonal axes give 2x2 = 4 variants per (manifold, dataset):

  geometry ∈ {"euclidean", "manifold"}
    euclidean: pairwise distances via L2 in the ambient space; tangent vectors
               are y - x (ambient); the update is additive x_hat = x + η V.
               This is the naïve baseline that ignores the geometry.

    manifold:  pairwise distances are geodesic; tangent vectors are log_x(y),
               which already live in T_x M; the update is x_hat = Exp_x(ηV).

  method ∈ {"base", "gradient"}
    base:      V is the kernel-weighted transport vector,
                 V = Σ_j W^+_j · tan(x, y_j^+) - Σ_k W^-_k · tan(x, y_k^-).

    gradient:  V uses the kernel score ∇_x log k(x,y) = ∇_x k(x,y) / k(x,y),
               which for k = exp(-d / T) equals tan(x, y) / (T · d).
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn

from manifolds.base import BaseManifold
from manifolds.hyperboloid import HyperboloidManifold
from manifolds.sphere import SphereManifold


class GeometricDrift(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        manifold: BaseManifold,
        dim_noise: int,
        dim_intrinsic: int,
        *,
        geometry: str = "manifold",
        method: str = "base",
        temperature: float = 0.5,
        eta: float = 1.0,
        ignore_self_neg: bool = True,
        max_step: float | None = 5.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if geometry not in ("euclidean", "manifold"):
            raise ValueError(f"geometry must be 'euclidean' or 'manifold', got {geometry!r}")
        if method not in ("base", "gradient"):
            raise ValueError(f"method must be 'base' or 'gradient', got {method!r}")

        self.net = backbone
        self.manifold = manifold
        self.dim_noise = dim_noise
        self.dim_intrinsic = dim_intrinsic
        self.dim_ambient = dim_intrinsic + 1
        self.geometry = geometry
        self.method = method
        self.temperature = temperature
        self.eta = eta
        self.ignore_self_neg = ignore_self_neg
        self.max_step = max_step
        self.eps = eps

    def sample(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return on-manifold samples in the ambient space [n, ambient_dim].

        Per manifold:
          - SphereManifold:      3D Gaussian noise -> net (3->3) -> L2-normalize.
          - HyperboloidManifold: 2D intrinsic noise -> wrap to H^2 -> net (3->3)
                                 -> unwrap -> wrap.
        """
        if isinstance(self.manifold, SphereManifold):
            e = torch.randn(n, self.dim_ambient, device=device)
            v = self.net(e)
            return v / v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        if isinstance(self.manifold, HyperboloidManifold):
            z = torch.randn(n, self.dim_intrinsic, device=device)
            e = self.manifold.wrap(z)
            v = self.net(e)
            return self.manifold.wrap(self.manifold.unwrap(v))
        raise TypeError(f"unsupported manifold: {type(self.manifold).__name__}")

    def random_noise(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Initial sample distribution before training (for visualizing step 0)."""
        if isinstance(self.manifold, SphereManifold):
            e = torch.randn(n, self.dim_ambient, device=device)
            return e / e.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        z = torch.randn(n, self.dim_intrinsic, device=device) * 0.3
        return self.manifold.wrap(z)

    def _pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.geometry == "manifold":
            return self.manifold.pairwise_distance(x, y)
        return torch.cdist(x, y, p=2)

    def _pairwise_tangent(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Vector from x[j] toward y[i], shape [B, B, D].

        manifold:  log_{x[j]}(y[i])       in T_{x[j]} M
        euclidean: y[i] - x[j]            in R^D
        """
        B, D = x.shape
        if self.geometry == "manifold":
            x_rep = x[:, None, :].expand(B, B, D).reshape(B * B, D)
            y_rep = y[None, :, :].expand(B, B, D).reshape(B * B, D)
            return self.manifold.log_map(x_rep, y_rep).reshape(B, B, D)
        return y[None, :, :].expand(B, B, D) - x[:, None, :].expand(B, B, D)

    def compute_V(
        self,
        x: torch.Tensor,
        yp: torch.Tensor,
        yn: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x.shape[0]
        device = x.device

        d_pos = self._pairwise_dist(x, yp)
        d_neg = self._pairwise_dist(x, yn)
        if self.ignore_self_neg:
            d_neg = d_neg + torch.eye(B, device=device, dtype=x.dtype) * 1e6

        # Bi-stochastic-style normalization (softmax over rows AND columns, geometric mean).
        logit_pos = -d_pos / self.temperature
        logit_neg = -d_neg / self.temperature
        logit = torch.cat([logit_neg, logit_pos], dim=1)        # [B, 2B]
        A_row = logit.softmax(dim=1)
        A_col = logit.softmax(dim=0)
        A = torch.sqrt(A_row * A_col + self.eps)
        A_neg, A_pos = A.split(B, dim=1)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        tan_pos = self._pairwise_tangent(x, yp)                 # [B, B, D]
        tan_neg = self._pairwise_tangent(x, yn)

        if self.method == "gradient":
            d_pos_safe = d_pos.clamp_min(self.eps)
            d_neg_safe = d_neg.clamp_min(self.eps)
            tan_pos = tan_pos / (self.temperature * d_pos_safe[..., None])
            tan_neg = tan_neg / (self.temperature * d_neg_safe[..., None])

        if self.ignore_self_neg:
            idx = torch.arange(B, device=device)
            tan_neg[idx, idx] = 0.0
            W_neg[idx, idx] = 0.0

        Vp = (W_pos[:, :, None] * tan_pos).sum(dim=1)           # [B, D]
        Vq = (W_neg[:, :, None] * tan_neg).sum(dim=1)
        V = Vp - Vq
        V_norm = torch.norm(V, dim=-1)

        if self.max_step is not None:
            scale = (self.max_step / V_norm.clamp_min(self.eps)).clamp(max=1.0)
            V = V * scale[:, None]
            V_norm = torch.norm(V, dim=-1)

        d_neg_finite = d_neg[d_neg < 1e5]
        stats = {
            "dist_pos_mean": d_pos.mean(),
            "dist_pos_std": d_pos.std(unbiased=False),
            "dist_neg_mean": d_neg_finite.mean() if d_neg_finite.numel() > 0 else d_neg.mean(),
            "dist_neg_std": d_neg_finite.std(unbiased=False) if d_neg_finite.numel() > 1 else torch.tensor(0.0, device=device),
            "V_norm_mean": V_norm.mean(),
            "V_norm_std": V_norm.std(unbiased=False),
        }
        return V, stats

    def _step(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if self.geometry == "euclidean":
            return x + self.eta * V
        # On the sphere, clamp eta so eta * ||V|| stays inside one chart (< pi).
        if isinstance(self.manifold, SphereManifold):
            Vn = V.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            eta_eff = torch.minimum(
                torch.tensor(self.eta, device=x.device, dtype=x.dtype),
                (torch.pi - self.eps) / Vn,
            )
            return self.manifold.exp_map(x, eta_eff * V)
        # Hyperboloid exp_map already clamps the norm internally.
        return self.manifold.exp_map(x, self.eta * V)

    def get_loss(self, y_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = y_pos.device
        B = y_pos.shape[0]

        x = self.sample(B, device=device)
        y_neg = x

        V, stats = self.compute_V(x, y_pos, y_neg)

        with torch.no_grad():
            x_hat = self._step(x, V)

        if self.geometry == "manifold":
            d = self.manifold.distance(x, x_hat)
            loss = (d ** 2).mean()
        else:
            loss = ((x_hat - x) ** 2).mean()

        return loss, stats
