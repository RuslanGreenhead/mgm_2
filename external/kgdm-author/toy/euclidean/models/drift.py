import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.special import kve as bessel_kve


class EuclideanDrift(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dim_noise: int = 3,
        device: Union[str, torch.device] = "cpu",
        *,
        temp: float = 0.35,
        max_step: float = 1.0,
        ignore_self_neg: bool = True,
        eps: float = 1e-8,
        update_type: str = "base",
        nu: float = 1.5,
    ):
        super().__init__()
        if update_type not in ("base", "grad"):
            raise ValueError(f"unknown update_type '{update_type}'")
        if nu <= 0:
            raise ValueError(f"matern nu must be > 0, got {nu}")
        self.net = backbone
        self.dim_noise = dim_noise
        self.device = device
        self.temp = float(temp)
        self.max_step = float(max_step)
        self.ignore_self_neg = ignore_self_neg
        self.eps = float(eps)
        self.update_type = update_type
        self.nu = float(nu)

    def sample(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or self.device
        z = torch.randn(batch_size, self.dim_noise, device=device)
        return self.net(z)

    def _matern_kernel_and_grad(self, dist: torch.Tensor, sigma: float):
        """Return (k, grad_coef) for the Matérn-ν kernel s.t.
        grad_x k(x, y) = grad_coef * (y - x).

        Closed forms for ν ∈ {0.5, 1.5, 2.5} and the Gaussian limit ν=∞;
        general ν via the Bessel-function formula in `_matern_general`.
        """
        nu = self.nu
        if nu == 0.5:
            k = torch.exp(-(dist / sigma))
            grad_coef = k / (sigma * dist.clamp_min(self.eps))
        elif nu == 1.5:
            r = math.sqrt(3.0) * dist / sigma
            e = torch.exp(-r)
            k = (1.0 + r) * e
            grad_coef = (3.0 / (sigma ** 2)) * e
        elif nu == 2.5:
            r = math.sqrt(5.0) * dist / sigma
            e = torch.exp(-r)
            k = (1.0 + r + (r ** 2) / 3.0) * e
            grad_coef = (5.0 / (3.0 * sigma ** 2)) * (1.0 + r) * e
        elif math.isinf(nu):
            k = torch.exp(-0.5 * (dist / sigma) ** 2)
            grad_coef = k / (sigma ** 2)
        else:
            k, grad_coef = self._matern_general(dist, sigma, nu)
        return k, grad_coef

    def _matern_general(self, dist: torch.Tensor, sigma: float, nu: float):
        # k(d) = c · u^nu · K_nu(u),  u = sqrt(2nu)·d/sigma,  c = 1/(Γ(nu)·2^(nu-1))
        # dk/du = -c · u^nu · K_(nu-1)(u)  →  grad_coef = c·u^nu·K_(nu-1)(u)·sqrt(2nu)/(σ·d)
        # uses kve = K_nu·exp(u) for numerical stability at large u; at very small u
        # with large nu, kve still overflows while the prefactor underflows (0·∞=NaN);
        # we detect those entries and replace them with the analytic small-u limits
        # (k→1; grad_coef→ν/((ν−1)σ²) for ν>1).
        d_np = dist.detach().cpu().numpy().astype(np.float64)
        d_safe = np.maximum(d_np, self.eps)
        u = math.sqrt(2.0 * nu) * d_safe / sigma
        log_c = -math.lgamma(nu) - (nu - 1.0) * math.log(2.0)
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            prefactor = np.exp(log_c + nu * np.log(u) - u)
            k_np = prefactor * bessel_kve(nu, u)
            grad_coef_np = (
                prefactor * bessel_kve(nu - 1.0, u)
                * math.sqrt(2.0 * nu) / (sigma * d_safe)
            )
        bad_k = ~np.isfinite(k_np)
        if bad_k.any():
            k_np = np.where(bad_k, 1.0, k_np)
        bad_g = ~np.isfinite(grad_coef_np)
        if bad_g.any():
            g_limit = nu / ((nu - 1.0) * sigma * sigma) if nu > 1.0 else 1.0 / (sigma * self.eps)
            grad_coef_np = np.where(bad_g, g_limit, grad_coef_np)
        zero_mask = d_np < self.eps
        k_np = np.where(zero_mask, 1.0, k_np)
        k = torch.from_numpy(k_np).to(device=dist.device, dtype=dist.dtype)
        grad_coef = torch.from_numpy(grad_coef_np).to(device=dist.device, dtype=dist.dtype)
        return k, grad_coef

    def compute_V(
        self,
        X: torch.Tensor,
        X_pos: torch.Tensor,
        X_neg: torch.Tensor,
        sigma: float,
    ):
        dist_pos = torch.cdist(X, X_pos)
        dist_neg = torch.cdist(X, X_neg)

        k_pos, grad_coef_pos = self._matern_kernel_and_grad(dist_pos, sigma)
        k_neg, grad_coef_neg = self._matern_kernel_and_grad(dist_neg, sigma)

        if self.ignore_self_neg and X.shape[0] == X_neg.shape[0]:
            eye = torch.eye(X.shape[0], device=X.device, dtype=torch.bool)
            k_neg = k_neg.masked_fill(eye, 0.0)
            grad_coef_neg = grad_coef_neg.masked_fill(eye, 0.0)

        diff_pos = X_pos.unsqueeze(0) - X.unsqueeze(1)  # [B, B, D]
        diff_neg = X_neg.unsqueeze(0) - X.unsqueeze(1)  # [B, B, D]

        if self.update_type == "base":
            V_pos = (k_pos.unsqueeze(-1) * diff_pos).sum(dim=1) / k_pos.sum(
                dim=1, keepdim=True
            ).clamp_min(self.eps)
            V_neg = (k_neg.unsqueeze(-1) * diff_neg).sum(dim=1) / k_neg.sum(
                dim=1, keepdim=True
            ).clamp_min(self.eps)
        else:  # "grad"
            grad_k_pos = grad_coef_pos.unsqueeze(-1) * diff_pos
            grad_k_neg = grad_coef_neg.unsqueeze(-1) * diff_neg
            V_pos = grad_k_pos.sum(dim=1) / k_pos.sum(dim=1, keepdim=True).clamp_min(self.eps)
            V_neg = grad_k_neg.sum(dim=1) / k_neg.sum(dim=1, keepdim=True).clamp_min(self.eps)

        V = V_pos - V_neg

        stats = {
            "dist_pos_mean": dist_pos.mean(),
            "dist_neg_mean": dist_neg.mean(),
            "k_pos_mean": k_pos.mean(),
            "k_neg_mean": k_neg.mean(),
            "V_pos_norm_mean": V_pos.norm(dim=-1).mean(),
            "V_neg_norm_mean": V_neg.norm(dim=-1).mean(),
            "V_norm_mean": V.norm(dim=-1).mean(),
        }
        return V, stats

    def get_loss(self, X_pos: torch.Tensor):
        X = self.sample(batch_size=X_pos.shape[0], device=X_pos.device)
        X_neg = X

        with torch.no_grad():
            V, stats = self.compute_V(X, X_pos, X_neg, sigma=self.temp)
            V_norm = V.norm(dim=-1, keepdim=True)
            V = V * (self.max_step / V_norm.clamp_min(self.max_step))
            X_tgt = X + V

        stats["X_norm_mean"] = X.norm(dim=-1).mean()
        stats["X_tgt_norm_mean"] = X_tgt.norm(dim=-1).mean()

        residual = X - X_tgt
        loss = residual.pow(2).sum(dim=-1).mean()
        return loss, stats
