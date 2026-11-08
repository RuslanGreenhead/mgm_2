from __future__ import annotations

import math
import torch
import torch.nn as nn


class TabulatedMaternS2(nn.Module):
    """
    Table-backed spectral Matern kernel on S^2 for any (nu, kappa) present in
    a table file produced by src/matern_tables.py:
        f(theta) = sum_l w_l P_l(cos theta)   (kernel, normalized to f(0)=1)
        g(theta) = sum_l w_l P_l'(cos theta)  (ambient-gradient scalar)
    Both are precomputed in float64 at Lmax=6000 on a sqrt-spaced theta grid
    (built automatically on first use) and linearly interpolated here, so
    kernel evaluation is O(1) per pair.

    k(x, x) = 1; grad_x k(x, y) = g(x.y) * y is the ambient gradient, with
    tangent projection left to the caller.

    self.ell is the kernel's e-folding angle (where f decays to 1/e), used by
    compute_V_spectral_multi(norm_scale=True) to rescale per-kappa fields.
    """

    def __init__(
        self,
        nu: float = 2.5,
        kappa: float = 1.0,
        table_path: str = "matern_tables.pt",
        eps: float = 1e-7,
    ):
        super().__init__()
        tab = torch.load(table_path, weights_only=True)
        key = f"{float(nu)}_{float(kappa)}"
        if key not in tab["f"]:
            raise ValueError(
                f"(nu={nu}, kappa={kappa}) not in {table_path} "
                f"(has {sorted(tab['f'])}); see src/matern_tables.py"
            )
        self.nu, self.kappa, self.eps = float(nu), float(kappa), float(eps)
        self.n = int(tab["n"])
        self.register_buffer("f_tab", tab["f"][key].float())
        self.register_buffer("g_tab", tab["g"][key].float())
        self.ell = self._e_folding_angle(tab["f"][key])

    def _e_folding_angle(self, f: torch.Tensor) -> float:
        """Angle where the kernel decays to 1/e, interpolated from the table."""
        target = math.exp(-1.0)
        idx = int((f < target).long().argmax())
        theta = lambda i: (i / (self.n - 1)) ** 2 * math.pi
        f0, f1 = float(f[idx - 1]), float(f[idx])
        frac = (f0 - target) / (f0 - f1)
        return theta(idx - 1) + frac * (theta(idx) - theta(idx - 1))

    def _interp(self, table: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Linear interpolation on the sqrt-spaced theta grid
        theta_i = (i / (n-1))^2 * pi, whose inverse is analytic."""
        theta = torch.acos(z)
        u = torch.sqrt(theta / math.pi).clamp(0.0, 1.0) * (self.n - 1)
        i0 = u.long().clamp(0, self.n - 2)
        frac = u - i0
        return table[i0] * (1.0 - frac) + table[i0 + 1] * frac

    @torch.no_grad()
    def k_pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """X: [B, 3], Y: [N, 3] unit-norm -> [B, N]."""
        out_dtype = X.dtype
        z = (X.float() @ Y.float().T).clamp(-1.0 + self.eps, 1.0 - self.eps)
        return self._interp(self.f_tab, z).to(out_dtype)

    @torch.no_grad()
    def grad_x_pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Ambient grad_x k(x, y) = g(x.y) * y. X: [B, 3], Y: [N, 3] -> [B, N, 3]."""
        out_dtype = X.dtype
        Xf, Yf = X.float(), Y.float()
        z = (Xf @ Yf.T).clamp(-1.0 + self.eps, 1.0 - self.eps)
        scalar = self._interp(self.g_tab, z)  # [B, N]
        return (scalar.unsqueeze(-1) * Yf.unsqueeze(0)).to(out_dtype)
