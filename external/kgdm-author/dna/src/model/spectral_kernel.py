"""
Spectral Matern kernels on spheres, in spherical-harmonic / addition-theorem
form. We implement S^2 (Legendre P_l, for the toy sandbox) and S^3 (Chebyshev
U_l, for DNA — each position of the (S^3_+)^L product manifold is a point on
S^3 in R^4).

Per the theory (Eq. 10), the truncated Matern kernel on S^{D-1} is
    k(x,y) = sum_{l=0}^{Lmax} w_l * C_l(x.y),
with the addition theorem giving:

  S^2 (D=3):  lambda_l = l(l+1),  C_l = P_l (Legendre),
              w_l = (2l+1)/(4 pi) * (kappa^2 + l(l+1))^{-(nu+1)}.
  S^3 (D=4):  lambda_l = l(l+2),  C_l = U_l (Chebyshev 2nd kind),
              w_l = (l+1)/(2 pi^2) * (kappa^2 + l(l+2))^{-(nu+3/2)}.

For nu = inf the Matern density becomes the heat kernel exp(-kappa^2 lambda/2).

We normalize POINTWISE so k(x,x) = 1 (P_l(1)=1, U_l(1)=l+1). This keeps every
per-position kernel value in (0, 1], so on the product manifold
    log k_prod(x,y) = sum_l log k(x_l, y_l)
is a sum of non-positive terms — numerically stable, no underflow.

Derivatives use the *differentiated* recurrence (clean, no z=+-1 singularity):
  Chebyshev:  U'_{l+1} = 2 U_l + 2z U'_l - U'_{l-1},   U'_0=0, U'_1=2.
  Legendre:   P'_{l+1} = (2l+1)(P_l + z P'_l)/(l+1) - l P'_{l-1}/(l+1), via
              differentiating the P recurrence;  P'_0=0, P'_1=1.
"""

from __future__ import annotations

import math
import torch


# --------------------------------------------------------------------------
# weight tables (pointwise-normalized so k(x,x)=1)
# --------------------------------------------------------------------------

def matern_weights_s3(nu: float, kappa: float, Lmax: int) -> torch.Tensor:
    """w_l for S^3 (Chebyshev-U), pointwise-normalized: sum_l w_l*(l+1) = 1."""
    ells = torch.arange(Lmax + 1, dtype=torch.float64)
    lamb = ells * (ells + 2.0)                      # eigenvalues l(l+2)
    coeff = (ells + 1.0) / (2.0 * math.pi ** 2)     # addition-theorem coeff
    if math.isinf(nu):
        w = coeff * torch.exp(-0.5 * kappa ** 2 * lamb)
    else:
        w = coeff * (kappa ** 2 + lamb) ** (-(nu + 1.5))
    w = w / (w * (ells + 1.0)).sum()                # k(x,x) = sum w_l U_l(1) = 1
    return w


def matern_weights_s2(nu: float, kappa: float, Lmax: int) -> torch.Tensor:
    """w_l for S^2 (Legendre), pointwise-normalized: sum_l w_l = 1."""
    ells = torch.arange(Lmax + 1, dtype=torch.float64)
    lamb = ells * (ells + 1.0)
    coeff = (2.0 * ells + 1.0) / (4.0 * math.pi)
    if math.isinf(nu):
        w = coeff * torch.exp(-0.5 * kappa ** 2 * lamb)
    else:
        w = coeff * (kappa ** 2 + lamb) ** (-(nu + 1.0))
    w = w / w.sum()                                 # k(x,x) = sum w_l P_l(1) = 1
    return w


# --------------------------------------------------------------------------
# incremental kernel value + d/dz, accumulated over l (memory-safe: never
# materializes the [Lmax, ...] stack — only O(1) buffers of z's shape). This is
# what the product-manifold DNA drift uses, where z is [N, M, L].
# --------------------------------------------------------------------------

def cheb_u_k_and_deriv(z: torch.Tensor, weights: torch.Tensor):
    """Chebyshev-U (S^3). Returns (k, dk) = (sum_l w_l U_l(z), sum_l w_l U_l'(z)).
    z: any shape, values in (-1, 1).  weights: [Lmax+1]."""
    w = weights.to(device=z.device, dtype=z.dtype)
    Lmax = w.shape[0] - 1
    U_prev = torch.ones_like(z)                 # U_0
    dU_prev = torch.zeros_like(z)               # U_0'
    k = w[0] * U_prev
    dk = w[0] * dU_prev
    if Lmax >= 1:
        U_cur = 2.0 * z                         # U_1
        dU_cur = torch.full_like(z, 2.0)        # U_1'
        k = k + w[1] * U_cur
        dk = dk + w[1] * dU_cur
        for l in range(1, Lmax):
            U_next = 2.0 * z * U_cur - U_prev
            dU_next = 2.0 * U_cur + 2.0 * z * dU_cur - dU_prev
            k = k + w[l + 1] * U_next
            dk = dk + w[l + 1] * dU_next
            U_prev, U_cur = U_cur, U_next
            dU_prev, dU_cur = dU_cur, dU_next
    return k, dk


# --------------------------------------------------------------------------
# lookup-table acceleration. k(z) and dk(z) are FIXED smooth 1-D functions once
# the weights are set, yet the recurrence above recomputes them via Lmax+1
# sequential elementwise passes over z EVERY training step (z is [N,M,L] ~1e7
# for DNA -> ~Lmax kernel launches over the full tensor, twice per step). We
# instead tabulate (k, dk) on a dense z-grid ONCE and linear-interpolate per
# step: one gather, no Python loop. A 16384-pt grid is float32-exact (rel err
# ~4e-6, validated in scripts.diag_spectral_lut). Drift V is computed under
# no_grad, so the (non-differentiable) interp is a sound drop-in.
# --------------------------------------------------------------------------

class KernelLUT:
    """Tabulated (k, dk) for a fixed weight vector; call like k_and_deriv(z)."""

    def __init__(self, weights: torch.Tensor, kfn, grid_size: int = 16384):
        # build the table in float64 for accuracy; cast to z's dtype/device lazily
        grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
        kg, dkg = kfn(grid, weights.to(torch.float64))
        self.G = grid_size
        self._kg64, self._dkg64 = kg, dkg          # canonical float64 table
        self.kg = self.dkg = None                  # device/dtype-matched cache
        self._key = None

    def __call__(self, z: torch.Tensor, weights=None):
        # signature mirrors cheb_u_k_and_deriv(z, weights); weights ignored here.
        # Match z's dtype/device exactly so the recurrence -> LUT swap is otherwise
        # transparent to the (float32, cuda) drift downstream.
        key = (z.device, z.dtype)
        if key != self._key:
            self.kg = self._kg64.to(device=z.device, dtype=z.dtype)
            self.dkg = self._dkg64.to(device=z.device, dtype=z.dtype)
            self._key = key
        t = (z + 1.0) * 0.5 * (self.G - 1)
        i0 = t.floor().long().clamp_(0, self.G - 2)
        frac = t - i0.to(z.dtype)
        k = self.kg[i0] * (1.0 - frac) + self.kg[i0 + 1] * frac
        dk = self.dkg[i0] * (1.0 - frac) + self.dkg[i0 + 1] * frac
        return k, dk


def make_cheb_lut(weights: torch.Tensor, grid_size: int = 16384) -> KernelLUT:
    """LUT for the S^3 Chebyshev-U kernel built from a Matern weight vector."""
    return KernelLUT(weights, cheb_u_k_and_deriv, grid_size)


def legendre_k_and_deriv(z: torch.Tensor, weights: torch.Tensor):
    """Legendre (S^2). Returns (k, dk) = (sum_l w_l P_l(z), sum_l w_l P_l'(z))."""
    w = weights.to(device=z.device, dtype=z.dtype)
    Lmax = w.shape[0] - 1
    P_prev = torch.ones_like(z)                 # P_0
    dP_prev = torch.zeros_like(z)               # P_0'
    k = w[0] * P_prev
    dk = w[0] * dP_prev
    if Lmax >= 1:
        P_cur = z.clone()                       # P_1
        dP_cur = torch.ones_like(z)             # P_1'
        k = k + w[1] * P_cur
        dk = dk + w[1] * dP_cur
        for l in range(1, Lmax):
            P_next = ((2 * l + 1) * z * P_cur - l * P_prev) / (l + 1)
            dP_next = ((2 * l + 1) * (P_cur + z * dP_cur) - l * dP_prev) / (l + 1)
            k = k + w[l + 1] * P_next
            dk = dk + w[l + 1] * dP_next
            P_prev, P_cur = P_cur, P_next
            dP_prev, dP_cur = dP_cur, dP_next
    return k, dk


# --------------------------------------------------------------------------
# standalone modules (full [B,N] stacks) — used by the toy / for validation,
# mirroring the reference S^2 implementation's API.
# --------------------------------------------------------------------------

class _SpectralSphere(torch.nn.Module):
    """k(X,Y) and ambient/manifold grad_X k for a sphere spectral kernel."""

    def __init__(self, weights: torch.Tensor, kfn, eps: float = 1e-7,
                 gradient_type: str = "manifold"):
        super().__init__()
        self.kfn = kfn
        self.eps = float(eps)
        self.gradient_type = gradient_type
        self.register_buffer("weights", weights.float())

    @torch.no_grad()
    def k_pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        z = (X.float() @ Y.float().T).clamp(-1.0 + self.eps, 1.0 - self.eps)
        k, _ = self.kfn(z, self.weights)
        return k.to(X.dtype)

    @torch.no_grad()
    def grad_x_pairwise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.k_and_gradx_pairwise(X, Y)[1]

    @torch.no_grad()
    def k_and_gradx_pairwise(self, X: torch.Tensor, Y: torch.Tensor):
        """k(X,Y) and grad_X k(X,Y) in a SINGLE recurrence pass. Returns ([B,N], [B,N,D])."""
        Xf, Yf = X.float(), Y.float()
        z = (Xf @ Yf.T).clamp(-1.0 + self.eps, 1.0 - self.eps)   # [B, N]
        k, scalar = self.kfn(z, self.weights)                    # k and d/dz k
        if self.gradient_type == "ambient":
            g = scalar.unsqueeze(-1) * Yf.unsqueeze(0)
        elif self.gradient_type == "manifold":
            y_tan = Yf.unsqueeze(0) - z.unsqueeze(-1) * Xf.unsqueeze(1)
            g = scalar.unsqueeze(-1) * y_tan
        else:
            raise ValueError(f"Unknown gradient_type {self.gradient_type}")
        return k.to(X.dtype), g.to(X.dtype)


class SpectralMaternS3(_SpectralSphere):
    def __init__(self, nu=1.5, kappa=1.0, Lmax=50, eps=1e-7,
                 gradient_type="manifold"):
        super().__init__(matern_weights_s3(nu, kappa, Lmax),
                         cheb_u_k_and_deriv, eps, gradient_type)


class SpectralMaternS2(_SpectralSphere):
    def __init__(self, nu=1.5, kappa=1.0, Lmax=50, eps=1e-7,
                 gradient_type="manifold"):
        super().__init__(matern_weights_s2(nu, kappa, Lmax),
                         legendre_k_and_deriv, eps, gradient_type)
