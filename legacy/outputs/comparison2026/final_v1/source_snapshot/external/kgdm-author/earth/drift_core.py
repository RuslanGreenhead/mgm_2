"""Drift fields for generative drifting on S^2.

Three drift spaces share the same structure:
  - euclidean : double-softmax affinity in ambient R^3.
  - sphere    : double-softmax affinity with geodesic distance and log/exp maps.
  - spectral  : Nadaraya-Watson kernel score with the spectral Matern kernel,
                summed over a kappa lengthscale ladder.

Multi-temperature drift is a plain sum of per-temperature fields.
"""

from __future__ import annotations

import torch


# ===========================================================================
# Euclidean drift (ambient R^D).
# ===========================================================================

def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    T: float,
    dist_norm: bool = False,
    grad_form: bool = False,
    kernel: str = "laplace",
    drop_T: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    x:     [N, D]      generated points
    y_pos: [N_pos, D]  positives (real)
    y_neg: [N_neg, D]  negatives (== x)
    T:     kernel temperature
    dist_norm: divide kernel distances by their batch mean (scale-free T).
    grad_form: per-pair direction = unit(y - x) instead of the displacement.
        The direction uses the raw distance; only the kernel weight uses the
        normalized/masked distance.
    kernel: "laplace" -> logits -d/T (kernel-gradient direction = unit(y-x));
        "gauss" -> logits -d^2/T (kernel-gradient direction = y-x, so gradient
        and displacement form coincide up to the per-T constant).
    drop_T: drop the per-T constant of the kernel gradient (1/T laplace,
        2/T gauss); drop_T=False keeps it, weighting small temperatures more
        heavily in the multi-T sum. Only affects grad_form=True.
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]

    dist_pos = torch.cdist(x, y_pos)            # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)            # [N, N_neg]

    # kernel-weight distances: optionally scale-free, self-pair masked
    wd_pos, wd_neg = dist_pos, dist_neg
    if dist_norm:
        mean_d = torch.cat([dist_pos, dist_neg], dim=1).mean().clamp_min(1e-6)
        wd_pos = dist_pos / mean_d
        wd_neg = dist_neg / mean_d
    if kernel == "gauss":
        wd_pos = wd_pos.square()
        wd_neg = wd_neg.square()
    wd_neg = wd_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6

    logit_pos = -wd_pos / T
    logit_neg = -wd_neg / T
    logit = torch.cat([logit_pos, logit_neg], dim=1)   # [N, N_pos + N_neg]

    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = torch.sqrt(A_row * A_col)

    A_pos, A_neg = torch.split(A, [N_pos, A.shape[1] - N_pos], dim=1)

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)      # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)      # [N, N_neg]

    if not grad_form:
        drift_pos = W_pos @ y_pos                        # [N, D]
        drift_neg = W_neg @ y_neg                        # [N, D]
        return drift_pos - drift_neg

    diff_pos = y_pos.unsqueeze(0) - x.unsqueeze(1)       # [N, N_pos, D]
    diff_neg = y_neg.unsqueeze(0) - x.unsqueeze(1)       # [N, N_neg, D]
    if kernel == "gauss":
        dir_pos, dir_neg = diff_pos, diff_neg
    else:
        dir_pos = diff_pos / dist_pos.clamp_min(eps).unsqueeze(-1)
        dir_neg = diff_neg / dist_neg.clamp_min(eps).unsqueeze(-1)
    idx = torch.arange(N, device=x.device)
    W_neg = W_neg.clone(); W_neg[idx, idx] = 0.0
    dir_neg[idx, idx] = 0.0
    drift_pos = (W_pos.unsqueeze(-1) * dir_pos).sum(dim=1)
    drift_neg = (W_neg.unsqueeze(-1) * dir_neg).sum(dim=1)
    drift = drift_pos - drift_neg
    if not drop_T:
        drift = drift * ((2.0 if kernel == "gauss" else 1.0) / T)
    return drift


def compute_V_multi(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temps,
    dist_norm: bool = False,
    grad_form: bool = False,
    kernel: str = "laplace",
    drop_T: bool = True,
) -> torch.Tensor:
    """Sum of per-temperature drift fields."""
    V = torch.zeros_like(x)
    for T in temps:
        V = V + compute_V(x, y_pos, y_neg, T, dist_norm=dist_norm,
                          grad_form=grad_form, kernel=kernel, drop_T=drop_T)
    return V


# ===========================================================================
# Sphere S^2 geometry helpers.
# ===========================================================================

def sphere_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def sphere_project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (x * v).sum(dim=-1, keepdim=True) * x


def sphere_exp(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exp_x(v) on the unit sphere. x: [..., 3] normalized; v tangent at x."""
    x = sphere_normalize(x, eps)
    v = sphere_project_tangent(x, v)
    v_norm = v.norm(dim=-1, keepdim=True)
    direction = v / v_norm.clamp_min(eps)
    out = torch.cos(v_norm) * x + torch.sin(v_norm) * direction
    approx = sphere_normalize(x + v, eps)
    out = torch.where(v_norm < 1e-5, approx, out)
    return sphere_normalize(out, eps)


def sphere_log(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log_x(y) on S^2. x, y: [..., 3]; returns tangent vectors at x."""
    x = sphere_normalize(x, eps)
    y = sphere_normalize(y, eps)
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)
    u = y - dot * x
    u_norm = u.norm(dim=-1, keepdim=True)
    out = theta / u_norm.clamp_min(eps) * u
    small = theta < 1e-5
    out = torch.where(small, sphere_project_tangent(x, y - x), out)
    return sphere_project_tangent(x, out)


def pairwise_sphere_log(x: torch.Tensor, y: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """x: [N, 3], y: [M, 3] -> [N, M, 3], entry [i, j] = Log_{x_i}(y_j)."""
    x = sphere_normalize(x, eps)
    y = sphere_normalize(y, eps)
    dot = (x @ y.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)        # [N, M]
    theta = torch.acos(dot)
    u = y.unsqueeze(0) - dot.unsqueeze(-1) * x.unsqueeze(1)  # [N, M, 3]
    u_norm = u.norm(dim=-1, keepdim=True)
    out = theta.unsqueeze(-1) / u_norm.clamp_min(eps) * u
    small = theta.unsqueeze(-1) < 1e-5
    first_order = sphere_project_tangent(x.unsqueeze(1),
                                         y.unsqueeze(0) - x.unsqueeze(1))
    out = torch.where(small, first_order, out)
    return sphere_project_tangent(x.unsqueeze(1), out)


def pairwise_geodesic(x: torch.Tensor, y: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """x: [N, 3], y: [M, 3] -> [N, M] geodesic angle."""
    x = sphere_normalize(x, eps)
    y = sphere_normalize(y, eps)
    dot = (x @ y.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot)


# ===========================================================================
# Sphere drift on S^2: the same double-softmax recipe with geodesic distance
# for the kernel and log-map tangents for the drift.
# ===========================================================================

def compute_V_sphere(
    x: torch.Tensor,        # [N, 3] sphere points (generated)
    y_pos: torch.Tensor,    # [N_pos, 3] sphere points (positives)
    y_neg: torch.Tensor,    # [N_neg, 3] sphere points (negatives == x)
    T: float,
    dist_norm: bool = False,
    grad_form: bool = False,
    kernel: str = "laplace",
    drop_T: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Riemannian double-softmax drift on S^2; returns tangent V at x.

    grad_form: per-pair direction = log_x(y) / d_geo (unit tangent); the
        direction uses the raw geodesic distance, the kernel weight the
        normalized/masked distance. grad_form=False -> log-map displacement.
    kernel: "laplace" -> logits -d_geo/T; "gauss" -> logits -d_geo^2/T
        (kernel-gradient direction = log_x(y), so gradient and displacement
        form coincide up to the per-T constant).
    drop_T: as in compute_V.
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]

    dist_pos = pairwise_geodesic(x, y_pos, eps)        # [N, N_pos]
    dist_neg = pairwise_geodesic(x, y_neg, eps)        # [N, N_neg]

    log_pos = pairwise_sphere_log(x, y_pos, eps)       # [N, N_pos, 3]
    log_neg = pairwise_sphere_log(x, y_neg, eps)       # [N, N_neg, 3]

    if grad_form and kernel != "gauss":
        dir_pos = log_pos / dist_pos.clamp_min(eps).unsqueeze(-1)
        dir_neg = log_neg / dist_neg.clamp_min(eps).unsqueeze(-1)
        idx = torch.arange(N, device=x.device)
        dir_neg[idx, idx] = 0.0
    else:
        # displacement form; also the gauss kernel-gradient direction
        # (log_neg diagonal is already 0 since log_x(x) = 0)
        dir_pos, dir_neg = log_pos, log_neg

    # kernel-weight distances: optionally scale-free, self-pair masked
    wd_pos, wd_neg = dist_pos, dist_neg
    if dist_norm:
        mean_d = torch.cat([dist_pos, dist_neg], dim=1).mean().clamp_min(1e-6)
        wd_pos = dist_pos / mean_d
        wd_neg = dist_neg / mean_d
    if kernel == "gauss":
        wd_pos = wd_pos.square()
        wd_neg = wd_neg.square()
    wd_neg = wd_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6

    logit = torch.cat([-wd_pos / T, -wd_neg / T], dim=1)
    A = torch.sqrt(logit.softmax(dim=-1) * logit.softmax(dim=-2))
    A_pos, A_neg = torch.split(A, [N_pos, A.shape[1] - N_pos], dim=1)

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)      # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)      # [N, N_neg]

    V_pos = (W_pos.unsqueeze(-1) * dir_pos).sum(dim=1)  # [N, 3]
    V_neg = (W_neg.unsqueeze(-1) * dir_neg).sum(dim=1)
    V = V_pos - V_neg
    if grad_form and not drop_T:
        V = V * ((2.0 if kernel == "gauss" else 1.0) / T)
    return sphere_project_tangent(x, V)


def compute_V_sphere_multi(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temps,
    dist_norm: bool = False,
    grad_form: bool = False,
    kernel: str = "laplace",
    drop_T: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sum of per-temperature sphere drift fields."""
    V = torch.zeros_like(x)
    for T in temps:
        V = V + compute_V_sphere(x, y_pos, y_neg, T, dist_norm=dist_norm,
                                 grad_form=grad_form, kernel=kernel,
                                 drop_T=drop_T, eps=eps)
    return sphere_project_tangent(x, V)


# ===========================================================================
# Spectral Matern kernel-score drift on S^2.
# Nadaraya-Watson score V = sum_j grad k / sum_j k (= grad_x log sum_j k(x, y_j))
# for positives minus negatives, summed over the kappa ladder.
# ===========================================================================

def compute_V_spectral_multi(
    x: torch.Tensor,        # [N, 3] sphere points (generated)
    y_pos: torch.Tensor,    # [N_pos, 3] sphere points (positives)
    y_neg: torch.Tensor,    # [N_neg, 3] sphere points (negatives == x)
    kernels,                # iterable of TabulatedMaternS2, one per kappa
    ignore_self_neg: bool = True,
    norm_scale: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sum of per-kappa Nadaraya-Watson kernel-score drifts; tangent V at x.

    norm_scale: multiply each per-kappa field by the kernel's e-folding angle
        (kernel.ell). The NW score has magnitude ~1/ell near the data, so this
        rescales every ladder rung to O(1) before summing (the drop_T analog
        for the spectral ladder).
    """
    x = sphere_normalize(x, eps)
    y_pos = sphere_normalize(y_pos, eps)
    y_neg = sphere_normalize(y_neg, eps)
    N = x.shape[0]

    V = torch.zeros_like(x)
    for kernel in kernels:
        k_pos = kernel.k_pairwise(x, y_pos)            # [N, N_pos]
        k_neg = kernel.k_pairwise(x, y_neg)            # [N, N_neg]
        grad_k_pos = kernel.grad_x_pairwise(x, y_pos)  # [N, N_pos, 3] (ambient)
        grad_k_neg = kernel.grad_x_pairwise(x, y_neg)  # [N, N_neg, 3]

        if ignore_self_neg and N == y_neg.shape[0]:
            eye = torch.eye(N, device=x.device, dtype=torch.bool)
            k_neg = k_neg.masked_fill(eye, 0.0)
            grad_k_neg = grad_k_neg.masked_fill(eye.unsqueeze(-1), 0.0)

        V_pos = grad_k_pos.sum(dim=1) / k_pos.sum(dim=1, keepdim=True).clamp_min(eps)
        V_neg = grad_k_neg.sum(dim=1) / k_neg.sum(dim=1, keepdim=True).clamp_min(eps)
        scale = kernel.ell if norm_scale else 1.0
        V = V + (V_pos - V_neg) * scale
    return sphere_project_tangent(x, V)
