"""
Euclidean drift core.

Design:
  - Bare-bones Algorithm-2 double-softmax affinity in RAW space.
  - Loss = ||x - sg(x + V)||²  → routes gradient as -V (move x along the drift).
  - Multi-temperature = plain SUM of per-T drift fields (no per-T force
    normalization).
  - Optional distance normalization (`dist_norm`): divide kernel distances by
    their batch mean so T is scale-free and transfers across data scales (§A.5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.spectral_kernel import cheb_u_k_and_deriv, make_cheb_lut

# LUT cache: weight vectors are built once per run and reused every step, so we
# tabulate (k, dk) once per distinct weight tensor instead of re-running the
# Lmax-step recurrence over the full [N,M,L] cosine tensor every step.
_LUT_CACHE: dict = {}


def _cheb_lut(weights: torch.Tensor):
    lut = _LUT_CACHE.get(weights)
    if lut is None:
        lut = make_cheb_lut(weights)
        _LUT_CACHE[weights] = lut
    return lut


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    T: float,
    dist_norm: bool = False,
    grad_form: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    x:     [N, D]      generated points
    y_pos: [N_pos, D]  positives (real)
    y_neg: [N_neg, D]  negatives (typically == x)
    T:     kernel temperature
    dist_norm: divide distances by their batch mean (scale-free T).
    grad_form: KGD kernel-gradient drift — per-pair direction = unit(y-x)
        instead of the raw displacement. RAW distance for the direction; only
        the kernel WEIGHT uses the normalized/masked distance. In ambient space
        this stays bounded only when the data is bounded (one-hot corners).
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]

    dist_pos = torch.cdist(x, y_pos)            # [N, N_pos] RAW
    dist_neg = torch.cdist(x, y_neg)            # [N, N_neg] RAW

    # kernel-WEIGHT distances: optionally scale-free + self-mask
    wd_pos, wd_neg = dist_pos, dist_neg
    if dist_norm:
        mean_d = torch.cat([dist_pos, dist_neg], dim=1).mean().clamp_min(1e-6)
        wd_pos = dist_pos / mean_d
        wd_neg = dist_neg / mean_d
    # ignore self (y_neg == x): push the diagonal to -inf logit
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

    # kernel-gradient form: per-pair direction = unit(y - x), raw distance
    diff_pos = y_pos.unsqueeze(0) - x.unsqueeze(1)       # [N, N_pos, D]
    diff_neg = y_neg.unsqueeze(0) - x.unsqueeze(1)       # [N, N_neg, D]
    dir_pos = diff_pos / dist_pos.clamp_min(eps).unsqueeze(-1)
    dir_neg = diff_neg / dist_neg.clamp_min(eps).unsqueeze(-1)
    idx = torch.arange(N, device=x.device)
    W_neg = W_neg.clone(); W_neg[idx, idx] = 0.0
    dir_neg[idx, idx] = 0.0
    drift_pos = (W_pos.unsqueeze(-1) * dir_pos).sum(dim=1)
    drift_neg = (W_neg.unsqueeze(-1) * dir_neg).sum(dim=1)
    return drift_pos - drift_neg


def compute_V_multi(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temps: tuple[float, ...],
    dist_norm: bool = False,
    grad_form: bool = False,
) -> torch.Tensor:
    """Plain SUM of per-temperature drift fields (no force normalization)."""
    V = torch.zeros_like(x)
    for T in temps:
        V = V + compute_V(x, y_pos, y_neg, T, dist_norm=dist_norm,
                          grad_form=grad_form)
    return V


def drift_loss(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    temps: tuple[float, ...],
    dist_norm: bool = False,
    grad_form: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    One drift step.
        x:     [N, D] generator output, MUST carry gradient.
        y_pos: [N_pos, D] positives (real).
    Negatives = x (detached). Returns (loss, info).
    """
    with torch.no_grad():
        V = compute_V_multi(x.detach(), y_pos, x.detach(), temps,
                            dist_norm=dist_norm, grad_form=grad_form)
        x_tgt = (x.detach() + V).detach()
    loss = F.mse_loss(x, x_tgt)
    info = {
        "loss": float(loss),
        "V_norm": float(V.norm(dim=-1).mean()),
        "x_norm": float(x.detach().norm(dim=-1).mean()),
    }
    return loss, info


# ===========================================================================
# Spherical drift on the product manifold (S^3_+)^L (Fisher-Rao / sqrt embed).
# Same verified recipe as the Euclidean core — bare-bones double-softmax
# weighting, multi-T PLAIN SUM (no per-T force norm), batch-mean dist-norm —
# but with Riemannian geometry: geodesic distance for the kernel, log-map
# tangent vectors for the drift direction, exp-map for the step. Manifold
# helpers ported from the KGD repo's dna_drift_sphere.py (well-tested).
# ===========================================================================

def _sphere_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def probs_to_sphere(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Simplex probs -> positive orthant of the sphere via phi(p)=sqrt(p)."""
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    return _sphere_normalize(p.sqrt(), eps)


def _project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project v onto the tangent space at x, positionwise."""
    return v - (x * v).sum(dim=-1, keepdim=True) * x


def _sphere_exp(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = _sphere_normalize(x, eps)
    v = _project_tangent(x, v)
    vn = v.norm(dim=-1, keepdim=True)
    direction = v / vn.clamp_min(eps)
    out = torch.cos(vn) * x + torch.sin(vn) * direction
    approx = _sphere_normalize(x + v, eps)
    out = torch.where(vn < 1e-5, approx, out)
    return _sphere_normalize(out, eps)


def _geodesic(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Positionwise geodesic angle on S^3. x,y: [...,L,4] -> [...,L]."""
    x = _sphere_normalize(x, eps)
    y = _sphere_normalize(y, eps)
    dot = (x * y).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot)


def _pairwise_geodesic(x: torch.Tensor, y: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    """Product-manifold geodesic distance. x:[N,L,4], y:[M,L,4] -> [N,M]."""
    x = _sphere_normalize(x, eps)
    y = _sphere_normalize(y, eps)
    dot = torch.einsum("nlc,mlc->nml", x, y).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)                                   # [N, M, L]
    return torch.sqrt(theta.pow(2).sum(dim=-1).clamp_min(eps))  # [N, M]


def _pairwise_log(x: torch.Tensor, y: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """Product log map. x:[N,L,4], y:[M,L,4] -> [N,M,L,4] tangent at x."""
    x = _sphere_normalize(x, eps)
    y = _sphere_normalize(y, eps)
    dot = torch.einsum("nlc,mlc->nml", x, y).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)                                   # [N, M, L]
    u = y.unsqueeze(0) - dot.unsqueeze(-1) * x.unsqueeze(1)    # [N, M, L, 4]
    u_norm = u.norm(dim=-1, keepdim=True)
    out = theta.unsqueeze(-1) / u_norm.clamp_min(eps) * u
    small = theta.unsqueeze(-1) < 1e-5
    first_order = _project_tangent(x.unsqueeze(1), y.unsqueeze(0) - x.unsqueeze(1))
    out = torch.where(small, first_order, out)
    return _project_tangent(x.unsqueeze(1), out)


def compute_V_sphere_multi(
    x: torch.Tensor,        # [N, L, 4] sphere points (generated)
    y_pos: torch.Tensor,    # [N_pos, L, 4] sphere points (positives)
    y_neg: torch.Tensor,    # [N_neg, L, 4] sphere points (negatives == x)
    temps: tuple[float, ...],
    dist_norm: bool = False,
    grad_form: bool = False,
    eps: float = 1e-8,
    kdist_pos: torch.Tensor | None = None,
    kdist_neg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Riemannian drift. Geometry (geodesic dist + log maps) is
    temperature-independent so it's computed ONCE; only the kernel weighting
    loops over temps, summed (no per-T force norm). Returns tangent V at x.

    grad_form: KGD kernel-gradient drift.
        Per-pair direction = log_x(y)/d_geo, the Frobenius-UNIT tangent (on the
        product manifold the stacked log has norm = product geodesic distance),
        i.e. ∇_x log k for k=exp(-d/T). Uses RAW geodesic distance for the
        direction; only the kernel WEIGHT uses the normalized/masked distance.
        grad_form=False -> displacement (raw log-map transport).

    kdist_pos/kdist_neg: OPTIONAL precomputed kernel-WEIGHT distances [N,N_pos]/
        [N,N_neg] (e.g. MAE-phi distances). When given, the locality (which
        positives/negatives weigh on x) lives in that FEATURE space instead of
        the sphere geodesic, while the transport DIRECTION stays the sphere
        log-map -> signal-aware locality without leaving the drift manifold."""
    N = x.shape[0]
    N_pos = y_pos.shape[0]

    dist_pos = _pairwise_geodesic(x, y_pos)            # [N, N_pos] RAW
    dist_neg = _pairwise_geodesic(x, y_neg)            # [N, N_neg] RAW

    log_pos = _pairwise_log(x, y_pos)                  # [N, N_pos, L, 4]
    log_neg = _pairwise_log(x, y_neg)                  # [N, N_neg, L, 4]

    if grad_form:
        # unit tangent direction (raw geodesic distance, scale-free by construction)
        dir_pos = log_pos / dist_pos.clamp_min(eps)[:, :, None, None]
        dir_neg = log_neg / dist_neg.clamp_min(eps)[:, :, None, None]
        idx = torch.arange(N, device=x.device)
        dir_neg[idx, idx] = 0.0                        # self-neg contributes nothing
    else:
        dir_pos, dir_neg = log_pos, log_neg

    # kernel-WEIGHT distances: phi-space if provided, else sphere geodesic.
    # Optionally scale-free + self-mask (NOT the direction).
    wd_pos = dist_pos if kdist_pos is None else kdist_pos
    wd_neg = dist_neg if kdist_neg is None else kdist_neg
    if dist_norm:
        mean_d = torch.cat([wd_pos, wd_neg], dim=1).mean().clamp_min(1e-6)
        wd_pos = wd_pos / mean_d
        wd_neg = wd_neg / mean_d
    wd_neg = wd_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6

    V = torch.zeros_like(x)
    for T in temps:
        logit = torch.cat([-wd_pos / T, -wd_neg / T], dim=1)
        A = torch.sqrt(logit.softmax(dim=-1) * logit.softmax(dim=-2))
        A_pos, A_neg = torch.split(A, [N_pos, A.shape[1] - N_pos], dim=1)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]
        V_pos = (W_pos.unsqueeze(-1).unsqueeze(-1) * dir_pos).sum(dim=1)
        V_neg = (W_neg.unsqueeze(-1).unsqueeze(-1) * dir_neg).sum(dim=1)
        V = V + (V_pos - V_neg)
    return _project_tangent(x, V)


def drift_loss_sphere(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    temps: tuple[float, ...],
    dist_norm: bool = False,
    grad_form: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Spherical drift step. x = sqrt(softmax(logits)) on (S^3_+)^L; positives
    are the one-hot corners; loss = geodesic²(x, sg(exp_x(V)))."""
    probs = F.softmax(logits, dim=-1)
    x = probs_to_sphere(probs)                          # [B, L, 4], grad
    with torch.no_grad():
        y_pos = probs_to_sphere(y_pos_onehot)
        x_d = x.detach()
        V = compute_V_sphere_multi(x_d, y_pos, x_d, temps, dist_norm=dist_norm,
                                   grad_form=grad_form)
        x_tgt = _sphere_exp(x_d, V).detach()
    geo = _geodesic(x, x_tgt)                            # [B, L]
    loss = geo.pow(2).mean()
    info = {
        "loss": float(loss),
        "V_norm": float(V.reshape(V.shape[0], -1).norm(dim=-1).mean()),
        "x_norm": float(x.detach().reshape(x.shape[0], -1).norm(dim=-1).mean()),
    }
    return loss, info


def drift_loss_mae_euclid(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    signal: torch.Tensor,        # [B, L, 1] conditioning signal (paired w/ batch)
    mae,                         # frozen PromoterMAE
    temps: tuple[float, ...],
    dist_norm: bool = True,
    grad_form: bool = False,
) -> tuple[torch.Tensor, dict]:
    """DEBUG: pure Euclidean drift ENTIRELY in the MAE phi feature space (no sphere).
    x = phi(softmax(logits), signal) carries grad THROUGH the frozen MAE back to the
    backbone; positives = phi(real, signal). Standard compute_V_multi / drift_loss in
    R^embed, so the weight AND the displacement share one coherent kernel (unlike the
    decoupled hybrid). Tests directly whether matching the MAE embedding yields good
    sequences (kmer) + conditioning (sei). grad path: phi_gen -> MAE encoder -> probs ->
    logits. Default grad_form=False (displacement form) -- unit-direction grad is unsafe
    in unbounded R^embed (feedback_grad_form_needs_manifold).

    phi_latR = paired ||phi_gen - phi_real||^2 / permuted: <1 <=> the drift is pulling
    each generated toward its OWN (same-signal) real => conditioning signal present."""
    probs = F.softmax(logits, dim=-1)
    _, phi_gen = mae.encode(probs, signal)              # [B, d] WITH grad through MAE
    with torch.no_grad():
        _, phi_real = mae.encode(y_pos_onehot, signal)  # [B, d]
    loss, info = drift_loss(phi_gen, phi_real, temps,
                            dist_norm=dist_norm, grad_form=grad_form)
    with torch.no_grad():
        B = phi_gen.shape[0]
        perm = torch.randperm(B, device=phi_gen.device)
        paired = (phi_gen - phi_real).pow(2).sum(-1).mean()
        permd = (phi_gen - phi_real[perm]).pow(2).sum(-1).mean()
        info["phi_latR"] = float(paired / permd.clamp_min(1e-9))
    return loss, info


def drift_loss_sphere_mae(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    signal: torch.Tensor,        # [B, L, 1] conditioning signal (paired w/ batch)
    mae,                         # frozen PromoterMAE; .embed(seq, signal) -> [B, d]
    temps: tuple[float, ...],
    dist_norm: bool = False,
    grad_form: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Sphere drift whose LOCALITY is the MAE phi (signal-aware), transport on the
    sphere. x = sqrt(softmax(logits)); positives = one-hot reals; the kernel WEIGHT
    distance is ||phi(x_i,sig_i) - phi(y_j,sig_j)|| in MAE-embedding space (so which
    real weighs on each generated depends on BOTH their signals -> breaks the batch
    permutation-invariance that blinds plain drifting to conditioning). Direction =
    sphere log-map, exactly as drift_loss_sphere. loss = geodesic²(x, sg(exp_x(V))).

    Diagnostic kdiag_ratio = mean diag(phi-dist gen_i,real_i) / mean off-diag: <1
    means generated sits nearest its OWN (same-signal) real in phi space."""
    probs = F.softmax(logits, dim=-1)
    x = probs_to_sphere(probs)                          # [B, L, 4], grad
    with torch.no_grad():
        y_pos = probs_to_sphere(y_pos_onehot)
        x_d = x.detach()
        phi_x = mae.embed(probs.detach(), signal)       # [B, d] generated locality
        phi_pos = mae.embed(y_pos_onehot, signal)       # [B, d] real locality
        kdist_pos = torch.cdist(phi_x, phi_pos)         # [B, B]
        kdist_neg = torch.cdist(phi_x, phi_x)           # [B, B] (neg == gen)
        V = compute_V_sphere_multi(x_d, y_pos, x_d, temps, dist_norm=dist_norm,
                                   grad_form=grad_form,
                                   kdist_pos=kdist_pos, kdist_neg=kdist_neg)
        x_tgt = _sphere_exp(x_d, V).detach()
        B = kdist_pos.shape[0]
        off = kdist_pos.sum() - kdist_pos.diagonal().sum()
        kdiag_ratio = float(kdist_pos.diagonal().mean()
                            / (off / max(B * (B - 1), 1)).clamp_min(1e-9))
    geo = _geodesic(x, x_tgt)                            # [B, L]
    loss = geo.pow(2).mean()
    info = {
        "loss": float(loss),
        "V_norm": float(V.reshape(V.shape[0], -1).norm(dim=-1).mean()),
        "x_norm": float(x.detach().reshape(x.shape[0], -1).norm(dim=-1).mean()),
        "kdiag_ratio": kdiag_ratio,
    }
    return loss, info


def drift_loss_sphere_plus_mae(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    signal: torch.Tensor,        # [B, L, 1] conditioning signal (paired w/ batch)
    mae,                         # frozen PromoterMAE
    temps: tuple[float, ...],
    lam: float,                  # phi-drift weight
    dist_norm: bool = False,     # for the SPHERE half
    grad_form: bool = False,     # SPHERE half: True (our best). phi half is ALWAYS
                                 # non-grad (displacement) -- unit-dir grad is unstable
                                 # in unbounded R^embed.
) -> tuple[torch.Tensor, dict]:
    """COMBINED (Form 1): the SPHERE drift supplies sequence QUALITY (geodesic loss,
    grad_form on the bounded manifold = our best generator), the EUCLIDEAN MAE-phi
    drift supplies CONDITIONING (displacement drift in R^embed, grad through the frozen
    MAE). loss = loss_sphere + lam * loss_phi. The two halves backprop into shared
    backbone logits; each delivers what the other lacks (sphere: kmer~0.85 but ratio~1;
    phi: kmer~0.21 but ratio<1, sigΔ 0.44). Tune lam to balance (watch loss_sphere vs
    loss_phi + phi_latR)."""
    loss_s, info_s = drift_loss_sphere(logits, y_pos_onehot, temps,
                                       dist_norm=dist_norm, grad_form=grad_form)
    loss_p, info_p = drift_loss_mae_euclid(logits, y_pos_onehot, signal, mae, temps,
                                           dist_norm=True, grad_form=False)
    loss = loss_s + lam * loss_p
    info = {
        "loss": float(loss),
        "loss_sphere": float(loss_s),
        "loss_phi": float(loss_p),
        "V_norm": info_s["V_norm"],
        "x_norm": info_s["x_norm"],
        "phi_V_norm": info_p["V_norm"],
        "phi_latR": info_p["phi_latR"],
    }
    return loss, info


def drift_loss_sphere_mae_jac(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    signal: torch.Tensor,        # [B, L, 1] conditioning signal (paired w/ batch)
    mae,                         # frozen PromoterMAE
    temps: tuple[float, ...],
    lam: float,                  # dimensionless: 1 = phi move == sphere move in magnitude
    dist_norm: bool = False,     # SPHERE half
    grad_form: bool = False,     # SPHERE half
) -> tuple[torch.Tensor, dict]:
    """COMBINED Form 2 (Jacobian): fold the MAE-phi drift into the ONE sphere target via
    J^T, per-sample NORM-MATCHED to ||V_sphere||, so the phi contribution is a controlled
    fraction `lam` of the sphere step EVERY step -> balance is STABLE across training
    (unlike Form 1's additive loss, whose two magnitudes drift apart). The phi drift's
    MAGNITUDE is discarded by the norm-match; only its DIRECTION (mapped to a sphere
    tangent by the MAE Jacobian-transpose) steers. loss = geodesic²(x, sg(exp_x(V_total))),
    V_total = V_sphere + lam * (J^T V_phi renormalized to ||V_sphere||). Main backward is
    cheap (through probs_to_sphere only; x_tgt detached); the MAE is used only to build the
    target. Diagnostics: cos(V_sphere, V_phi_tangent), nr (pre-scale norm ratio), phi_latR."""
    probs = F.softmax(logits, dim=-1)
    x = probs_to_sphere(probs)                          # [B, L, 4] carries grad
    x_d = x.detach()
    B = x_d.shape[0]
    with torch.no_grad():
        y_pos = probs_to_sphere(y_pos_onehot)
        V_sphere = compute_V_sphere_multi(x_d, y_pos, x_d, temps,
                                          dist_norm=dist_norm, grad_form=grad_form)
        _, phi_gen = mae.encode(x_d.pow(2), signal)     # x.pow(2) = probs (simplex)
        _, phi_real = mae.encode(y_pos_onehot, signal)
        V_phi = compute_V_multi(phi_gen, phi_real, phi_gen, temps,
                                dist_norm=True, grad_form=False)   # [B, d] displacement
        perm = torch.randperm(B, device=x_d.device)
        paired = (phi_gen - phi_real).pow(2).sum(-1).mean()
        permd = (phi_gen - phi_real[perm]).pow(2).sum(-1).mean()
        latR = float(paired / permd.clamp_min(1e-9))
    # J^T V_phi : map the phi-space direction to a sphere-tangent move
    with torch.enable_grad():
        xg = x_d.detach().requires_grad_(True)
        _, phi_xg = mae.encode(xg.pow(2), signal)
        g = torch.autograd.grad((phi_xg * V_phi).sum(), xg)[0]
    V_phi_t = _project_tangent(x_d, g)
    ns = V_sphere.reshape(B, -1).norm(dim=-1)
    nphi = V_phi_t.reshape(B, -1).norm(dim=-1).clamp_min(1e-12)
    V_phi_scaled = V_phi_t * (ns / nphi).view(-1, 1, 1)            # norm-match per-sample
    cos = float(F.cosine_similarity(V_sphere.reshape(B, -1),
                                    V_phi_t.reshape(B, -1), dim=-1).mean())
    nr = float((nphi / ns.clamp_min(1e-12)).mean())
    with torch.no_grad():
        V_total = V_sphere + lam * V_phi_scaled
        x_tgt = _sphere_exp(x_d, V_total).detach()
    geo = _geodesic(x, x_tgt)
    loss = geo.pow(2).mean()
    info = {
        "loss": float(loss),
        "V_norm": float(V_total.reshape(B, -1).norm(dim=-1).mean()),
        "x_norm": float(x_d.reshape(B, -1).norm(dim=-1).mean()),
        "phi_cos": cos,
        "phi_nr": nr,
        "phi_latR": latR,
    }
    return loss, info


# ===========================================================================
# Spectral Matern KERNEL SCORE drift on (S^3_+)^L. The drift is the Nadaraya-
# Watson kernel score  V = sum_j gradK / sum_j K  (= grad_x log sum_j k(x,y_j)),
# taken for positives minus negatives:  V = score_real(x) - score_gen(x). The
# kernel is the truncated spherical-harmonic Matern (Chebyshev-U / Gegenbauer
# C^1 form) per position; the product-manifold kernel k = prod_l k_l. No 1/d
# singularity: the per-position direction (dk/k)(y - z x) vanishes smoothly as
# points coincide (sqrt(1-z^2) -> 0). Multiple kappa lengthscales sum (multi-T
# analog). See compute_V_sphere_spectral_multi for the stable log-sum-exp form.
# ===========================================================================

def compute_V_sphere_spectral_multi(
    x: torch.Tensor,        # [N, L, 4] sphere points (generated)
    y_pos: torch.Tensor,    # [N_pos, L, 4] sphere points (positives)
    y_neg: torch.Tensor,    # [N_neg, L, 4] sphere points (negatives == x)
    weights_list: list[torch.Tensor],   # one Chebyshev-U weight vector per kappa
    eps: float = 1e-8,
) -> torch.Tensor:
    """Nadaraya-Watson kernel score drift:  V = sum_j gradK / sum_j K, for the
    positives minus the negatives, i.e. V = score_pos(x) - score_neg(x) where
    score(x) = grad_x log sum_j k(x, y_j). On the product manifold (S^3_+)^L the
    kernel is k = prod_l k_l, which underflows, so we use the equivalent stable
    form: the j-weights are softmax_j(sum_l log k_l) (= k_j / sum_j k_j), and
    grad_x k_j / k_j per position l is (dk_l/k_l)(y_l - (x_l.y_l) x_l). Returns
    tangent V at x. Multiple kappa lengthscales sum (analog of multi-T)."""
    N = x.shape[0]
    x = _sphere_normalize(x, eps)
    y_pos = _sphere_normalize(y_pos, eps)
    y_neg = _sphere_normalize(y_neg, eps)

    # per-position cosines z = x_l . y_l  -> [N, M, L]
    z_pos = torch.einsum("nlc,mlc->nml", x, y_pos).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    z_neg = torch.einsum("nlc,mlc->nml", x, y_neg).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    tan_pos = y_pos.unsqueeze(0) - z_pos.unsqueeze(-1) * x.unsqueeze(1)  # [N,M,L,4]
    tan_neg = y_neg.unsqueeze(0) - z_neg.unsqueeze(-1) * x.unsqueeze(1)

    idx = torch.arange(N, device=x.device)
    V = torch.zeros_like(x)
    for weights in weights_list:
        kfn = _cheb_lut(weights)                             # tabulated, no Python recurrence
        k_pos, dk_pos = kfn(z_pos)                           # [N, N_pos, L]
        k_neg, dk_neg = kfn(z_neg)

        # Nadaraya-Watson weights w_j = k_j / sum_j k_j  (= softmax of log k_j)
        logit_pos = k_pos.clamp_min(eps).log().sum(dim=-1)   # [N, N_pos]
        logit_neg = k_neg.clamp_min(eps).log().sum(dim=-1)   # [N, N_neg]
        logit_neg = logit_neg - torch.eye(N, device=x.device, dtype=x.dtype) * 1e9
        w_pos = logit_pos.softmax(dim=1)                     # [N, N_pos]
        w_neg = logit_neg.softmax(dim=1)

        # per-pair, per-position direction = grad_x k_j / k_j = (dk/k)(y - z x)
        dir_pos = (dk_pos / k_pos.clamp_min(eps)).unsqueeze(-1) * tan_pos
        dir_neg = (dk_neg / k_neg.clamp_min(eps)).unsqueeze(-1) * tan_neg
        dir_neg[idx, idx] = 0.0

        V_pos = (w_pos.unsqueeze(-1).unsqueeze(-1) * dir_pos).sum(dim=1)
        V_neg = (w_neg.unsqueeze(-1).unsqueeze(-1) * dir_neg).sum(dim=1)
        V = V + (V_pos - V_neg)
    return _project_tangent(x, V)


def drift_loss_sphere_spectral(
    logits: torch.Tensor,        # [B, L, 4] backbone output (carries grad)
    y_pos_onehot: torch.Tensor,  # [B, L, 4] one-hot real targets
    weights_list: list[torch.Tensor],
    eta: float = 1.0,
    max_step: float = 0.9 * 3.141592653589793,
) -> tuple[torch.Tensor, dict]:
    """Spectral Matern spherical drift step. x = sqrt(softmax(logits)) on
    (S^3_+)^L; positives are the one-hot corners; loss = geodesic²(x, sg(exp_x(V))).
    eta scales the step; per-position tangent norm is clamped < max_step to keep
    the exp-map from wrapping (the spectral score has no a-priori bound)."""
    probs = F.softmax(logits, dim=-1)
    x = probs_to_sphere(probs)                          # [B, L, 4], grad
    with torch.no_grad():
        y_pos = probs_to_sphere(y_pos_onehot)
        x_d = x.detach()
        V = compute_V_sphere_spectral_multi(x_d, y_pos, x_d, weights_list)
        V = eta * V
        vn = V.norm(dim=-1, keepdim=True)               # [B, L, 1]
        scale = (max_step / vn.clamp_min(1e-8)).clamp(max=1.0)
        V = V * scale
        x_tgt = _sphere_exp(x_d, V).detach()
    geo = _geodesic(x, x_tgt)                            # [B, L]
    loss = geo.pow(2).mean()
    info = {
        "loss": float(loss),
        "V_norm": float(V.reshape(V.shape[0], -1).norm(dim=-1).mean()),
        "x_norm": float(x.detach().reshape(x.shape[0], -1).norm(dim=-1).mean()),
    }
    return loss, info
