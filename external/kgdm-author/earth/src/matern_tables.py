"""Lookup tables for the spectral Matern kernels on S^2.

The spectral kernel is a 1-D function of the angle, so for each (nu, kappa)
we evaluate once, in float64:
    f(theta) = sum_l w_l P_l(cos theta)      (kernel; f(0)=1 normalized)
    g(theta) = sum_l w_l P_l'(cos theta)     (ambient-gradient scalar:
                                              grad_x k = g(x.y) * y)
on a sqrt-spaced theta grid; TabulatedMaternS2 interpolates at training time.

ensure_tables() is called automatically before any spectral run: it builds
whichever (nu, kappa) combos are missing from the table file and merges them
in. Prebuild every combo listed in configs/winners.yaml with:
    python -m src.matern_tables
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch

DELTA = 0.002                    # nu=0.5 mollification scale (rad)
LMAX = 6000
N_GRID = 4096


def _key(nu: float, kappa: float) -> str:
    return f"{float(nu)}_{float(kappa)}"


def weights(nu: float, kappa: float) -> torch.Tensor:
    """Normalized spectral weights (k(x,x)=1). The nu=0.5 kernel has a cusp
    at theta=0, so its weights are mollified by exp(-DELTA^2/2 l(l+1)), which
    keeps the kernel PD and makes the series geometrically convergent."""
    l = torch.arange(LMAX + 1, dtype=torch.float64)
    lam = l * (l + 1.0)
    coeff = (2.0 * l + 1.0) / (4.0 * math.pi)
    if math.isinf(nu):
        w = coeff * torch.exp(-0.5 * kappa ** 2 * lam)
    else:
        w = coeff * (kappa ** 2 + lam) ** (-(nu + 1.0))
    if nu == 0.5:
        w = w * torch.exp(-(DELTA ** 2 / 2) * lam)
    return w / w.sum()


def build_tables(combos):
    """f, g tables for the given (nu, kappa) combos, keyed by _key().

    One shared Legendre recursion over the theta grid plus midpoints, with a
    snapshot at LMAX//2; prints the truncation and interpolation error per
    combo (relative errors are large only where the kernel tail underflows)."""
    combos = sorted(set((float(nu), float(k)) for nu, k in combos))
    i = torch.arange(N_GRID, dtype=torch.float64)
    theta_grid = (i / (N_GRID - 1)) ** 2 * math.pi          # sqrt-spaced
    mid = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    theta = torch.cat([theta_grid, mid])

    W = {c: weights(*c) for c in combos}

    z = torch.cos(theta)
    P_prev, P_curr = torch.ones_like(z), z.clone()          # P_0, P_1
    D_prev, D_curr = torch.zeros_like(z), torch.ones_like(z)  # P_0', P_1'
    f = {c: W[c][0] * P_prev + W[c][1] * P_curr for c in combos}
    g = {c: W[c][1] * D_curr.clone() for c in combos}
    snap = None
    for l in range(1, LMAX):
        P_next = ((2 * l + 1) * z * P_curr - l * P_prev) / (l + 1)
        D_next = ((2 * l + 1) * (P_curr + z * D_curr) - l * D_prev) / (l + 1)
        P_prev, P_curr = P_curr, P_next
        D_prev, D_curr = D_curr, D_next
        for c in combos:
            f[c] = f[c] + W[c][l + 1] * P_curr
            g[c] = g[c] + W[c][l + 1] * D_curr
        if l + 1 == LMAX // 2:
            snap = ({c: f[c].clone() for c in combos},
                    {c: g[c].clone() for c in combos})

    def maxrel(a, b):
        return ((a - b).abs() / b.abs().clamp_min(1e-12)).max().item()

    f_tabs, g_tabs = {}, {}
    for c in combos:
        nu, kappa = c
        fe2, ge2 = maxrel(snap[0][c], f[c]), maxrel(snap[1][c], g[c])
        fg, fm = f[c][:N_GRID], f[c][N_GRID:]
        gg, gm = g[c][:N_GRID], g[c][N_GRID:]
        fi = maxrel(0.5 * (fg[:-1] + fg[1:]), fm)
        gi = maxrel(0.5 * (gg[:-1] + gg[1:]), gm)
        print(f"matern table nu={str(nu):>4} kappa={kappa:>8}: "
              f"conv(L{LMAX//2} vs L{LMAX}) f {fe2:.1e} g {ge2:.1e} | "
              f"interp@mid f {fi:.1e} g {gi:.1e} | f(0)={fg[0]:.4f}")
        f_tabs[_key(*c)] = fg
        g_tabs[_key(*c)] = gg
    return f_tabs, g_tabs


def ensure_tables(path: str, combos) -> None:
    """Build any (nu, kappa) combos missing from the table file at `path`
    and merge them in (atomic replace, safe under concurrent runs)."""
    p = Path(path)
    existing = None
    missing = [(float(nu), float(k)) for nu, k in combos]
    if p.exists():
        existing = torch.load(p, weights_only=True)
        missing = [c for c in missing if _key(*c) not in existing["f"]]
        if not missing:
            return
    f_tabs, g_tabs = build_tables(missing)
    if existing is not None:
        f_tabs = {**existing["f"], **f_tabs}
        g_tabs = {**existing["g"], **g_tabs}
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    torch.save({"n": N_GRID, "Lmax": LMAX, "delta_nu05": DELTA,
                "f": f_tabs, "g": g_tabs}, tmp)
    os.replace(tmp, p)
    print(f"matern tables -> {p}")


def main():
    """Prebuild the tables for every spectral formulation in winners.yaml."""
    import yaml
    with open("configs/winners.yaml") as f:
        winners = yaml.safe_load(f)
    combos = [(spec["matern_nu"], kappa)
              for spec in winners.values() if spec.get("drift_type") == "spectral"
              for ds in spec["per_dataset"].values()
              for kappa in ds["matern_kappas"]]
    ensure_tables("matern_tables.pt", combos)
    print(f"{len(set(combos))} combos present")


if __name__ == "__main__":
    main()
