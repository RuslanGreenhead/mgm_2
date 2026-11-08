"""Render Fig. 5.2 (C2ST vs Matérn ν) from run_sweep.py's metrics.json.

Edit METHODS_KEEP / NU_DROP below to adjust which lines/points are shown.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path("results/swissroll/paper-fig/metrics.json")
OUT_DIR = Path("results/swissroll/paper-fig")

METHODS_KEEP = ["base", "grad"]
NU_DROP = {2.5}

METHOD_COLORS = {"base": "#2ca02c", "grad": "#1f77b4"}
# Display labels only -- keys stay the data tokens (config update_type / JSON
# "name"); these strings are what appears in the figure legend.
METHOD_LABELS = {"base": "Displacement", "grad": "Gradient"}


def _mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _nu_value(nu) -> float:
    if isinstance(nu, str):
        return math.inf if nu == "inf" else float(nu)
    return float(nu)


def _filtered_per_method(blocks):
    out = {}
    for b in blocks:
        if b["name"] not in METHODS_KEEP:
            continue
        rows = []
        for entry in b["data"]:
            nu = _nu_value(entry["nu"])
            # ν=100 stays as the finite stand-in for the Gaussian limit; only
            # an explicit math.inf is dropped from the log-x plot.
            if not math.isfinite(nu) or nu in NU_DROP:
                continue
            mc, sc = _mean_std(entry["values_c2st"])
            rows.append((nu, mc, sc))
        rows.sort(key=lambda r: r[0])
        out[b["name"]] = rows
    return out


def _plot_panel(ax, per_method, ylabel):
    for name in METHODS_KEEP:
        if name not in per_method:
            continue
        rows = per_method[name]
        xs = np.array([r[0] for r in rows])
        ms = np.array([r[1] for r in rows])
        ss = np.array([r[2] for r in rows])
        color = METHOD_COLORS.get(name)
        ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, ms, marker="o", color=color,
                label=METHOD_LABELS.get(name, name), lw=2.0,
                ms=6, mec="white", mew=1.2, zorder=3)
    ax.axhline(0.5, ls="--", c="gray", lw=1, alpha=0.7, label="chance", zorder=1)
    xs_any = [r[0] for r in next(iter(per_method.values()))]
    ax.set_xscale("log")
    ax.set_xlabel(
        "Matérn smoothness $\\nu$ (log; $\\nu=100 \\approx$ Gaussian)",
        fontsize=12,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(xs_any)
    ax.set_xticklabels([f"{x:g}" for x in xs_any])
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.25, which="both", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=11, loc="best", frameon=False)


def main():
    data = json.load(open(SRC))
    per_method = _filtered_per_method(data["results"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_panel(ax, per_method, "C2ST accuracy (final, mean ± std)")
    fig.tight_layout()
    out = OUT_DIR / "paper_c2st_vs_nu.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    print(f"\nshown ν: {[r[0] for r in next(iter(per_method.values()))]}")
    print(f"shown methods: {list(per_method.keys())}")


if __name__ == "__main__":
    main()
