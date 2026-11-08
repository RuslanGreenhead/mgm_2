"""Compact 2x4 swissroll comparison for the main-body of the paper.

Rows:    swissroll_sphere, swissroll_hyperboloid
Columns: Ambient-Displacement, Ambient-Gradient, Manifold-Displacement, Manifold-Gradient

Display: final-step samples on the appropriate manifold, picking the seed
with the lowest sw_min for visual fidelity. Per-cell SW (mean ± std across
3 seeds) sits above each axes.

Output: results/compare_grid_main.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from viz.hyperboloid import draw_hyperboloid_surface, set_clean_3d_hyperboloid, to_plot_xyz
from viz.sphere import draw_sphere_surface, set_clean_3d_sphere


DATASETS = ["swissroll_sphere", "swissroll_hyperboloid"]
DATASET_LABELS = {
    "swissroll_sphere":      "Swiss roll\non $S^2$",
    "swissroll_hyperboloid": "Swiss roll\non $H^2$",
}
MODES = [
    ("euclidean", "base"),
    ("euclidean", "gradient"),
    ("manifold",  "base"),
    ("manifold",  "gradient"),
]
# Display labels only -- the tuple KEYS stay the data tokens written by main.py
# (results/<dataset>/<geom>-<method>/...); these strings are what appears on the
# figures: euclidean->Ambient, manifold->Manifold, base->Displacement,
# gradient->Gradient.
MODE_LABELS = {
    ("euclidean", "base"):     "Ambient · Displacement",
    ("euclidean", "gradient"): "Ambient · Gradient",
    ("manifold",  "base"):     "Manifold · Displacement",
    ("manifold",  "gradient"): "Manifold · Gradient",
}
METHOD_COLORS = {"base": "#2ca02c", "gradient": "#1f77b4"}
ROOT = Path("results")


def _load(ds: str, geom: str, meth: str):
    parent = ROOT / ds / f"{geom}-{meth}"
    if not parent.exists():
        return None
    seed_dirs = sorted(parent.glob("seed*"))
    if not seed_dirs:
        seed_dirs = [parent]

    sws_min = []
    display_dir = None
    display_step = None
    display_sw = float("inf")
    manifold_name = None

    for sd in seed_dirs:
        mp = sd / "metrics.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if not meta["log"]:
            continue
        if manifold_name is None:
            manifold_name = meta["config"]["manifold"]
        sw_min = min(r["sliced_w2"] for r in meta["log"])
        sws_min.append(sw_min)
        if sw_min < display_sw:
            display_sw = sw_min
            display_dir = sd
            display_step = sorted(meta["captured_steps"])[-1]

    if display_dir is None:
        return None

    samples = np.load(display_dir / f"samples_step{display_step:05d}.npy")
    target = np.load(display_dir / "target_pool.npy")
    return {
        "samples": samples,
        "target": target,
        "manifold": manifold_name,
        "n_seeds": len(sws_min),
        "sw_mean": float(np.mean(sws_min)),
        "sw_std":  float(np.std(sws_min, ddof=0)) if len(sws_min) > 1 else 0.0,
    }


def _plot_cell(ax, samples_ambient, manifold_name, color):
    if manifold_name == "sphere":
        set_clean_3d_sphere(ax, view=(50, 32))
        draw_sphere_surface(ax)
        pts = samples_ambient[np.argsort(samples_ambient[:, 2])]
    else:
        set_clean_3d_hyperboloid(ax, view=(45, 32))
        draw_hyperboloid_surface(ax)
        pts = to_plot_xyz(samples_ambient)
        pts = pts[np.argsort(pts[:, 2])]

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=3.5, alpha=0.75, color=color,
        edgecolors="white", linewidths=0.2, zorder=3,
    )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.dpi": 220,
    })

    n_rows = len(DATASETS)
    n_cols = len(MODES) + 1
    target_col = len(MODES)
    target_color = "#ff7f0e"

    fig = plt.figure(figsize=(2.7 * n_cols, 2.70 * n_rows))
    gs_left, gs_right = 0.005, 0.995
    gs_top, gs_bottom = 0.86, 0.02
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=gs_left, right=gs_right, top=gs_top, bottom=gs_bottom,
        wspace=-0.20, hspace=0.05,
        height_ratios=[1.16, 1.0],
    )
    row0_axes = [None] * n_cols

    for r, ds in enumerate(DATASETS):
        target_pts = None
        target_manifold = None
        for c, (geom, meth) in enumerate(MODES):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            if r == 0:
                row0_axes[c] = ax
            data = _load(ds, geom, meth)
            if data is None:
                ax.text2D(0.5, 0.5, "missing", color="#d62728",
                          ha="center", va="center", transform=ax.transAxes)
                continue

            color = METHOD_COLORS[meth]
            _plot_cell(ax, data["samples"], data["manifold"], color)

            if target_pts is None:
                target_pts = data["target"]
                target_manifold = data["manifold"]

            sw_m, sw_s = data["sw_mean"], data["sw_std"]
            n = data["n_seeds"]
            sub = (f"SW $={sw_m:.3f}\\pm{sw_s:.3f}$"
                   if n > 1 else f"SW $={sw_m:.3f}$")
            sw_y = 0.92 if r == 0 else 1.0
            ax.text2D(
                0.5, sw_y, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5, color="#333333",
            )

        ax_t = fig.add_subplot(gs[r, target_col], projection="3d")
        if r == 0:
            row0_axes[target_col] = ax_t
        if target_pts is not None:
            _plot_cell(ax_t, target_pts, target_manifold, target_color)

    def _x_center(ax):
        pos = ax.get_position()
        return 0.5 * (pos.x0 + pos.x1)

    for c, mode in enumerate(MODES):
        fig.text(
            _x_center(row0_axes[c]), 0.88,
            MODE_LABELS[mode],
            rotation=0, va="bottom", ha="center",
            fontsize=10.5, fontweight="bold",
            color=METHOD_COLORS[mode[1]],
        )

    fig.text(
        _x_center(row0_axes[target_col]), 0.88,
        "Target",
        rotation=0, va="bottom", ha="center",
        fontsize=10.5, fontweight="bold",
        color=target_color,
    )

    out = ROOT / "compare_grid_main.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    _trim_whitespace(out, sides=("bottom",), pad=8)
    print(f"wrote {out}")


def _trim_whitespace(path: Path, sides=("bottom",), pad: int = 0, threshold: int = 250):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    nonwhite = (arr < threshold).any(axis=2)
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    h, w = arr.shape[:2]
    top, bottom = (rows[0], rows[-1] + 1)
    left, right = (cols[0], cols[-1] + 1)
    if "top" in sides:
        top = max(0, top - pad)
    else:
        top = 0
    if "bottom" in sides:
        bottom = min(h, bottom + pad)
    else:
        bottom = h
    if "left" in sides:
        left = max(0, left - pad)
    else:
        left = 0
    if "right" in sides:
        right = min(w, right + pad)
    else:
        right = w
    img.crop((left, top, right, bottom)).save(path)


if __name__ == "__main__":
    main()
