"""Render the 4x4 comparison grid + per-cell metrics.

Rows:    datasets (checkerboard_sphere, checkerboard_hyperboloid, swissroll_sphere, swissroll_hyperboloid)
Columns: modes    (Ambient-Displacement, Ambient-Gradient, Manifold-Displacement, Manifold-Gradient)

Display: final-step samples on the appropriate manifold.
Metrics: sw = min sliced-W2 during training; tile = final-step tile accuracy.

Usage: python make_compare_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.hyperboloid import draw_hyperboloid_surface, set_clean_3d_hyperboloid, to_plot_xyz
from viz.sphere import draw_sphere_surface, set_clean_3d_sphere


DATASETS = [
    "checkerboard_sphere",
    "checkerboard_hyperboloid",
    "swissroll_sphere",
    "swissroll_hyperboloid",
]
DATASET_LABELS = {
    "checkerboard_sphere":      "Checkerboard\non $S^2$",
    "checkerboard_hyperboloid": "Checkerboard\non $H^2$",
    "swissroll_sphere":         "Swiss roll\non $S^2$",
    "swissroll_hyperboloid":    "Swiss roll\non $H^2$",
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
    """Load all seeds for one (dataset, geometry, method) cell.

    Returns sw_min ± std and tile_final ± std across seeds, plus the
    final-step snapshot of the seed with the lowest sw_min for display.
    """
    parent = ROOT / ds / f"{geom}-{meth}"
    if not parent.exists():
        return None
    seed_dirs = sorted(parent.glob("seed*"))
    if not seed_dirs:
        seed_dirs = [parent]

    sws_min, tiles_final = [], []
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

        log = meta["log"]
        sw_min = min(r["sliced_w2"] for r in log)
        sws_min.append(sw_min)
        final = log[-1]
        if "tile_acc" in final:
            tiles_final.append(final["tile_acc"])

        if sw_min < display_sw:
            display_sw = sw_min
            display_dir = sd
            captured = sorted(meta["captured_steps"])
            display_step = captured[-1]

    if display_dir is None:
        return None

    samples = np.load(display_dir / f"samples_step{display_step:05d}.npy")
    target = np.load(display_dir / "target_pool.npy")

    return {
        "samples": samples,
        "target": target,
        "manifold": manifold_name,
        "n_seeds": len(sws_min),
        "sw_mean":  float(np.mean(sws_min)) if sws_min else float("nan"),
        "sw_std":   float(np.std(sws_min, ddof=0)) if len(sws_min) > 1 else 0.0,
        "tile_mean": float(np.mean(tiles_final)) if tiles_final else None,
        "tile_std":  float(np.std(tiles_final, ddof=0)) if len(tiles_final) > 1 else 0.0,
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
        s=4.5, alpha=0.7, color=color,
        edgecolors="white", linewidths=0.25, zorder=3,
    )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "savefig.dpi": 200,
    })

    n_rows = len(DATASETS)
    n_cols = len(MODES)

    fig = plt.figure(figsize=(3.4 * n_cols, 3.6 * n_rows))
    gs_left, gs_right = 0.07, 0.99
    gs_top, gs_bottom = 0.92, 0.04
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=gs_left, right=gs_right, top=gs_top, bottom=gs_bottom,
        wspace=0.05, hspace=0.30,
    )
    row_height = (gs_top - gs_bottom) / n_rows
    col_width  = (gs_right - gs_left) / n_cols

    summary = []
    missing = []

    for r, ds in enumerate(DATASETS):
        for c, (geom, meth) in enumerate(MODES):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            data = _load(ds, geom, meth)
            if data is None:
                ax.text2D(0.5, 0.5, "missing", color="#d62728",
                          ha="center", va="center", transform=ax.transAxes)
                missing.append((ds, geom, meth))
                continue

            color = METHOD_COLORS[meth]
            _plot_cell(ax, data["samples"], data["manifold"], color)

            sw_m, sw_s = data["sw_mean"], data["sw_std"]
            tile_m, tile_s = data["tile_mean"], data["tile_std"]
            n = data["n_seeds"]
            sub_sw = f"SW $={sw_m:.3f}\\pm{sw_s:.3f}$" if n > 1 else f"SW $={sw_m:.3f}$"
            sub = sub_sw
            if tile_m is not None:
                sub_tile = (f"   acc $={tile_m:.2f}\\pm{tile_s:.2f}$"
                            if n > 1 else f"   acc $={tile_m:.2f}$")
                sub += sub_tile

            # Per-cell stats sit *above* the axes, so they never overlap the surface.
            ax.text2D(
                0.5, 1.04, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, color="#333333",
            )
            summary.append((ds, geom, meth, n, sw_m, sw_s, tile_m, tile_s))

        # Row label on the left, centered on its row in gridspec coordinates.
        y_center = gs_top - (r + 0.5) * row_height
        fig.text(
            0.012, y_center,
            DATASET_LABELS[ds],
            rotation=0, va="center", ha="left",
            fontsize=12, fontweight="bold", color="#222222",
        )

    # Column headers, centered above their column in gridspec coordinates.
    for c, mode in enumerate(MODES):
        x_center = gs_left + (c + 0.5) * col_width
        fig.text(
            x_center, 0.965,
            MODE_LABELS[mode],
            rotation=0, va="bottom", ha="center",
            fontsize=12.5, fontweight="bold",
            color=METHOD_COLORS[mode[1]],
        )

    fig.suptitle(
        "Drift quality across geometry · method  (3 seeds; SW = min during training)",
        fontsize=13.5, y=0.995,
    )

    out = ROOT / "compare_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    if summary:
        print(f"\n{'dataset':<28} {'mode':<22} {'n':>2} {'sw mean±std':>16} {'tile mean±std':>16}")
        print("-" * 90)
        for ds, geom, meth, n, sw_m, sw_s, tile_m, tile_s in summary:
            sw_cell = f"{sw_m:.4f}±{sw_s:.4f}"
            tile_cell = f"{tile_m:.3f}±{tile_s:.3f}" if tile_m is not None else "       —       "
            print(f"{ds:<28} {geom + '-' + meth:<22} {n:>2} {sw_cell:>16} {tile_cell:>16}")

    if missing:
        print(f"\n[warn] {len(missing)} configs missing:")
        for ds, geom, meth in missing:
            print(f"   {ds} / {geom}-{meth}")


if __name__ == "__main__":
    main()
