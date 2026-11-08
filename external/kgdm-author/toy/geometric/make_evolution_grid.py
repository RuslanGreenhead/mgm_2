"""Per-dataset evolution figure: 4 modes (rows) × captured steps (columns).

For each (geometry, method) cell we pick the seed with the lowest sw_min and
display its samples at every captured training step. One figure per dataset.

Output: results/<dataset>/evolution_grid.png

Usage: python make_evolution_grid.py
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
    "checkerboard_sphere":      "Checkerboard on $S^2$",
    "checkerboard_hyperboloid": "Checkerboard on $H^2$",
    "swissroll_sphere":         "Swiss roll on $S^2$",
    "swissroll_hyperboloid":    "Swiss roll on $H^2$",
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


def _pick_best_seed(parent: Path):
    seed_dirs = sorted(parent.glob("seed*"))
    if not seed_dirs:
        seed_dirs = [parent]

    best_dir = None
    best_sw = float("inf")
    captured = None
    manifold_name = None

    for sd in seed_dirs:
        mp = sd / "metrics.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if not meta["log"]:
            continue
        sw_min = min(r["sliced_w2"] for r in meta["log"])
        if sw_min < best_sw:
            best_sw = sw_min
            best_dir = sd
            captured = sorted(meta["captured_steps"])
            manifold_name = meta["config"]["manifold"]

    return best_dir, captured, manifold_name


def _plot_cell(ax, samples_ambient, manifold_name, color, view):
    if manifold_name == "sphere":
        set_clean_3d_sphere(ax, view=view)
        draw_sphere_surface(ax)
        pts = samples_ambient[np.argsort(samples_ambient[:, 2])]
    else:
        set_clean_3d_hyperboloid(ax, view=view)
        draw_hyperboloid_surface(ax)
        pts = to_plot_xyz(samples_ambient)
        pts = pts[np.argsort(pts[:, 2])]

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=3.5, alpha=0.75, color=color,
        edgecolors="white", linewidths=0.2, zorder=3,
    )


def render_one(dataset: str):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
    })

    is_hyp = "hyperboloid" in dataset
    view = (45, 32) if is_hyp else (50, 32)

    rows_data = []
    step_sets = []
    for geom, meth in MODES:
        parent = ROOT / dataset / f"{geom}-{meth}"
        if not parent.exists():
            rows_data.append(None)
            continue
        best_dir, captured, manifold = _pick_best_seed(parent)
        if best_dir is None:
            rows_data.append(None)
            continue
        rows_data.append((best_dir, captured, manifold, geom, meth))
        step_sets.append(set(captured))

    if not step_sets:
        common_steps = None
    else:
        common_steps = sorted(set.intersection(*step_sets))

    if common_steps is None:
        print(f"[skip] {dataset}: no data")
        return

    n_rows = len(MODES)
    n_cols = len(common_steps)

    fig = plt.figure(figsize=(2.6 * n_cols, 2.8 * n_rows))
    gs_left, gs_right = 0.10, 0.99
    gs_top, gs_bottom = 0.91, 0.04
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=gs_left, right=gs_right, top=gs_top, bottom=gs_bottom,
        wspace=0.05, hspace=0.18,
    )
    row_height = (gs_top - gs_bottom) / n_rows
    col_width  = (gs_right - gs_left) / n_cols

    for r, row in enumerate(rows_data):
        for c, step in enumerate(common_steps):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            if row is None:
                ax.text2D(0.5, 0.5, "missing", color="#d62728",
                          ha="center", va="center", transform=ax.transAxes)
                continue
            best_dir, captured, manifold, geom, meth = row
            sample_path = best_dir / f"samples_step{step:05d}.npy"
            if not sample_path.exists():
                ax.text2D(0.5, 0.5, "—", color="#999999",
                          ha="center", va="center", transform=ax.transAxes)
                continue
            samples = np.load(sample_path)
            color = METHOD_COLORS[meth]
            _plot_cell(ax, samples, manifold, color, view)

        if row is not None:
            geom, meth = row[3], row[4]
            label = MODE_LABELS[(geom, meth)]
            color = METHOD_COLORS[meth]
        else:
            label = MODE_LABELS[MODES[r]]
            color = "#999999"
        y_center = gs_top - (r + 0.5) * row_height
        fig.text(
            0.012, y_center, label,
            rotation=0, va="center", ha="left",
            fontsize=12, fontweight="bold", color=color,
        )

    for c, step in enumerate(common_steps):
        x_center = gs_left + (c + 0.5) * col_width
        fig.text(
            x_center, 0.95,
            f"step {step}",
            rotation=0, va="bottom", ha="center",
            fontsize=12, fontweight="bold", color="#222222",
        )

    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: distribution evolution (best seed per cell)",
        fontsize=13.5, y=0.99,
    )

    out = ROOT / dataset / "evolution_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    for ds in DATASETS:
        render_one(ds)


if __name__ == "__main__":
    main()
