"""Minimal 3D plotting helpers for samples on S^2."""
from __future__ import annotations

import numpy as np


def set_clean_3d_sphere(ax, view=(20, 30), title: str = ""):
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_linewidth(0)
        axis._axinfo["grid"]["linewidth"] = 0
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    if title:
        ax.set_title(title, pad=6, color="black")


def draw_sphere_surface(ax):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x, y, z,
        rstride=2, cstride=2,
        linewidth=0.35,
        edgecolor=(0.2, 0.2, 0.2, 0.18),
        color=(0.85, 0.85, 0.85, 0.04),
        antialiased=True, shade=False,
    )
