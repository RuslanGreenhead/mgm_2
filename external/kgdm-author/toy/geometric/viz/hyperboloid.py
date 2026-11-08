"""Minimal 3D plotting helpers for samples on H^2.

The hyperboloid is plotted with axes (x_1, x_2, x_0) -> (X, Y, Z), i.e. the
spatial components horizontally and the time-like x_0 vertically.
"""
from __future__ import annotations

import numpy as np


def set_clean_3d_hyperboloid(
    ax,
    view=(20, 30),
    title: str = "",
    xlim=(-1.4, 1.4),
    ylim=(-1.4, 1.4),
    zlim=(0.9, 2.2),
):
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_linewidth(0)
        axis._axinfo["grid"]["linewidth"] = 0
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    if title:
        ax.set_title(title, pad=6, color="black")


def draw_hyperboloid_surface(ax, r_max: float = 1.4):
    r = np.linspace(0, r_max, 30)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, Theta = np.meshgrid(r, theta)
    x0 = np.cosh(R)
    x1 = np.sinh(R) * np.cos(Theta)
    x2 = np.sinh(R) * np.sin(Theta)
    ax.plot_surface(
        x1, x2, x0,
        rstride=2, cstride=2,
        linewidth=0.35,
        edgecolor=(0.2, 0.2, 0.2, 0.18),
        color=(0.85, 0.85, 0.85, 0.04),
        antialiased=True, shade=False,
    )


def to_plot_xyz(ambient: np.ndarray) -> np.ndarray:
    """Reorder hyperboloid ambient (x0, x1, x2) -> plot (x1, x2, x0)."""
    return np.stack([ambient[:, 1], ambient[:, 2], ambient[:, 0]], axis=1)
