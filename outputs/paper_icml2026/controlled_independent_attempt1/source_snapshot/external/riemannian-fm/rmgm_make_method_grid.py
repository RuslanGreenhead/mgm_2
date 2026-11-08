"""Render side-by-side target/R-MGM/RFM comparison panels.

The earlier quick plots overlay data and generated samples. This script makes a
paper-checking view where each method occupies its own cell:

    Ground truth | R-MGM | RFM full ODE | RFM 1-step

It uses the current local experiment outputs and resamples from checkpoints
instead of composing old PNGs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifm.manifolds.hyperbolic import PoincareBall
from manifm.manifolds.spd import SPD
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus
from manifm.rmgm import build_critic, build_generator
from rfm_baseline_train import BaseSampler, TrainConfig as RFMConfig, sample_rfm
from rmgm_train import TrainConfig as RMGMConfig
from rmgm_train import make_experiment, poincare_plot_radius


MANIFEST: Dict[str, Dict[str, str]] = {
    "sphere": {
        "rmgm": "../../outputs/rmgm_refine_sphere_local_gmm/sphere/seed0_20260731_232205/checkpoints/best.pt",
        "rfm": "../../outputs/rfm_baseline/sphere/seed0_20260731_230414/checkpoints/last.pt",
        "rfm_1nfe": "../../outputs/rfm_baseline_1nfe/sphere/seed0_20260731_230654/checkpoints/last.pt",
    },
    "spd": {
        "rmgm": "../../outputs/rmgm_refine_spd_best/spd/seed0_20260731_234324/checkpoints/best.pt",
        "rfm": "../../outputs/rfm_baseline/spd/seed0_20260731_230439/checkpoints/last.pt",
        "rfm_1nfe": "../../outputs/rfm_baseline_1nfe/spd/seed0_20260731_230715/checkpoints/last.pt",
    },
    "hyperbolic": {
        "rmgm": "../../outputs/rmgm_refine_hyperbolic_local_gmm/hyperbolic/seed0_20260731_232251/checkpoints/best.pt",
        "rfm": "../../outputs/rfm_baseline/hyperbolic/seed0_20260731_230521/checkpoints/last.pt",
        "rfm_1nfe": "../../outputs/rfm_baseline_1nfe/hyperbolic/seed0_20260731_230748/checkpoints/last.pt",
    },
    "torus2d": {
        "rmgm": "../../outputs/rmgm_refine_torus_gmm_kmeans/torus2d/seed0_20260731_231953/checkpoints/best.pt",
        "rfm": "../../outputs/rfm_baseline/torus2d/seed0_20260731_230550/checkpoints/last.pt",
        "rfm_1nfe": "../../outputs/rfm_baseline_1nfe/torus2d/seed0_20260731_230809/checkpoints/last.pt",
    },
}

COLUMNS = [
    ("data", "Ground truth"),
    ("rmgm", "R-MGM"),
    ("rfm", "RFM full ODE"),
    ("rfm_1nfe", "RFM 1-step"),
]

COLORS = {
    "data": "black",
    "rmgm": "#d9782f",
    "rfm": "#3569b8",
    "rfm_1nfe": "#8a63aa",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../../outputs/rmgm_summary/method_comparison_grid.png")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    rows = list(MANIFEST)
    fig = plt.figure(figsize=(14.0, 13.2))
    gs = fig.add_gridspec(len(rows), len(COLUMNS), wspace=0.04, hspace=0.10)

    for row_idx, experiment in enumerate(rows):
        manifold, dim, data_sampler = make_experiment(experiment)
        method_samples = sample_all_methods(experiment, manifold, dim, data_sampler, args.samples, device)

        for col_idx, (method, title) in enumerate(COLUMNS):
            projection = "3d" if isinstance(manifold, (Sphere, SPD)) else None
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=projection)
            plot_single(ax, experiment, manifold, method_samples[method], COLORS[method])
            if row_idx == 0:
                ax.set_title(title, fontsize=16, pad=10)
            if col_idx == 0:
                text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
                text_fn(
                    -0.16,
                    0.50,
                    row_label(experiment),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=15,
                    fontweight="bold",
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


@torch.no_grad()
def sample_all_methods(
    experiment: str,
    manifold: Any,
    dim: int,
    data_sampler: Any,
    n: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    dtype = torch.float32
    samples: Dict[str, torch.Tensor] = {
        "data": data_sampler.sample(n, device, dtype).detach().cpu()
    }
    samples["rmgm"] = sample_rmgm_checkpoint(
        MANIFEST[experiment]["rmgm"], manifold, dim, n, device
    ).detach().cpu()
    samples["rfm"] = sample_rfm_checkpoint(
        MANIFEST[experiment]["rfm"], manifold, dim, n, device
    ).detach().cpu()
    samples["rfm_1nfe"] = sample_rfm_checkpoint(
        MANIFEST[experiment]["rfm_1nfe"], manifold, dim, n, device
    ).detach().cpu()
    return samples


@torch.no_grad()
def sample_rmgm_checkpoint(path: str, manifold: Any, dim: int, n: int, device: torch.device) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu")
    cfg = RMGMConfig(**ckpt["config"])
    dtype = getattr(torch, cfg.dtype)
    generator_cfg = {
        "generator": {
            "latent_dim": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "activation": "silu",
            "generator_type": cfg.generator_type,
            "generator_components": cfg.generator_components,
            "generator_residual_scale": cfg.generator_residual_scale,
            "torus_map": cfg.torus_map,
            "torus_generator": cfg.torus_generator,
            "torus_components": cfg.torus_components,
            "torus_residual_scale": cfg.torus_residual_scale,
            "torus_mixture_init": cfg.torus_mixture_init,
        }
    }
    generator = build_generator(generator_cfg, manifold, dim).to(device=device, dtype=dtype)
    generator.load_state_dict(ckpt["generator"])
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    return generator(z)


@torch.no_grad()
def sample_rfm_checkpoint(path: str, manifold: Any, dim: int, n: int, device: torch.device) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu")
    cfg = RFMConfig(**ckpt["config"])
    dtype = getattr(torch, cfg.dtype)
    field_cfg = {
        "critic": {
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "activation": "silu",
            "fourier": 8,
            "metric_normalize": cfg.metric_normalize,
        }
    }
    field = build_critic(field_cfg, manifold, dim).to(device=device, dtype=dtype)
    field.load_state_dict(ckpt["field"])
    base_sampler = BaseSampler(manifold, dim, hyperbolic_scale=cfg.hyperbolic_base_scale)
    return sample_rfm(cfg, manifold, base_sampler, field, n, device, dtype)


def plot_single(ax: Any, experiment: str, manifold: Any, x: torch.Tensor, color: str) -> None:
    if isinstance(manifold, FlatTorus):
        plot_torus(ax, x, color)
    elif isinstance(manifold, PoincareBall):
        plot_poincare(ax, manifold, x, color)
    elif isinstance(manifold, SPD):
        plot_spd(ax, x, color)
    elif isinstance(manifold, Sphere):
        plot_sphere(ax, x, color)
    else:
        arr = x.numpy()
        ax.scatter(arr[:, 0], arr[:, 1], s=2.0, color=color, alpha=0.42, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
    ax.set_facecolor("white")


def plot_torus(ax: Any, x: torch.Tensor, color: str) -> None:
    arr = (((x.numpy() + math.pi) % (2 * math.pi)) - math.pi) / math.pi * 180.0
    ax.scatter(arr[:, 0], arr[:, 1], s=2.0, color=color, alpha=0.38, linewidths=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.2", lw=0.45, ls="--", alpha=0.45)
    ax.axvline(0, color="0.2", lw=0.45, ls="--", alpha=0.45)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.tick_params(labelsize=8, length=2)


def plot_poincare(ax: Any, manifold: Any, x: torch.Tensor, color: str) -> None:
    radius = poincare_plot_radius(manifold)
    arr = x.numpy()
    ax.scatter(arr[:, 0], arr[:, 1], s=2.0, color=color, alpha=0.38, linewidths=0)
    ax.add_patch(plt.Circle((0, 0), radius, color="black", fill=False, lw=0.9))
    pad = radius * 0.05
    ax.set_xlim(-(radius + pad), radius + pad)
    ax.set_ylim(-(radius + pad), radius + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_sphere(ax: Any, x: torch.Tensor, color: str) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 42j, 0 : np.pi : 22j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#edf5fa", alpha=0.18, linewidth=0)
    arr = x.numpy()
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=2.0, color=color, alpha=0.34, linewidths=0)
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_spd(ax: Any, x: torch.Tensor, color: str) -> None:
    rr = np.linspace(0, 3.4, 56)
    th = np.linspace(0, 2 * np.pi, 56)
    r_grid, t_grid = np.meshgrid(rr, th)
    cone_x = r_grid * np.cos(t_grid)
    cone_y = r_grid * np.sin(t_grid)
    cone_z = r_grid
    ax.plot_surface(cone_x, cone_y, cone_z, color="#edf5fa", alpha=0.18, linewidth=0)
    arr = x.numpy()
    c = arr[:, 1]
    u = 0.5 * (arr[:, 0] + arr[:, 2])
    v = 0.5 * (arr[:, 0] - arr[:, 2])
    ax.scatter(c, v, u, s=2.2, color=color, alpha=0.38, linewidths=0)
    ax.view_init(azim=88, elev=12)
    ax.set_axis_off()


def row_label(experiment: str) -> str:
    return {
        "sphere": "Sphere S2",
        "spd": "SPD(2)",
        "hyperbolic": "Poincare disk",
        "torus2d": "Flat torus",
    }.get(experiment, experiment)


if __name__ == "__main__":
    main()
