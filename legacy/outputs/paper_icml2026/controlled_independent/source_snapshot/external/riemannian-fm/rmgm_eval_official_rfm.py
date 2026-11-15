"""Evaluate an official RFM Lightning checkpoint with R-MGM sample metrics.

This script is intentionally self-contained so it can import a clean checkout
of the original RFM package while living next to the R-MGM utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))

    from omegaconf import OmegaConf
    from manifm.datasets import get_loaders
    from manifm.eval_utils import get_job_directory
    from manifm.manifolds import FlatTorus, Mesh, PoincareBall, Sphere, SPD
    from manifm.model_pl import ManifoldFMLitModule
    from manifm.utils import lonlat_from_cartesian

    del Mesh
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = Path(args.checkpoint).resolve()
    job_dir = Path(get_job_directory(str(checkpoint)))
    cfg = OmegaConf.load(job_dir / ".hydra" / "config.yaml")
    _, _, test_loader = get_loaders(cfg)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = ManifoldFMLitModule(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    data, base = collect_test_tensors(test_loader, args.eval_samples)
    data = data.to(device=device, dtype=torch.float32)
    if base is not None:
        base = base.to(device=device, dtype=torch.float32)

    samples = sample_model(model, data.shape[0], device, args.sample_batch_size, base)
    samples = model.manifold.projx(samples)

    metric_n = min(args.metric_samples, data.shape[0], samples.shape[0])
    metrics = sample_metrics(
        cfg.data,
        model.manifold,
        samples[:metric_n],
        data[:metric_n],
        (FlatTorus, Sphere, SPD, PoincareBall),
    )
    metrics.update(
        {
            "data": str(cfg.data),
            "dim": int(model.dim),
            "checkpoint": str(checkpoint),
            "job_dir": str(job_dir),
            "global_step": int(ckpt.get("global_step", -1)),
            "eval_samples": int(data.shape[0]),
            "metric_samples": int(metric_n),
            "method": "official_rfm",
        }
    )

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "samples": samples.detach().cpu(),
            "data": data.detach().cpu(),
            "base": None if base is None else base.detach().cpu(),
            "metrics": metrics,
        },
        out_dir / "samples.pt",
    )
    save_plot(
        cfg.data,
        model.manifold,
        data.detach().cpu(),
        samples.detach().cpu(),
        out_dir / "official_rfm_samples.png",
        (FlatTorus, Sphere, SPD, PoincareBall),
        lonlat_from_cartesian,
    )
    print(json.dumps(metrics, indent=2))


def collect_test_tensors(loader: Any, n: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    data_parts = []
    base_parts = []
    has_base = False
    total = 0
    for batch in loader:
        if isinstance(batch, dict):
            x1 = batch["x1"]
            x0 = batch.get("x0")
            if x0 is not None:
                has_base = True
                base_parts.append(x0.detach().cpu())
        else:
            x1 = batch
        data_parts.append(x1.detach().cpu())
        total += int(x1.shape[0])
        if total >= n:
            break
    data = torch.cat(data_parts, dim=0)[:n]
    base = torch.cat(base_parts, dim=0)[:n] if has_base else None
    return data, base


@torch.no_grad()
def sample_model(
    model: torch.nn.Module,
    n: int,
    device: torch.device,
    batch_size: int,
    base: torch.Tensor | None,
) -> torch.Tensor:
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x0 = None if base is None else base[start:end].to(device)
        chunks.append(model.sample(end - start, device=device, x0=x0).detach().cpu())
    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def sample_metrics(
    name: str,
    manifold: Any,
    samples: torch.Tensor,
    data: torch.Tensor,
    classes: tuple[type, type, type, type],
) -> dict[str, float]:
    del name
    flat_torus_cls, sphere_cls, _, _ = classes
    embed_samples = embed_for_metrics(manifold, samples, flat_torus_cls)
    embed_data = embed_for_metrics(manifold, data, flat_torus_cls)
    mmd2 = rbf_mmd2(embed_samples, embed_data).item()
    nn_s2d, nn_d2s = symmetric_nearest_distances(manifold, samples, data, classes)
    data_radius = intrinsic_reference_radius(manifold, data, classes)
    coverage = (nn_d2s <= data_radius).float().mean().item()
    return {
        "mmd2": float(mmd2),
        "nn_sample_to_data": float(nn_s2d.mean().item()),
        "nn_data_to_sample": float(nn_d2s.mean().item()),
        "symmetric_nn": float(0.5 * (nn_s2d.mean() + nn_d2s.mean()).item()),
        "coverage_radius": float(data_radius),
        "coverage_at_radius": float(coverage),
        "constraint_error": float(constraint_error(manifold, samples)),
    }


def embed_for_metrics(manifold: Any, x: torch.Tensor, flat_torus_cls: type) -> torch.Tensor:
    if isinstance(manifold, flat_torus_cls):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    return x.reshape(x.shape[0], -1)


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0], 1024)
    x = x[:n]
    y = y[:n]
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    median = torch.median(d2[d2 > 0]).clamp_min(1e-6)
    gamma = 1.0 / median
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


@torch.no_grad()
def symmetric_nearest_distances(
    manifold: Any,
    samples: torch.Tensor,
    data: torch.Tensor,
    classes: tuple[type, type, type, type],
) -> tuple[torch.Tensor, torch.Tensor]:
    s2d = min_pairwise_dist(manifold, samples, data, classes)
    d2s = min_pairwise_dist(manifold, data, samples, classes)
    return s2d, d2s


@torch.no_grad()
def intrinsic_reference_radius(
    manifold: Any, data: torch.Tensor, classes: tuple[type, type, type, type]
) -> float:
    n = min(data.shape[0], 1024)
    x = data[:n]
    d = pairwise_dist(manifold, x, x, classes)
    eps = 1e-7
    d = d.masked_fill(d <= eps, float("inf"))
    nn = d.min(dim=1).values
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return eps
    return float(nn.median().item())


@torch.no_grad()
def min_pairwise_dist(
    manifold: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    classes: tuple[type, type, type, type],
    chunk: int = 512,
) -> torch.Tensor:
    mins = []
    for start in range(0, x.shape[0], chunk):
        d = pairwise_dist(manifold, x[start : start + chunk], y, classes)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0)


@torch.no_grad()
def pairwise_dist(
    manifold: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    classes: tuple[type, type, type, type],
) -> torch.Tensor:
    flat_torus_cls, sphere_cls, _, _ = classes
    if isinstance(manifold, sphere_cls):
        dots = x @ y.T
        return dots.clamp(-1.0, 1.0).acos()
    if isinstance(manifold, flat_torus_cls):
        diff = circular_difference(x[:, None, :], y[None, :, :])
        return diff.pow(2).sum(dim=-1).sqrt()
    return torch.cdist(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))


def circular_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x - y), torch.cos(x - y))


@torch.no_grad()
def constraint_error(manifold: Any, samples: torch.Tensor) -> float:
    try:
        projected = manifold.projx(samples)
        return (projected - samples).reshape(samples.shape[0], -1).norm(dim=-1).mean().item()
    except Exception:
        return float("nan")


def save_plot(
    name: str,
    manifold: Any,
    data: torch.Tensor,
    samples: torch.Tensor,
    path: Path,
    classes: tuple[type, type, type, type],
    lonlat_from_cartesian: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_torus_cls, sphere_cls, spd_cls, poincare_cls = classes
    if isinstance(manifold, sphere_cls) and data.shape[-1] == 3:
        plot_earth(data, samples, path, lonlat_from_cartesian)
    elif isinstance(manifold, flat_torus_cls):
        plot_torus(data, samples, path)
    elif isinstance(manifold, spd_cls):
        plot_spd(data, samples, path)
    elif isinstance(manifold, poincare_cls):
        plot_poincare(data, samples, path)
    elif manifold.__class__.__name__ == "Mesh":
        plot_mesh(data, samples, manifold, path)
    else:
        plot_ambient(data, samples, path)


def plot_earth(data: torch.Tensor, samples: torch.Tensor, path: Path, lonlat_fn: Any) -> None:
    data_ll = lonlat_fn(data) / math.pi * 180.0
    samples_ll = lonlat_fn(samples) / math.pi * 180.0
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    try:
        import geopandas as gpd

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, antialiased=False, color="#d8d8d8", edgecolor="#f7f7f7", linewidth=0.2)
    except Exception:
        ax.set_facecolor("#f1f1f1")
    ax.scatter(data_ll[:, 0], data_ll[:, 1], s=1.2, c="black", alpha=0.20, linewidths=0, label="test data")
    ax.scatter(samples_ll[:, 0], samples_ll[:, 1], s=1.5, c="#2166ac", alpha=0.38, linewidths=0, label="official RFM")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Official RFM / " + str(path.parent.parent.name))
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_torus(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    def deg(x: torch.Tensor) -> np.ndarray:
        return (((x + math.pi) % (2 * math.pi) - math.pi) / math.pi * 180.0).numpy()

    data_deg = deg(data)
    samples_deg = deg(samples)
    dim = data.shape[-1]
    if dim <= 2:
        pairs = [(0, 1)]
    else:
        pairs = [(i, i + 1) for i in range(min(dim - 1, 6))]
    cols = min(3, len(pairs))
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.0 * rows), squeeze=False)
    for ax, (i, j) in zip(axes.flat, pairs):
        ax.scatter(data_deg[:, i], data_deg[:, j], s=1.0, c="black", alpha=0.18, linewidths=0, label="test data")
        ax.scatter(samples_deg[:, i], samples_deg[:, j], s=1.3, c="#2166ac", alpha=0.36, linewidths=0, label="official RFM")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.35)
        ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.35)
        ax.set_xlabel(f"angle {i}")
        ax.set_ylabel(f"angle {j}")
    for ax in axes.flat[len(pairs) :]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_spd(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig = plt.figure(figsize=(6.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    def coords(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = x.numpy()
        c = x[:, 1]
        u = 0.5 * (x[:, 0] + x[:, 2])
        v = 0.5 * (x[:, 0] - x[:, 2])
        return c, v, u

    d = data[:, :3]
    s = samples[:, :3]
    ax.scatter(*coords(d), marker=".", c="black", s=2.0, alpha=0.24, label="test data")
    ax.scatter(*coords(s), marker=".", c="#2166ac", s=2.5, alpha=0.46, label="official RFM")
    ax.set_xlabel("b")
    ax.set_ylabel("(a-c)/2")
    ax.set_zlabel("(a+c)/2")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_poincare(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(data[:, 0], data[:, 1], s=1.3, c="black", alpha=0.22, linewidths=0, label="test data")
    ax.scatter(samples[:, 0], samples[:, 1], s=1.6, c="#2166ac", alpha=0.42, linewidths=0, label="official RFM")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="black", fill=False, lw=1.0))
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_mesh(data: torch.Tensor, samples: torch.Tensor, manifold: Any, path: Path) -> None:
    fig = plt.figure(figsize=(6.4, 5.9))
    ax = fig.add_subplot(111, projection="3d")
    v = manifold.v.detach().cpu().numpy()
    f = manifold.f.detach().cpu().numpy()
    try:
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#d8d8d8", alpha=0.18, linewidth=0)
    except Exception:
        pass
    d = data.numpy()
    s = samples.numpy()
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=1.6, c="black", alpha=0.18, linewidths=0, label="test data")
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2.0, c="#2166ac", alpha=0.42, linewidths=0, label="official RFM")
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-45)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def plot_ambient(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    d = data.reshape(data.shape[0], -1).numpy()
    s = samples.reshape(samples.shape[0], -1).numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(d[:, 0], d[:, 1], s=1.4, c="black", alpha=0.20, linewidths=0, label="test data")
    ax.scatter(s[:, 0], s[:, 1], s=1.8, c="#2166ac", alpha=0.42, linewidths=0, label="official RFM")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
