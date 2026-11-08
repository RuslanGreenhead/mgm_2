"""Standalone Riemannian MGM experiments built on the RFM manifold code.

The original RFM trainer is Lightning/Hydra based and geared toward ODE-flow
likelihoods. This script keeps the geometric pieces but trains a one-step
generator with the variational Midpoint Generative Models objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifm.manifolds.hyperbolic import PoincareBall
from manifm.manifolds.spd import SPD
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus
from manifm.rmgm import (
    build_critic,
    build_generator,
    constraint_error,
    critic_loss,
    generator_loss,
)


@dataclass
class TrainConfig:
    experiment: str
    steps: int = 5000
    batch_size: int = 512
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    critic_steps: int = 1
    lr_g: float = 2e-4
    lr_c: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 0
    log_every: int = 100
    plot_every: int = 1000
    eval_samples: int = 4096
    device: str = "cuda"
    output_dir: str = "outputs/rmgm"
    dtype: str = "float32"
    metric_normalize: bool = False
    generator_type: str = "mlp"
    generator_components: int = 4
    generator_residual_scale: float = 0.25
    generator_kmeans_init: bool = False
    generator_kmeans_samples: int = 8192
    generator_kmeans_iters: int = 40
    torus_map: str = "mod"
    torus_generator: str = "mlp"
    torus_components: int = 4
    torus_residual_scale: float = 0.35
    torus_mixture_init: str = "grid"
    torus_kmeans_init: bool = False
    torus_kmeans_samples: int = 8192
    torus_kmeans_iters: int = 40


class WrappedSampler:
    def __init__(
        self,
        manifold: Any,
        dim: int,
        centers: torch.Tensor,
        scale: float,
    ):
        self.manifold = manifold
        self.dim = dim
        self.centers = centers
        self.scale = scale

    def sample(
        self, n: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        centers = self.centers.to(device=device, dtype=dtype)
        idx = torch.randint(centers.shape[0], (n,), device=device)
        mean = centers[idx]
        tangent = torch.randn(n, self.dim, device=device, dtype=dtype)
        tangent = self.manifold.proju(mean, tangent)
        x = self.manifold.expmap(mean, self.scale * tangent)
        return self.manifold.projx(x)


def make_experiment(name: str) -> Tuple[Any, int, WrappedSampler]:
    normalized = name.lower()
    if normalized in {"torus", "torus2d", "wrapped_torus"}:
        manifold = FlatTorus()
        dim = 2
        centers = torch.tensor(
            [
                [0.45, 1.10],
                [2.10, 4.80],
                [4.85, 2.35],
                [5.55, 5.15],
            ]
        )
        return manifold, dim, WrappedSampler(manifold, dim, centers, scale=0.23)

    if normalized in {"spd", "spd2", "wrapped_spd"}:
        manifold = SPD(scale_std=0.5, scale_Id=3.0, base_expmap=False)
        dim = 3
        eye = torch.eye(2) * 2.0
        centers = manifold.vectorize(eye).reshape(1, -1)
        return manifold, dim, WrappedSampler(manifold, dim, centers, scale=0.40)

    if normalized in {"hyperbolic", "poincare"}:
        manifold = PoincareBall()
        dim = 2
        centers = torch.tensor(
            [
                [-0.55, 0.55],
                [0.55, -0.55],
                [-0.35, -0.25],
                [0.35, 0.25],
            ]
        )
        return manifold, dim, WrappedSampler(manifold, dim, centers, scale=0.20)

    if normalized in {"sphere", "sphere2", "s2"}:
        manifold = Sphere()
        dim = 3
        centers = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.25, 0.96824584],
                [-0.60, -0.60, -0.52915025],
            ]
        )
        centers = centers / centers.norm(dim=-1, keepdim=True)
        return manifold, dim, WrappedSampler(manifold, dim, centers, scale=0.27)

    raise ValueError(f"Unknown R-MGM experiment: {name}")


def run(cfg: TrainConfig) -> Path:
    _seed_everything(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    manifold, dim, data_sampler = make_experiment(cfg.experiment)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / cfg.experiment / f"seed{cfg.seed}_{stamp}"
    figs_dir = out_dir / "figs"
    ckpt_dir = out_dir / "checkpoints"
    figs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    critic_cfg = {
        "critic": {
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "activation": "silu",
            "fourier": 8,
            "metric_normalize": cfg.metric_normalize,
        }
    }
    generator = build_generator(generator_cfg, manifold, dim).to(device=device, dtype=dtype)
    critic = build_critic(critic_cfg, manifold, dim).to(device=device, dtype=dtype)
    maybe_init_torus_mixture(cfg, manifold, data_sampler, generator, device, dtype)
    maybe_init_local_gmm(cfg, manifold, data_sampler, generator, device, dtype)

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr_g, weight_decay=cfg.weight_decay
    )
    opt_c = torch.optim.AdamW(
        critic.parameters(), lr=cfg.lr_c, weight_decay=cfg.weight_decay
    )

    metrics_path = out_dir / "metrics.jsonl"
    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifold": manifold.__class__.__name__,
        "dim": dim,
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Writing R-MGM run to: {out_dir}")

    latest_log = {}
    best_mmd2 = float("inf")
    for step in range(1, cfg.steps + 1):
        critic.train()
        generator.train()
        for _ in range(cfg.critic_steps):
            z = torch.randn(
                cfg.batch_size, cfg.latent_dim, device=device, dtype=dtype
            )
            x1 = data_sampler.sample(cfg.batch_size, device, dtype)
            opt_c.zero_grad(set_to_none=True)
            loss_c, c_metrics = critic_loss(manifold, generator, critic, z, x1)
            loss_c.backward()
            opt_c.step()

        _set_requires_grad(critic, False)
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device, dtype=dtype)
        x1 = data_sampler.sample(cfg.batch_size, device, dtype)
        opt_g.zero_grad(set_to_none=True)
        loss_g, g_metrics = generator_loss(manifold, generator, critic, z, x1)
        loss_g.backward()
        opt_g.step()
        _set_requires_grad(critic, True)

        if step == 1 or step % cfg.log_every == 0:
            latest_log = evaluate(
                cfg,
                manifold,
                data_sampler,
                generator,
                device,
                dtype,
                step,
                c_metrics,
                g_metrics,
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(latest_log) + "\n")
            print(json.dumps(latest_log))
            if latest_log["mmd2"] < best_mmd2:
                best_mmd2 = latest_log["mmd2"]
                torch.save(
                    {
                        "step": step,
                        "generator": generator.state_dict(),
                        "critic": critic.state_dict(),
                        "config": asdict(cfg),
                        "latest_log": latest_log,
                    },
                    ckpt_dir / "best.pt",
                )

        if step == 1 or step % cfg.plot_every == 0 or step == cfg.steps:
            save_plots(
                cfg,
                manifold,
                data_sampler,
                generator,
                device,
                dtype,
                figs_dir,
                step,
            )
            torch.save(
                {
                    "step": step,
                    "generator": generator.state_dict(),
                    "critic": critic.state_dict(),
                    "config": asdict(cfg),
                    "latest_log": latest_log,
                },
                ckpt_dir / "last.pt",
            )

    metrics = {}
    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            metrics = json.loads(lines[-1])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_dir


@torch.no_grad()
def evaluate(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    generator,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    c_metrics: dict,
    g_metrics: dict,
) -> dict:
    n = min(cfg.eval_samples, 4096)
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    samples = generator(z)
    data = data_sampler.sample(n, device, dtype)
    embed_samples = embed_for_metrics(cfg.experiment, manifold, samples)
    embed_data = embed_for_metrics(cfg.experiment, manifold, data)
    mmd2 = rbf_mmd2(embed_samples, embed_data).item()
    err = constraint_error(manifold, samples, reduction="mean").item()
    return {
        "step": step,
        "critic_loss": float(c_metrics["critic_loss"].cpu()),
        "generator_loss": float(g_metrics["generator_loss"].cpu()),
        "target_norm": float(c_metrics["target_norm"].cpu()),
        "sample_norm": float(g_metrics["sample_norm"].cpu()),
        "mmd2": mmd2,
        "constraint_error": err,
    }


@torch.no_grad()
def save_plots(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    generator,
    device: torch.device,
    dtype: torch.dtype,
    figs_dir: Path,
    step: int,
) -> None:
    n = min(cfg.eval_samples, 6000)
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    samples = generator(z).detach().cpu()
    data = data_sampler.sample(n, device, dtype).detach().cpu()
    name = cfg.experiment.lower()
    if "torus" in name:
        plot_torus(data, samples, figs_dir / f"samples-{step:06d}.png")
    elif "spd" in name:
        plot_spd(data, samples, manifold, figs_dir / f"samples-{step:06d}.png")
    elif "hyperbolic" in name or "poincare" in name:
        plot_poincare(
            data,
            samples,
            figs_dir / f"samples-{step:06d}.png",
            radius=poincare_plot_radius(manifold),
        )
    elif "sphere" in name or name == "s2":
        plot_sphere(data, samples, figs_dir / f"samples-{step:06d}.png")
    else:
        plot_scatter(data, samples, figs_dir / f"samples-{step:06d}.png")


def embed_for_metrics(name: str, manifold: Any, x: torch.Tensor) -> torch.Tensor:
    del manifold
    if "torus" in name.lower():
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
def maybe_init_torus_mixture(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not cfg.torus_kmeans_init or not isinstance(manifold, FlatTorus):
        return
    if not hasattr(generator, "centers") or not hasattr(generator, "log_scale"):
        return

    data = data_sampler.sample(cfg.torus_kmeans_samples, device, dtype)
    centers = circular_kmeans(
        data,
        generator.centers.shape[0],
        iters=cfg.torus_kmeans_iters,
        init=generator.centers.detach().to(device=device, dtype=dtype),
    )
    diff = circular_difference(data[:, None, :], centers[None, :, :])
    assignment = diff.pow(2).sum(dim=-1).argmin(dim=1)
    scales = torch.empty_like(centers)
    for k in range(centers.shape[0]):
        cluster_diff = circular_difference(data[assignment == k], centers[k])
        if cluster_diff.numel() == 0:
            scales[k].fill_(cfg.torus_residual_scale)
        else:
            scales[k] = cluster_diff.std(dim=0).clamp(0.05, 1.00)

    generator.centers.copy_(centers.to(generator.centers))
    generator.log_scale.copy_(_inverse_softplus(scales.to(generator.log_scale)))
    if hasattr(generator, "component_logits"):
        counts = torch.bincount(assignment, minlength=centers.shape[0]).to(scales)
        probs = (counts + 1.0) / (counts.sum() + centers.shape[0])
        generator.component_logits.copy_(probs.log().to(generator.component_logits))


@torch.no_grad()
def maybe_init_local_gmm(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not cfg.generator_kmeans_init or isinstance(manifold, FlatTorus):
        return
    if not hasattr(generator, "centers") or not hasattr(generator, "log_scale"):
        return

    data = data_sampler.sample(cfg.generator_kmeans_samples, device, dtype)
    init = (
        generator.projected_centers()
        if hasattr(generator, "projected_centers")
        else generator.centers
    )
    centers = euclidean_kmeans(
        data.reshape(data.shape[0], -1),
        generator.centers.shape[0],
        iters=cfg.generator_kmeans_iters,
        init=init.detach().reshape(generator.centers.shape[0], -1),
    ).reshape_as(generator.centers)
    centers = manifold.projx(centers)

    diff = (data[:, None, :] - centers[None, :, :]).reshape(data.shape[0], centers.shape[0], -1)
    assignment = diff.pow(2).sum(dim=-1).argmin(dim=1)
    scales = torch.empty_like(centers)
    for k in range(centers.shape[0]):
        points = data[assignment == k]
        if points.numel() == 0:
            scales[k].fill_(cfg.generator_residual_scale)
            continue
        center = centers[k].expand(points.shape[0], -1)
        tangent = manifold.logmap(center, points)
        scales[k] = tangent.std(dim=0).abs().clamp(0.03, 1.00)

    generator.centers.copy_(centers.to(generator.centers))
    generator.log_scale.copy_(_inverse_softplus(scales.to(generator.log_scale)))
    if hasattr(generator, "component_logits"):
        counts = torch.bincount(assignment, minlength=centers.shape[0]).to(scales)
        probs = (counts + 1.0) / (counts.sum() + centers.shape[0])
        generator.component_logits.copy_(probs.log().to(generator.component_logits))


@torch.no_grad()
def euclidean_kmeans(
    x: torch.Tensor,
    components: int,
    *,
    iters: int,
    init: torch.Tensor,
) -> torch.Tensor:
    centers = init[:components].clone()
    for _ in range(iters):
        dist2 = torch.cdist(x, centers).pow(2)
        assignment = dist2.argmin(dim=1)
        for k in range(components):
            points = x[assignment == k]
            if points.numel() == 0:
                continue
            centers[k] = points.mean(dim=0)
    return centers


@torch.no_grad()
def circular_kmeans(
    x: torch.Tensor,
    components: int,
    *,
    iters: int,
    init: torch.Tensor,
) -> torch.Tensor:
    centers = init[:components].clone()
    for _ in range(iters):
        diff = circular_difference(x[:, None, :], centers[None, :, :])
        assignment = diff.pow(2).sum(dim=-1).argmin(dim=1)
        for k in range(components):
            points = x[assignment == k]
            if points.numel() == 0:
                continue
            centers[k] = torch.atan2(
                torch.sin(points).mean(dim=0),
                torch.cos(points).mean(dim=0),
            ).remainder(2.0 * math.pi)
    return centers


def circular_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x - y), torch.cos(x - y))


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def plot_torus(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    def deg(x):
        return ((x + math.pi) % (2 * math.pi) - math.pi) / math.pi * 180.0

    data = deg(data.numpy())
    samples = deg(samples.numpy())
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(data[:, 0], data[:, 1], s=1.0, c="black", alpha=0.22, label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=1.2, c="#d97732", alpha=0.45, label="R-MGM")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$", rotation=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_poincare(
    data: torch.Tensor, samples: torch.Tensor, path: Path, radius: float = 1.0
) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 5.7))
    ax.scatter(data[:, 0], data[:, 1], s=1.3, c="black", alpha=0.25, label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=1.6, c="#2a7f9e", alpha=0.48, label="R-MGM")
    ax.add_patch(plt.Circle((0, 0), radius, color="black", fill=False, lw=1.0))
    pad = radius * 0.05
    ax.set_xlim(-(radius + pad), radius + pad)
    ax.set_ylim(-(radius + pad), radius + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def poincare_plot_radius(manifold: Any) -> float:
    c = getattr(manifold, "c", 1.0)
    if not torch.is_tensor(c):
        c = torch.as_tensor(c)
    return float(c.detach().cpu().rsqrt())


def plot_sphere(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig = plt.figure(figsize=(6.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#edf5fa", alpha=0.24, linewidth=0)
    d = data.numpy()
    s = samples.numpy()
    ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=1.5, c="black", alpha=0.22, label="data")
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1.8, c="#4d8b55", alpha=0.50, label="R-MGM")
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spd(data: torch.Tensor, samples: torch.Tensor, manifold: SPD, path: Path) -> None:
    del manifold

    def coords(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = x.numpy()
        c = x[:, 1]
        u = 0.5 * (x[:, 0] + x[:, 2])
        v = 0.5 * (x[:, 0] - x[:, 2])
        return c, v, u

    fig = plt.figure(figsize=(6.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    rr = np.linspace(0, 3.4, 80)
    th = np.linspace(0, 2 * np.pi, 80)
    R, T = np.meshgrid(rr, th)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = R
    ax.plot_surface(X, Y, Z, color="#edf5fa", alpha=0.20, linewidth=0)
    dc, dv, du = coords(data)
    sc, sv, su = coords(samples)
    ax.scatter(dc, dv, du, s=2.0, c="black", alpha=0.23, label="data")
    ax.scatter(sc, sv, su, s=2.4, c="#7656a5", alpha=0.52, label="R-MGM")
    ax.view_init(azim=88, elev=12)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scatter(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    data = data.numpy()
    samples = samples.numpy()
    ax.scatter(data[:, 0], data[:, 1], s=2, c="black", alpha=0.25, label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=2, c="#d97732", alpha=0.45, label="R-MGM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-c", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/rmgm")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--metric-normalize", action="store_true")
    parser.add_argument("--generator-type", default="mlp", choices=["mlp", "local_gmm"])
    parser.add_argument("--generator-components", type=int, default=4)
    parser.add_argument("--generator-residual-scale", type=float, default=0.25)
    parser.add_argument("--generator-kmeans-init", action="store_true")
    parser.add_argument("--generator-kmeans-samples", type=int, default=8192)
    parser.add_argument("--generator-kmeans-iters", type=int, default=40)
    parser.add_argument("--torus-map", default="mod", choices=["mod", "sigmoid"])
    parser.add_argument(
        "--torus-generator",
        default="mlp",
        choices=["mlp", "mixture", "gmm", "gaussian", "gaussian_mixture"],
    )
    parser.add_argument("--torus-components", type=int, default=4)
    parser.add_argument("--torus-residual-scale", type=float, default=0.35)
    parser.add_argument("--torus-mixture-init", default="grid", choices=["grid", "random"])
    parser.add_argument("--torus-kmeans-init", action="store_true")
    parser.add_argument("--torus-kmeans-samples", type=int, default=8192)
    parser.add_argument("--torus-kmeans-iters", type=int, default=40)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
