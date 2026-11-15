"""Lightweight Riemannian Flow Matching baseline for R-MGM experiments.

This script intentionally mirrors ``rmgm_train.py``: it uses the same synthetic
target samplers, the same MMD embedding, and the same plotting coordinates. The
baseline trains a time-dependent tangent vector field with the standard RFM
loss and samples by integrating that field from a simple manifold base
distribution.
"""

from __future__ import annotations

import argparse
import json
import math
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
from manifm.rmgm import build_critic, constraint_error, metric_inner, midpoint_path
from rmgm_train import WrappedSampler, embed_for_metrics, make_experiment, rbf_mmd2


@dataclass
class TrainConfig:
    experiment: str
    steps: int = 5000
    batch_size: int = 512
    hidden_dim: int = 256
    num_layers: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 0
    log_every: int = 100
    plot_every: int = 1000
    eval_samples: int = 4096
    ode_steps: int = 128
    device: str = "cuda"
    output_dir: str = "outputs/rfm_baseline"
    dtype: str = "float32"
    metric_normalize: bool = False
    hyperbolic_base_scale: float = 0.45


class BaseSampler:
    """Simple base distributions in the same representation as RFM manifolds."""

    def __init__(self, manifold: Any, dim: int, hyperbolic_scale: float = 0.45):
        self.manifold = manifold
        self.dim = dim
        self.hyperbolic_scale = hyperbolic_scale

    def sample(
        self, n: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if isinstance(self.manifold, PoincareBall):
            return self._sample_poincare(n, device, dtype)

        try:
            x = self.manifold.random_base(n, self.dim, dtype=dtype, device=device)
        except TypeError:
            x = self.manifold.random_base(n, self.dim).to(device=device, dtype=dtype)
        x = x.reshape(n, self.dim)
        return self.manifold.projx(x)

    def _sample_poincare(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        u = torch.randn(n, self.dim, device=device, dtype=dtype) * self.hyperbolic_scale
        if hasattr(self.manifold, "expmap0"):
            return self.manifold.projx(self.manifold.expmap0(u))

        norm = u.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        c = getattr(self.manifold, "c", 1.0)
        if not torch.is_tensor(c):
            c = torch.as_tensor(c, dtype=dtype, device=device)
        radius = c.rsqrt()
        return u / norm * torch.tanh(norm) * radius * 0.995


def run(cfg: TrainConfig) -> Path:
    _seed_everything(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    manifold, dim, data_sampler = make_experiment(cfg.experiment)
    base_sampler = BaseSampler(
        manifold, dim, hyperbolic_scale=cfg.hyperbolic_base_scale
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / cfg.experiment / f"seed{cfg.seed}_{stamp}"
    figs_dir = out_dir / "figs"
    ckpt_dir = out_dir / "checkpoints"
    figs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    opt = torch.optim.AdamW(
        field.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    metrics_path = out_dir / "metrics.jsonl"
    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifold": manifold.__class__.__name__,
        "dim": dim,
        "baseline": "rfm_euler",
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Writing RFM baseline run to: {out_dir}")

    latest_log = {}
    best_mmd2 = float("inf")
    for step in range(1, cfg.steps + 1):
        field.train()
        x0 = base_sampler.sample(cfg.batch_size, device, dtype)
        x1 = data_sampler.sample(cfg.batch_size, device, dtype)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        x_t, u_t = rfm_path(manifold, x0, x1, t)

        opt.zero_grad(set_to_none=True)
        pred = field(t, x_t)
        diff = pred - u_t
        loss = metric_inner(manifold, x_t, diff, diff).mean() / dim
        loss.backward()
        opt.step()

        if step == 1 or step % cfg.log_every == 0:
            latest_log = evaluate(
                cfg,
                manifold,
                data_sampler,
                base_sampler,
                field,
                device,
                dtype,
                step,
                loss.detach(),
                x_t.detach(),
                u_t.detach(),
                pred.detach(),
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(latest_log) + "\n")
            print(json.dumps(latest_log))
            if latest_log["mmd2"] < best_mmd2:
                best_mmd2 = latest_log["mmd2"]
                torch.save(
                    {
                        "step": step,
                        "field": field.state_dict(),
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
                base_sampler,
                field,
                device,
                dtype,
                figs_dir,
                step,
            )
            torch.save(
                {
                    "step": step,
                    "field": field.state_dict(),
                    "config": asdict(cfg),
                    "latest_log": latest_log,
                },
                ckpt_dir / "last.pt",
            )

    metrics = {}
    if metrics_path.exists():
        lines = [
            line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
        ]
        if lines:
            metrics = json.loads(lines[-1])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_dir


@torch.no_grad()
def evaluate(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    base_sampler: BaseSampler,
    field: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    loss: torch.Tensor,
    x_t: torch.Tensor,
    u_t: torch.Tensor,
    pred: torch.Tensor,
) -> dict:
    n = min(cfg.eval_samples, 4096)
    samples = sample_rfm(cfg, manifold, base_sampler, field, n, device, dtype)
    data = data_sampler.sample(n, device, dtype)
    embed_samples = embed_for_metrics(cfg.experiment, manifold, samples)
    embed_data = embed_for_metrics(cfg.experiment, manifold, data)
    mmd2 = rbf_mmd2(embed_samples, embed_data).item()
    err = constraint_error(manifold, samples, reduction="mean").item()
    diff = pred - u_t
    return {
        "step": step,
        "rfm_loss": float(loss.cpu()),
        "target_norm": float(metric_inner(manifold, x_t, u_t, u_t).mean().cpu()),
        "pred_norm": float(metric_inner(manifold, x_t, pred, pred).mean().cpu()),
        "train_residual": float(metric_inner(manifold, x_t, diff, diff).mean().cpu()),
        "mmd2": mmd2,
        "constraint_error": err,
        "ode_steps": cfg.ode_steps,
    }


@torch.no_grad()
def sample_rfm(
    cfg: TrainConfig,
    manifold: Any,
    base_sampler: BaseSampler,
    field: torch.nn.Module,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    field.eval()
    x = base_sampler.sample(n, device, dtype)
    dt = 1.0 / cfg.ode_steps
    for i in range(cfg.ode_steps):
        t = torch.full(
            (n, 1), (i + 0.5) * dt, device=device, dtype=dtype
        )
        v = field(t, x)
        x = _manifold_euler_step(manifold, x, v, dt)
    return manifold.projx(x)


def _manifold_euler_step(
    manifold: Any, x: torch.Tensor, v: torch.Tensor, dt: float
) -> torch.Tensor:
    try:
        x_next = manifold.expmap(x, dt * v)
    except NotImplementedError:
        x_next = x + dt * v
    return manifold.projx(x_next)


def rfm_path(
    manifold: Any, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(manifold, SPD):
        return _spd_eigh_path(manifold, x0, x1, t)
    return midpoint_path(manifold, x0, x1, t)


def _spd_eigh_path(
    manifold: SPD, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    A = manifold.devectorize(x0)
    B = manifold.devectorize(x1)
    dtype = A.dtype
    A = A.double()
    B = B.double()
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A_sqrt = _sqrtmh(A) + 1e-6 * eye
    inv_A_sqrt = torch.linalg.inv(A_sqrt)
    C = inv_A_sqrt @ B @ inv_A_sqrt
    C = 0.5 * (C + C.transpose(-1, -2))

    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = eigvals.clamp_min(1e-12)
    time = _time_vector(t, x0).double().unsqueeze(-1)
    C_t = eigvecs @ torch.diag_embed(eigvals.pow(time)) @ eigvecs.transpose(-1, -2)
    log_C = eigvecs @ torch.diag_embed(eigvals.log()) @ eigvecs.transpose(-1, -2)
    x_t = A_sqrt @ C_t @ A_sqrt
    u_t = A_sqrt @ (C_t @ log_C) @ A_sqrt
    return manifold.vectorize(x_t).to(dtype), manifold.vectorize(u_t).to(dtype)


def _sqrtmh(A: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp_min(0.0)
    return (eigvecs * eigvals.sqrt().unsqueeze(-2)) @ eigvecs.transpose(-1, -2)


def _time_vector(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        return t.expand(batch)
    if t.shape == (batch,):
        return t
    if t.shape == (batch, 1):
        return t.reshape(batch)
    raise ValueError(
        f"Expected scalar, ({batch},), or ({batch}, 1) time tensor; got {tuple(t.shape)}."
    )


@torch.no_grad()
def save_plots(
    cfg: TrainConfig,
    manifold: Any,
    data_sampler: WrappedSampler,
    base_sampler: BaseSampler,
    field: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    figs_dir: Path,
    step: int,
) -> None:
    n = min(cfg.eval_samples, 6000)
    samples = sample_rfm(cfg, manifold, base_sampler, field, n, device, dtype).cpu()
    data = data_sampler.sample(n, device, dtype).cpu()
    name = cfg.experiment.lower()
    if "torus" in name:
        plot_torus(data, samples, figs_dir / f"samples-{step:06d}.png")
    elif "spd" in name:
        plot_spd(data, samples, figs_dir / f"samples-{step:06d}.png")
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


def plot_torus(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    def deg(x):
        return ((x + math.pi) % (2 * math.pi) - math.pi) / math.pi * 180.0

    data = deg(data.numpy())
    samples = deg(samples.numpy())
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(data[:, 0], data[:, 1], s=1.0, c="black", alpha=0.22, label="data")
    ax.scatter(samples[:, 0], samples[:, 1], s=1.2, c="#3569b8", alpha=0.45, label="RFM")
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
    ax.scatter(samples[:, 0], samples[:, 1], s=1.6, c="#3569b8", alpha=0.48, label="RFM")
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
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1.8, c="#3569b8", alpha=0.50, label="RFM")
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spd(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
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
    ax.scatter(sc, sv, su, s=2.4, c="#3569b8", alpha=0.52, label="RFM")
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
    ax.scatter(samples[:, 0], samples[:, 1], s=2, c="#3569b8", alpha=0.45, label="RFM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/rfm_baseline")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--metric-normalize", action="store_true")
    parser.add_argument("--hyperbolic-base-scale", type=float, default=0.45)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
