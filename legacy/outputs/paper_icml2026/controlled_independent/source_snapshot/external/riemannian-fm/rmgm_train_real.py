"""Real-data R-MGM and lightweight RFM experiments.

This script is the real-data counterpart of ``rmgm_train.py``.  It reads the
datasets shipped with the RFM repository, keeps the author's deterministic
train/val/test split, and trains either the implicit one-step R-MGM generator or
a lightweight sample-based RFM baseline with the same sample metrics.

R-MGM is implicit, so this file does not report exact NLL.  The original RFM
Lightning pipeline in ``train.py``/``evaluate.py`` is still the right path for
ODE-flow likelihoods.  Here we report sample metrics that are available for
both methods.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from manifm.datasets import _get_dataset
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus
from manifm.manifolds.mesh import Metric
from manifm.manifolds.spd import SPD, plot_cone
from manifm.rmgm import (
    build_critic,
    build_generator,
    constraint_error,
    critic_loss,
    generator_loss,
    metric_inner,
)
from manifm.utils import lonlat_from_cartesian
from rfm_baseline_train import BaseSampler, rfm_path, sample_rfm
from rmgm_train import (
    circular_difference,
    rbf_mmd2,
)


EARTH_DATASETS = {"volcano", "earthquake", "fire", "flood"}
TORUS_DATASETS = {"general", "glycine", "proline", "prepro", "rna"}
MESH_DATASETS = {
    "simple_bunny",
    "bunny10",
    "bunny50",
    "bunny100",
    "spot10",
    "spot50",
    "spot100",
    "maze3v2",
    "maze4v2",
}


@dataclass
class RealTrainConfig:
    data: str
    method: str = "rmgm"
    steps: int = 5000
    batch_size: int = 512
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    critic_steps: int = 1
    lr_g: float = 2e-4
    lr_c: float = 2e-4
    lr: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 0
    data_seed: int | None = None
    log_every: int = 100
    plot_every: int = 1000
    eval_samples: int = 4096
    ode_steps: int = 128
    device: str = "cuda"
    output_dir: str = "../../outputs/rmgm_real"
    dtype: str = "float32"
    metric_normalize: bool = False
    earth_datadir: str = "data/earth_data"
    top500_datadir: str = "data/top500"
    rna_datadir: str = "data/rna"
    mesh_datadir: str = "data/mesh"
    wrapped_dim: int | None = None
    wrapped_n_mixtures: int = 1
    wrapped_scale: float | None = None
    mesh_numeigs: int | None = None
    mesh_metric: str | None = None
    mesh_temp: float = 0.25
    generator_type: str = "mlp"
    generator_components: int = 4
    generator_residual_scale: float = 0.25
    generator_kmeans_init: bool = False
    generator_kmeans_samples: int = 8192
    generator_kmeans_iters: int = 40
    sphere_map_temperature: float = 80.0
    sphere_map_vertex_stride: int = 1
    torus_map: str = "mod"
    torus_generator: str = "mlp"
    torus_components: int = 8
    torus_residual_scale: float = 0.35
    torus_mixture_init: str = "grid"
    torus_kmeans_init: bool = False
    torus_kmeans_samples: int = 8192
    torus_kmeans_iters: int = 40


class DotConfig(dict):
    """Small OmegaConf-like mapping for the repository dataset helpers."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class EmpiricalSampler:
    def __init__(self, data: torch.Tensor, manifold: Any):
        self.data = data.detach().cpu()
        self.manifold = manifold

    def sample(
        self, n: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        idx = torch.randint(len(self.data), (n,))
        x = self.data[idx].to(device=device, dtype=dtype)
        return self.manifold.projx(x)

    def all(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.data.to(device=device, dtype=dtype)


def run(cfg: RealTrainConfig) -> Path:
    _seed_everything(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    real_data = load_real_dataset(cfg)
    manifold = real_data["manifold"]
    dim = real_data["dim"]
    train_sampler = EmpiricalSampler(real_data["train"], manifold)
    eval_sampler = EmpiricalSampler(real_data["test"], manifold)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / cfg.data / cfg.method / f"seed{cfg.seed}_{stamp}"
    figs_dir = out_dir / "figs"
    ckpt_dir = out_dir / "checkpoints"
    figs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifold": manifold.__class__.__name__,
        "dim": dim,
        "dataset_sizes": real_data["sizes"],
        "wrapped": real_data["wrapped"],
        "mesh_note": real_data["mesh_note"],
        "metrics_note": (
            "R-MGM is an implicit one-step generator; exact NLL is not computed "
            "here. Use train.py/evaluate.py for original RFM likelihoods."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Writing real-data run to: {out_dir}")

    if cfg.method == "rmgm":
        _run_rmgm(cfg, manifold, dim, train_sampler, eval_sampler, device, dtype, out_dir)
    elif cfg.method == "rfm":
        _run_rfm(cfg, manifold, dim, train_sampler, eval_sampler, device, dtype, out_dir)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    return out_dir


def _run_rmgm(
    cfg: RealTrainConfig,
    manifold: Any,
    dim: int,
    train_sampler: EmpiricalSampler,
    eval_sampler: EmpiricalSampler,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
) -> None:
    figs_dir = out_dir / "figs"
    ckpt_dir = out_dir / "checkpoints"
    generator_cfg = {
        "generator": {
            "latent_dim": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "activation": "silu",
            "generator_type": cfg.generator_type,
            "generator_components": cfg.generator_components,
            "generator_residual_scale": cfg.generator_residual_scale,
            "sphere_map_temperature": cfg.sphere_map_temperature,
            "sphere_map_vertex_stride": cfg.sphere_map_vertex_stride,
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
    maybe_init_torus_mixture_real(cfg, manifold, train_sampler, generator, device, dtype)
    maybe_init_local_gmm_real(cfg, manifold, train_sampler, generator, device, dtype)

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr_g, weight_decay=cfg.weight_decay
    )
    opt_c = torch.optim.AdamW(
        critic.parameters(), lr=cfg.lr_c, weight_decay=cfg.weight_decay
    )

    metrics_path = out_dir / "metrics.jsonl"
    latest_log: dict[str, Any] = {}
    best_score = float("inf")
    for step in range(1, cfg.steps + 1):
        critic.train()
        generator.train()
        for _ in range(cfg.critic_steps):
            z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device, dtype=dtype)
            x1 = train_sampler.sample(cfg.batch_size, device, dtype)
            opt_c.zero_grad(set_to_none=True)
            loss_c, c_metrics = critic_loss(manifold, generator, critic, z, x1)
            loss_c.backward()
            opt_c.step()

        _set_requires_grad(critic, False)
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device, dtype=dtype)
        x1 = train_sampler.sample(cfg.batch_size, device, dtype)
        opt_g.zero_grad(set_to_none=True)
        loss_g, g_metrics = generator_loss(manifold, generator, critic, z, x1)
        loss_g.backward()
        opt_g.step()
        _set_requires_grad(critic, True)

        if step == 1 or step % cfg.log_every == 0:
            latest_log = evaluate_rmgm(
                cfg,
                manifold,
                eval_sampler,
                generator,
                device,
                dtype,
                step,
                c_metrics,
                g_metrics,
            )
            _append_jsonl(metrics_path, latest_log)
            print(json.dumps(latest_log))
            if latest_log["symmetric_nn"] < best_score:
                best_score = latest_log["symmetric_nn"]
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
            save_real_plots(
                cfg,
                manifold,
                eval_sampler,
                generator=generator,
                field=None,
                device=device,
                dtype=dtype,
                figs_dir=figs_dir,
                step=step,
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

    _write_last_metrics(out_dir, metrics_path)


def _run_rfm(
    cfg: RealTrainConfig,
    manifold: Any,
    dim: int,
    train_sampler: EmpiricalSampler,
    eval_sampler: EmpiricalSampler,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
) -> None:
    figs_dir = out_dir / "figs"
    ckpt_dir = out_dir / "checkpoints"
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
    base_sampler = BaseSampler(manifold, dim)
    opt = torch.optim.AdamW(field.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = out_dir / "metrics.jsonl"
    latest_log: dict[str, Any] = {}
    best_score = float("inf")
    for step in range(1, cfg.steps + 1):
        field.train()
        x0 = base_sampler.sample(cfg.batch_size, device, dtype)
        x1 = train_sampler.sample(cfg.batch_size, device, dtype)
        t = torch.rand(cfg.batch_size, 1, device=device, dtype=dtype)
        x_t, u_t = rfm_path(manifold, x0, x1, t)

        opt.zero_grad(set_to_none=True)
        pred = field(t, x_t)
        diff = pred - u_t
        loss = metric_inner(manifold, x_t, diff, diff).mean() / dim
        loss.backward()
        opt.step()

        if step == 1 or step % cfg.log_every == 0:
            latest_log = evaluate_rfm(
                cfg,
                manifold,
                eval_sampler,
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
            _append_jsonl(metrics_path, latest_log)
            print(json.dumps(latest_log))
            if latest_log["symmetric_nn"] < best_score:
                best_score = latest_log["symmetric_nn"]
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
            save_real_plots(
                cfg,
                manifold,
                eval_sampler,
                generator=None,
                field=field,
                device=device,
                dtype=dtype,
                figs_dir=figs_dir,
                step=step,
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

    _write_last_metrics(out_dir, metrics_path)


@torch.no_grad()
def evaluate_rmgm(
    cfg: RealTrainConfig,
    manifold: Any,
    eval_sampler: EmpiricalSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    c_metrics: dict,
    g_metrics: dict,
) -> dict[str, Any]:
    n = min(cfg.eval_samples, len(eval_sampler.data))
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    samples = generator(z)
    data = eval_sampler.sample(n, device, dtype)
    metrics = sample_metrics(cfg.data, manifold, samples, data)
    metrics.update(
        {
            "step": step,
            "method": "rmgm",
            "critic_loss": float(c_metrics["critic_loss"].cpu()),
            "generator_loss": float(g_metrics["generator_loss"].cpu()),
            "target_norm": float(c_metrics["target_norm"].cpu()),
            "sample_norm": float(g_metrics["sample_norm"].cpu()),
            "constraint_error": float(
                constraint_error(manifold, samples, reduction="mean").cpu()
            ),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_rfm(
    cfg: RealTrainConfig,
    manifold: Any,
    eval_sampler: EmpiricalSampler,
    base_sampler: BaseSampler,
    field: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    loss: torch.Tensor,
    x_t: torch.Tensor,
    u_t: torch.Tensor,
    pred: torch.Tensor,
) -> dict[str, Any]:
    n = min(cfg.eval_samples, len(eval_sampler.data))
    samples = sample_rfm(cfg, manifold, base_sampler, field, n, device, dtype)
    data = eval_sampler.sample(n, device, dtype)
    diff = pred - u_t
    metrics = sample_metrics(cfg.data, manifold, samples, data)
    metrics.update(
        {
            "step": step,
            "method": "rfm",
            "rfm_loss": float(loss.cpu()),
            "target_norm": float(metric_inner(manifold, x_t, u_t, u_t).mean().cpu()),
            "pred_norm": float(metric_inner(manifold, x_t, pred, pred).mean().cpu()),
            "train_residual": float(metric_inner(manifold, x_t, diff, diff).mean().cpu()),
            "constraint_error": float(
                constraint_error(manifold, samples, reduction="mean").cpu()
            ),
            "ode_steps": cfg.ode_steps,
        }
    )
    return metrics


@torch.no_grad()
def sample_metrics(
    name: str, manifold: Any, samples: torch.Tensor, data: torch.Tensor
) -> dict[str, float]:
    del name
    embed_samples = embed_real_for_metrics(manifold, samples)
    embed_data = embed_real_for_metrics(manifold, data)
    mmd2 = rbf_mmd2(embed_samples, embed_data).item()
    nn_s2d, nn_d2s = symmetric_nearest_distances(manifold, samples, data)
    data_radius = intrinsic_reference_radius(manifold, data)
    coverage = (nn_d2s <= data_radius).float().mean().item()
    return {
        "mmd2": float(mmd2),
        "nn_sample_to_data": float(nn_s2d.mean().item()),
        "nn_data_to_sample": float(nn_d2s.mean().item()),
        "symmetric_nn": float(0.5 * (nn_s2d.mean() + nn_d2s.mean()).item()),
        "coverage_radius": float(data_radius),
        "coverage_at_radius": float(coverage),
    }


def embed_real_for_metrics(manifold: Any, x: torch.Tensor) -> torch.Tensor:
    if isinstance(manifold, FlatTorus):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    return x.reshape(x.shape[0], -1)


@torch.no_grad()
def symmetric_nearest_distances(
    manifold: Any, samples: torch.Tensor, data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    samples = samples.detach()
    data = data.detach()
    s2d = _min_pairwise_dist(manifold, samples, data)
    d2s = _min_pairwise_dist(manifold, data, samples)
    return s2d, d2s


@torch.no_grad()
def intrinsic_reference_radius(manifold: Any, data: torch.Tensor) -> float:
    n = min(data.shape[0], 1024)
    x = data[:n]
    d = _pairwise_dist(manifold, x, x)
    eps = 1e-7
    d = d.masked_fill(d <= eps, float("inf"))
    nn = d.min(dim=1).values
    nn = nn[torch.isfinite(nn)]
    if nn.numel() == 0:
        return eps
    return float(nn.median().item())


@torch.no_grad()
def _min_pairwise_dist(
    manifold: Any, x: torch.Tensor, y: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    mins = []
    for start in range(0, x.shape[0], chunk):
        d = _pairwise_dist(manifold, x[start : start + chunk], y)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0)


@torch.no_grad()
def _pairwise_dist(manifold: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if isinstance(manifold, Sphere):
        dots = x @ y.T
        return dots.clamp(-1.0, 1.0).acos()
    if isinstance(manifold, FlatTorus):
        diff = circular_difference(x[:, None, :], y[None, :, :])
        return diff.pow(2).sum(dim=-1).sqrt()
    return torch.cdist(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))


@torch.no_grad()
def save_real_plots(
    cfg: RealTrainConfig,
    manifold: Any,
    eval_sampler: EmpiricalSampler,
    *,
    generator: torch.nn.Module | None,
    field: torch.nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
    figs_dir: Path,
    step: int,
) -> None:
    n = min(cfg.eval_samples, len(eval_sampler.data), 8000)
    data = eval_sampler.sample(n, device, dtype)
    if cfg.method == "rmgm":
        assert generator is not None
        z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
        samples = generator(z)
    else:
        assert field is not None
        base_sampler = BaseSampler(manifold, eval_sampler.data.shape[-1])
        samples = sample_rfm(cfg, manifold, base_sampler, field, n, device, dtype)

    data_cpu = data.detach().cpu()
    samples_cpu = samples.detach().cpu()
    if isinstance(manifold, Sphere) and data_cpu.shape[-1] == 3:
        plot_earth(data_cpu, samples_cpu, figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png")
    elif isinstance(manifold, FlatTorus):
        path = figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png"
        if data_cpu.shape[-1] > 2:
            plot_torus_pairs(data_cpu, samples_cpu, path, title=f"{cfg.data} / {cfg.method}")
        else:
            plot_torus_real(data_cpu, samples_cpu, path, title=f"{cfg.data} / {cfg.method}")
    elif isinstance(manifold, SPD):
        plot_spd_real(data_cpu, samples_cpu, figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png")
    elif manifold.__class__.__name__ == "PoincareBall":
        plot_poincare_real(data_cpu, samples_cpu, figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png")
    elif manifold.__class__.__name__ == "Mesh":
        plot_mesh_real(
            data_cpu,
            samples_cpu,
            manifold,
            figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png",
        )
    else:
        plot_ambient(data_cpu, samples_cpu, figs_dir / f"{cfg.data}-{cfg.method}-{step:06d}.png")


def plot_earth(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    data_ll = lonlat_from_cartesian(data) / math.pi * 180.0
    samples_ll = lonlat_from_cartesian(samples) / math.pi * 180.0
    fig, ax = plt.subplots(figsize=(9.0, 4.25))
    plotted_map = False
    try:
        import geopandas as gpd

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, antialiased=False, color="#d6d6d6", edgecolor="#f3f3f3", linewidth=0.2)
        plotted_map = True
    except Exception:
        ax.set_facecolor("#f2f2f2")

    ax.scatter(
        data_ll[:, 0].numpy(),
        data_ll[:, 1].numpy(),
        s=1.3,
        c="#b33a3a",
        alpha=0.36,
        linewidths=0,
        label="test data",
    )
    ax.scatter(
        samples_ll[:, 0].numpy(),
        samples_ll[:, 1].numpy(),
        s=1.3,
        c="#1a8f58",
        alpha=0.34,
        linewidths=0,
        label="generated",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(axis="both", which="both", direction="in", length=3)
    ax.legend(loc="lower left", frameon=False, markerscale=4)
    if not plotted_map:
        ax.grid(color="white", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_torus_real(data: torch.Tensor, samples: torch.Tensor, path: Path, title: str) -> None:
    def deg(x: torch.Tensor) -> np.ndarray:
        return (((x + math.pi) % (2.0 * math.pi) - math.pi) / math.pi * 180.0).numpy()

    data2 = deg(data[..., :2])
    samples2 = deg(samples[..., :2])
    fig, ax = plt.subplots(figsize=(6.1, 5.2))
    ax.scatter(data2[:, 0], data2[:, 1], marker=".", c="#b33a3a", s=1.0, alpha=0.32)
    ax.scatter(samples2[:, 0], samples2[:, 1], marker=".", c="#3569b8", s=1.0, alpha=0.36)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.55, linewidth=0.5)
    ax.axvline(x=0.0, color="black", linestyle="--", alpha=0.55, linewidth=0.5)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$", rotation=0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_torus_pairs(data: torch.Tensor, samples: torch.Tensor, path: Path, title: str) -> None:
    def deg(x: torch.Tensor) -> np.ndarray:
        return (((x + math.pi) % (2.0 * math.pi) - math.pi) / math.pi * 180.0).numpy()

    dim = data.shape[-1]
    candidate_pairs = [(0, 1), (0, 2), (1, 2), (max(0, dim - 2), dim - 1)]
    pairs = []
    for pair in candidate_pairs:
        if pair[0] < dim and pair[1] < dim and pair[0] != pair[1] and pair not in pairs:
            pairs.append(pair)
    data_deg = deg(data)
    samples_deg = deg(samples)
    ncols = min(2, len(pairs))
    nrows = int(math.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), squeeze=False)
    for ax, (i, j) in zip(axes.reshape(-1), pairs):
        ax.scatter(data_deg[:, i], data_deg[:, j], marker=".", c="#b33a3a", s=1.0, alpha=0.30)
        ax.scatter(samples_deg[:, i], samples_deg[:, j], marker=".", c="#3569b8", s=1.0, alpha=0.34)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.40, linewidth=0.45)
        ax.axvline(x=0.0, color="black", linestyle="--", alpha=0.40, linewidth=0.45)
        ax.set_xlabel(rf"$\theta_{i}$")
        ax.set_ylabel(rf"$\theta_{j}$", rotation=0)
    for ax in axes.reshape(-1)[len(pairs) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_spd_real(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    manifold = SPD()
    ax = plot_cone()

    def cone_coords(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mats = manifold.devectorize(x)[..., :2, :2]
        vec = manifold.vectorize(mats).cpu().numpy()
        c = vec[:, 1]
        u = 0.5 * (vec[:, 0] + vec[:, 2])
        v = 0.5 * (vec[:, 0] - vec[:, 2])
        return c, v, u

    c, v, u = cone_coords(data)
    ax.scatter(c, v, u, marker=".", c="#b33a3a", s=3, alpha=0.35, label="test data")
    c, v, u = cone_coords(samples)
    ax.scatter(c, v, u, marker=".", c="#3569b8", s=3, alpha=0.45, label="generated")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_poincare_real(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=0.8)
    ax.add_patch(circle)
    ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=2, c="#b33a3a", alpha=0.35, label="test data")
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, c="#3569b8", alpha=0.45, label="generated")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_mesh_real(data: torch.Tensor, samples: torch.Tensor, manifold: Any, path: Path) -> None:
    v = manifold.v.detach().cpu().numpy()
    f = manifold.f.detach().cpu().numpy()
    data_np = data.numpy()
    samples_np = samples.numpy()
    if max(abs(v[:, 2].max()), abs(v[:, 2].min())) < 1e-6:
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        for tri in f:
            pts = v[tri]
            ax.plot(
                [pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0]],
                [pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1]],
                color="#c8c8c8",
                linewidth=0.15,
                alpha=0.45,
            )
        ax.scatter(data_np[:, 0], data_np[:, 1], s=3, c="#b33a3a", alpha=0.35, label="test data")
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=3, c="#3569b8", alpha=0.55, label="generated")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="lower left", frameon=False, markerscale=3)
    else:
        fig = plt.figure(figsize=(6.2, 5.8))
        ax = fig.add_subplot(projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#d8d8d8", linewidth=0.05, alpha=0.20)
        ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], s=3, c="#b33a3a", alpha=0.30, label="test data")
        ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], s=3, c="#3569b8", alpha=0.45, label="generated")
        ax.set_axis_off()
        ax.legend(frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_ambient(data: torch.Tensor, samples: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    data_np = data.reshape(data.shape[0], -1).numpy()
    samples_np = samples.reshape(samples.shape[0], -1).numpy()
    ax.scatter(data_np[:, 0], data_np[:, 1], s=2, c="black", alpha=0.25, label="data")
    ax.scatter(samples_np[:, 0], samples_np[:, 1], s=2, c="#3569b8", alpha=0.45, label="generated")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


@torch.no_grad()
def maybe_init_torus_mixture_real(
    cfg: RealTrainConfig,
    manifold: Any,
    data_sampler: EmpiricalSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not cfg.torus_kmeans_init or not isinstance(manifold, FlatTorus):
        return
    if not hasattr(generator, "centers") or not hasattr(generator, "log_scale"):
        return

    n = min(cfg.torus_kmeans_samples, len(data_sampler.data))
    data = data_sampler.sample(n, device, dtype)
    centers = circular_kmeans_real(
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
        if cluster_diff.shape[0] <= 1:
            scales[k].fill_(cfg.torus_residual_scale)
        else:
            scales[k] = cluster_diff.std(dim=0, unbiased=False).clamp(0.03, 1.00)

    generator.centers.copy_(centers.to(generator.centers))
    generator.log_scale.copy_(inverse_softplus(scales.to(generator.log_scale)))
    if hasattr(generator, "component_logits"):
        counts = torch.bincount(assignment, minlength=centers.shape[0]).to(scales)
        probs = (counts + 1.0) / (counts.sum() + centers.shape[0])
        generator.component_logits.copy_(probs.log().to(generator.component_logits))


@torch.no_grad()
def maybe_init_local_gmm_real(
    cfg: RealTrainConfig,
    manifold: Any,
    data_sampler: EmpiricalSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not cfg.generator_kmeans_init or isinstance(manifold, FlatTorus):
        return
    if not hasattr(generator, "centers") or not hasattr(generator, "log_scale"):
        return

    n = min(cfg.generator_kmeans_samples, len(data_sampler.data))
    data = data_sampler.sample(n, device, dtype)
    if hasattr(generator, "initialize_from_mesh_data"):
        generator.initialize_from_mesh_data(data, iters=cfg.generator_kmeans_iters)
        return
    init = (
        generator.projected_centers()
        if hasattr(generator, "projected_centers")
        else generator.centers
    )
    centers = euclidean_kmeans_real(
        data.reshape(data.shape[0], -1),
        generator.centers.shape[0],
        iters=cfg.generator_kmeans_iters,
        init=init.detach().reshape(generator.centers.shape[0], -1),
    ).reshape_as(generator.centers)
    centers = manifold.projx(centers)

    diff = (data[:, None, :] - centers[None, :, :]).reshape(
        data.shape[0], centers.shape[0], -1
    )
    assignment = diff.pow(2).sum(dim=-1).argmin(dim=1)
    scales = torch.empty_like(centers)
    for k in range(centers.shape[0]):
        points = data[assignment == k]
        if points.shape[0] <= 1:
            scales[k].fill_(cfg.generator_residual_scale)
            continue
        center = centers[k].expand(points.shape[0], -1)
        try:
            tangent = manifold.logmap(center, points)
        except NotImplementedError:
            tangent = manifold.proju(center, points - center)
        scales[k] = tangent.std(dim=0, unbiased=False).abs().clamp(0.03, 1.00)

    generator.centers.copy_(centers.to(generator.centers))
    generator.log_scale.copy_(inverse_softplus(scales.to(generator.log_scale)))
    if hasattr(generator, "component_logits"):
        counts = torch.bincount(assignment, minlength=centers.shape[0]).to(scales)
        probs = (counts + 1.0) / (counts.sum() + centers.shape[0])
        generator.component_logits.copy_(probs.log().to(generator.component_logits))


@torch.no_grad()
def euclidean_kmeans_real(
    x: torch.Tensor,
    components: int,
    *,
    iters: int,
    init: torch.Tensor,
) -> torch.Tensor:
    centers = init[:components].clone()
    if x.shape[0] < components:
        repeats = math.ceil(components / max(x.shape[0], 1))
        centers = x.repeat(repeats, 1)[:components].clone()
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
def circular_kmeans_real(
    x: torch.Tensor,
    components: int,
    *,
    iters: int,
    init: torch.Tensor,
) -> torch.Tensor:
    centers = init[:components].clone()
    if x.shape[0] < components:
        repeats = math.ceil(components / max(x.shape[0], 1))
        centers = x.repeat(repeats, 1)[:components].clone()
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


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def load_real_dataset(cfg: RealTrainConfig) -> dict[str, Any]:
    wrapped_dim, wrapped_scale = wrapped_defaults(cfg)
    data_cfg = DotConfig(
        data=cfg.data,
        seed=cfg.seed,
        data_seed=cfg.data_seed,
        earth_datadir=cfg.earth_datadir,
        top500_datadir=cfg.top500_datadir,
        rna_datadir=cfg.rna_datadir,
        mesh_datadir=cfg.mesh_datadir,
        wrapped=DotConfig(
            dim=wrapped_dim,
            n_mixtures=cfg.wrapped_n_mixtures,
            scale=wrapped_scale,
        ),
    )
    dataset, _ = _get_dataset(data_cfg)
    manifold_attr = getattr(dataset, "manifold", None)
    if inspect.ismethod(manifold_attr) or inspect.isfunction(manifold_attr):
        mesh_numeigs, mesh_metric = mesh_defaults(cfg)
        manifold = manifold_attr(
            numeigs=mesh_numeigs,
            metric=Metric(mesh_metric),
            temp=cfg.mesh_temp,
        )
        mesh_note = {
            "mesh_numeigs": mesh_numeigs,
            "mesh_metric": mesh_metric,
            "mesh_temp": cfg.mesh_temp,
            "mesh_mgm_path": "local_projected_surrogate",
        }
    else:
        manifold = manifold_attr
        mesh_note = None
    dim = int(dataset.dim)
    train_set, val_set, test_set = split_dataset(dataset, cfg)
    train = stack_dataset(train_set)
    val = stack_dataset(val_set)
    test = stack_dataset(test_set)
    return {
        "manifold": manifold,
        "dim": dim,
        "train": train,
        "val": val,
        "test": test,
        "sizes": {
            "total": len(dataset),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "mesh_note": mesh_note,
        "wrapped": {
            "dim": wrapped_dim,
            "n_mixtures": cfg.wrapped_n_mixtures,
            "scale": wrapped_scale,
        },
    }


def wrapped_defaults(cfg: RealTrainConfig) -> tuple[int, float]:
    if cfg.data == "wrapped_torus":
        return cfg.wrapped_dim or 5, cfg.wrapped_scale or 0.2
    if cfg.data == "wrapped_spd":
        return cfg.wrapped_dim or 3, cfg.wrapped_scale or 0.4
    return cfg.wrapped_dim or 2, cfg.wrapped_scale or 0.2


def mesh_defaults(cfg: RealTrainConfig) -> tuple[int, str]:
    if cfg.mesh_numeigs is not None:
        numeigs = cfg.mesh_numeigs
    elif cfg.data == "maze4v2":
        numeigs = 30
    elif cfg.data == "simple_bunny":
        numeigs = 100
    else:
        numeigs = 200
    metric = cfg.mesh_metric or "biharmonic"
    return int(numeigs), metric


def split_dataset(dataset: Any, cfg: RealTrainConfig) -> Tuple[Subset, Subset, Subset]:
    n = len(dataset)
    n_val = n_test = n // 10
    n_train = n - n_val - n_test
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    return torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )


def stack_dataset(dataset: Iterable[Any]) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=8192, shuffle=False)
    chunks = []
    for batch in loader:
        if isinstance(batch, dict):
            batch = batch["x1"]
        chunks.append(batch.detach().float().cpu())
    if not chunks:
        raise ValueError("Cannot stack an empty dataset split.")
    return torch.cat(chunks, dim=0)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _write_last_metrics(out_dir: Path, metrics_path: Path) -> None:
    metrics = {}
    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            metrics = json.loads(lines[-1])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> RealTrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", default="rmgm", choices=["rmgm", "rfm"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-c", type=float, default=2e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="../../outputs/rmgm_real")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--metric-normalize", action="store_true")
    parser.add_argument("--earth-datadir", default="data/earth_data")
    parser.add_argument("--top500-datadir", default="data/top500")
    parser.add_argument("--rna-datadir", default="data/rna")
    parser.add_argument("--mesh-datadir", default="data/mesh")
    parser.add_argument("--wrapped-dim", type=int)
    parser.add_argument("--wrapped-n-mixtures", type=int, default=1)
    parser.add_argument("--wrapped-scale", type=float)
    parser.add_argument("--mesh-numeigs", type=int)
    parser.add_argument("--mesh-metric")
    parser.add_argument("--mesh-temp", type=float, default=0.25)
    parser.add_argument(
        "--generator-type",
        default="mlp",
        choices=["mlp", "local_gmm", "sphere_map", "sphere_map_gmm"],
    )
    parser.add_argument("--generator-components", type=int, default=4)
    parser.add_argument("--generator-residual-scale", type=float, default=0.25)
    parser.add_argument("--generator-kmeans-init", action="store_true")
    parser.add_argument("--generator-kmeans-samples", type=int, default=8192)
    parser.add_argument("--generator-kmeans-iters", type=int, default=40)
    parser.add_argument("--sphere-map-temperature", type=float, default=80.0)
    parser.add_argument("--sphere-map-vertex-stride", type=int, default=1)
    parser.add_argument("--torus-map", default="mod", choices=["mod", "sigmoid"])
    parser.add_argument(
        "--torus-generator",
        default="mlp",
        choices=["mlp", "mixture", "gmm", "gaussian", "gaussian_mixture"],
    )
    parser.add_argument("--torus-components", type=int, default=8)
    parser.add_argument("--torus-residual-scale", type=float, default=0.35)
    parser.add_argument("--torus-mixture-init", default="grid", choices=["grid", "random"])
    parser.add_argument("--torus-kmeans-init", action="store_true")
    parser.add_argument("--torus-kmeans-samples", type=int, default=8192)
    parser.add_argument("--torus-kmeans-iters", type=int, default=40)
    return RealTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
