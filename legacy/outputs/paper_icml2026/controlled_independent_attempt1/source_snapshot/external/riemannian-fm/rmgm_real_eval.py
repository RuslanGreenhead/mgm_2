"""Real-data sample evaluation for R-MGM and RFM checkpoints.

This script deliberately avoids training. It reuses the RFM dataset/manifold
loading path, then evaluates generated samples with metrics that do not require
an explicit likelihood: manifold MMD, nearest-neighbor coverage/precision,
two-sample kNN accuracy, and pairwise geodesic energy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from manifm.datasets import get_loaders, get_manifold
from manifm.eval_utils import load_model
from manifm.manifolds import FlatTorus, Sphere, SPD
from manifm.rmgm import build_critic, build_generator, constraint_error
from manifm.utils import lonlat_from_cartesian
from rfm_baseline_train import BaseSampler, sample_rfm


@dataclass
class EvalResult:
    method: str
    metrics: dict[str, float | int | str | None]
    samples_path: str | None


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    dtype = getattr(torch, args.dtype)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_eval_cfg(args)
    _, _, test_loader = get_loaders(cfg)
    manifold, dim = get_manifold(cfg)
    heldout = collect_loader_tensor(test_loader, args.num_samples, device, dtype)
    heldout = manifold.projx(heldout)

    results: list[EvalResult] = []
    if args.rmgm_checkpoint:
        generator, ckpt_step = load_rmgm_generator(
            args.rmgm_checkpoint, manifold, dim, device, dtype
        )
        samples = sample_rmgm(generator, args.num_samples, args.latent_dim, device, dtype)
        results.append(
            evaluate_samples(
                "rmgm",
                manifold,
                dim,
                heldout,
                samples,
                out_dir,
                cfg.data,
                ckpt_step=ckpt_step,
                make_plots=args.plots,
            )
        )

    if args.rfm_checkpoint:
        rfm_samples, rfm_metrics = sample_lightning_rfm(
            args.rfm_checkpoint,
            heldout,
            args.num_samples,
            device,
            args.rfm_nll,
            args.nll_batch_size,
        )
        result = evaluate_samples(
            "rfm",
            manifold,
            dim,
            heldout,
            rfm_samples.to(device=device, dtype=dtype),
            out_dir,
            cfg.data,
            ckpt_step=None,
            make_plots=args.plots,
        )
        result.metrics.update(rfm_metrics)
        results.append(result)

    if args.rfm_baseline_checkpoint:
        samples, step = sample_rfm_baseline_checkpoint(
            args.rfm_baseline_checkpoint,
            manifold,
            dim,
            args.num_samples,
            device,
            dtype,
        )
        results.append(
            evaluate_samples(
                "rfm_baseline",
                manifold,
                dim,
                heldout,
                samples,
                out_dir,
                cfg.data,
                ckpt_step=step,
                make_plots=args.plots,
            )
        )

    if not results:
        raise SystemExit(
            "Provide at least one of --rmgm-checkpoint, --rfm-checkpoint, "
            "or --rfm-baseline-checkpoint."
        )

    rows = [dict(method=row.method, **row.metrics, samples_plot=row.samples_path) for row in results]
    metrics_path = out_dir / "real_eval_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote metrics to {metrics_path}")


def load_eval_cfg(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.config)
    if args.experiment_config:
        exp_cfg = OmegaConf.load(args.experiment_config)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    if args.data:
        cfg.data = args.data
    cfg.optim.val_batch_size = args.num_samples
    cfg.optim.batch_size = min(cfg.optim.batch_size, args.num_samples)
    cfg.use_wandb = False
    cfg.visualize = False
    if args.earth_datadir:
        cfg.earth_datadir = str(args.earth_datadir)
    if args.rna_datadir:
        cfg.rna_datadir = str(args.rna_datadir)
    if args.top500_datadir:
        cfg.top500_datadir = str(args.top500_datadir)
    if args.mesh_datadir:
        cfg.mesh_datadir = str(args.mesh_datadir)
    return cfg


def collect_loader_tensor(
    loader: torch.utils.data.DataLoader,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunks = []
    remaining = n
    for batch in loader:
        x = batch["x1"] if isinstance(batch, dict) else batch
        x = x[:remaining].to(device=device, dtype=dtype)
        chunks.append(x)
        remaining -= x.shape[0]
        if remaining <= 0:
            break
    if not chunks:
        raise ValueError("The test loader did not yield any tensors.")
    return torch.cat(chunks, dim=0)


def load_rmgm_generator(
    checkpoint: Path,
    manifold: Any,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, int | None]:
    ckpt = torch_load_cpu(checkpoint)
    cfg = ckpt.get("config", {})
    generator_cfg = {
        "generator": {
            "latent_dim": cfg_value(cfg, "latent_dim", dim),
            "hidden_dim": cfg_value(cfg, "hidden_dim", 256),
            "num_layers": cfg_value(cfg, "num_layers", 3),
            "activation": cfg_value(cfg, "activation", "silu"),
            "generator_type": cfg_value(cfg, "generator_type", "mlp"),
            "generator_components": cfg_value(cfg, "generator_components", 4),
            "generator_residual_scale": cfg_value(cfg, "generator_residual_scale", 0.25),
            "torus_map": cfg_value(cfg, "torus_map", "mod"),
            "torus_generator": cfg_value(cfg, "torus_generator", "mlp"),
            "torus_components": cfg_value(cfg, "torus_components", 4),
            "torus_residual_scale": cfg_value(cfg, "torus_residual_scale", 0.35),
            "torus_mixture_init": cfg_value(cfg, "torus_mixture_init", "grid"),
        }
    }
    generator = build_generator(generator_cfg, manifold, dim).to(device=device, dtype=dtype)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    return generator, ckpt.get("step")


@torch.no_grad()
def sample_rmgm(
    generator: torch.nn.Module,
    n: int,
    latent_dim_override: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_dim = latent_dim_override or int(getattr(generator, "latent_dim"))
    z = torch.randn(n, latent_dim, device=device, dtype=dtype)
    return generator(z)


@torch.no_grad()
def sample_lightning_rfm(
    checkpoint: Path,
    heldout: torch.Tensor,
    n: int,
    device: torch.device,
    compute_nll: bool,
    nll_batch_size: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    cfg, model = load_model(str(checkpoint))
    model = model.to(device)
    model.eval()
    samples = model.sample(n, device=device)
    metrics: dict[str, float | int] = {}
    if compute_nll:
        nll_values = []
        for batch in torch.split(heldout, nll_batch_size):
            logp = model.compute_exact_loglikelihood(batch)
            nll_values.append((-logp).detach())
        nll = torch.cat(nll_values)
        metrics["heldout_nll_mean"] = float(nll.mean().cpu())
        metrics["heldout_nll_std"] = float(nll.std(unbiased=False).cpu())
    metrics["rfm_eval_projx"] = bool(cfg.get("eval_projx", False))
    return samples, metrics


def sample_rfm_baseline_checkpoint(
    checkpoint: Path,
    manifold: Any,
    dim: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int | None]:
    ckpt = torch_load_cpu(checkpoint)
    cfg = ckpt.get("config", {})
    field_cfg = {
        "critic": {
            "hidden_dim": cfg_value(cfg, "hidden_dim", 256),
            "num_layers": cfg_value(cfg, "num_layers", 4),
            "activation": "silu",
            "fourier": 8,
            "metric_normalize": cfg_value(cfg, "metric_normalize", False),
        }
    }
    eval_cfg = argparse.Namespace(
        ode_steps=int(cfg_value(cfg, "ode_steps", 128)),
        hyperbolic_base_scale=float(cfg_value(cfg, "hyperbolic_base_scale", 0.45)),
    )
    field = build_critic(field_cfg, manifold, dim).to(device=device, dtype=dtype)
    field.load_state_dict(ckpt["field"])
    field.eval()
    base_sampler = BaseSampler(
        manifold,
        dim,
        hyperbolic_scale=float(cfg_value(cfg, "hyperbolic_base_scale", 0.45)),
    )
    samples = sample_rfm(eval_cfg, manifold, base_sampler, field, n, device, dtype)
    return samples, ckpt.get("step")


def evaluate_samples(
    method: str,
    manifold: Any,
    dim: int,
    heldout: torch.Tensor,
    samples: torch.Tensor,
    out_dir: Path,
    data_name: str,
    *,
    ckpt_step: int | None,
    make_plots: bool,
) -> EvalResult:
    n = min(heldout.shape[0], samples.shape[0])
    heldout = heldout[:n]
    samples = manifold.projx(samples[:n].to(heldout))

    dist_xs = pairwise_distances(manifold, heldout, samples)
    dist_xx = pairwise_distances(manifold, heldout, heldout)
    dist_ss = pairwise_distances(manifold, samples, samples)
    emb_x = embed_for_mmd(manifold, heldout)
    emb_s = embed_for_mmd(manifold, samples)

    data_nn_scale = kth_nonself_distance(dist_xx, k=2).median().clamp_min(1e-8)
    data_to_sample = dist_xs.min(dim=1).values
    sample_to_data = dist_xs.min(dim=0).values

    metrics: dict[str, float | int | str | None] = {
        "checkpoint_step": ckpt_step,
        "num_samples": n,
        "mmd2": float(rbf_mmd2(emb_s, emb_x).cpu()),
        "coverage_nn_mean": float(data_to_sample.mean().cpu()),
        "coverage_nn_median": float(data_to_sample.median().cpu()),
        "coverage_nn_q90": float(torch.quantile(data_to_sample, 0.90).cpu()),
        "coverage_frac_within_train_scale": float((data_to_sample <= data_nn_scale).float().mean().cpu()),
        "precision_nn_mean": float(sample_to_data.mean().cpu()),
        "precision_nn_median": float(sample_to_data.median().cpu()),
        "precision_nn_q90": float(torch.quantile(sample_to_data, 0.90).cpu()),
        "precision_frac_within_train_scale": float((sample_to_data <= data_nn_scale).float().mean().cpu()),
        "data_pairwise_energy": float(nonself_mean_square(dist_xx).cpu()),
        "sample_pairwise_energy": float(nonself_mean_square(dist_ss).cpu()),
        "energy_ratio_sample_over_data": float(
            (nonself_mean_square(dist_ss) / nonself_mean_square(dist_xx).clamp_min(1e-8)).cpu()
        ),
        "knn_two_sample_acc_k1": float(two_sample_knn_accuracy(dist_xx, dist_ss, dist_xs, k=1).cpu()),
        "knn_two_sample_acc_k5": float(two_sample_knn_accuracy(dist_xx, dist_ss, dist_xs, k=5).cpu()),
        "constraint_error_mean": float(constraint_error(manifold, samples, reduction="mean").cpu()),
        "constraint_error_max": float(constraint_error(manifold, samples, reduction="max").cpu()),
    }

    samples_path = None
    if make_plots:
        plot_path = out_dir / f"{data_name}-{method}-samples.png"
        plot_samples(manifold, heldout.detach().cpu(), samples.detach().cpu(), method, plot_path)
        samples_path = str(plot_path)
    return EvalResult(method=method, metrics=metrics, samples_path=samples_path)


def pairwise_distances(
    manifold: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    rows = []
    for x_chunk in torch.split(x, chunk_size):
        rows.append(_pairwise_distances_chunk(manifold, x_chunk, y))
    return torch.cat(rows, dim=0)


def _pairwise_distances_chunk(manifold: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if isinstance(manifold, FlatTorus):
        diff = torch.atan2(
            torch.sin(x[:, None, :] - y[None, :, :]),
            torch.cos(x[:, None, :] - y[None, :, :]),
        )
        return diff.norm(dim=-1)
    if isinstance(manifold, Sphere):
        dots = (x[:, None, :] * y[None, :, :]).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(dots)
    if isinstance(manifold, SPD):
        x_rep = x[:, None, :].expand(-1, y.shape[0], -1).reshape(-1, x.shape[-1])
        y_rep = y[None, :, :].expand(x.shape[0], -1, -1).reshape(-1, y.shape[-1])
        tangent = manifold.logmap(x_rep, y_rep)
        d2 = manifold.inner(x_rep, tangent, tangent).clamp_min(0.0)
        return d2.sqrt().reshape(x.shape[0], y.shape[0])
    if hasattr(manifold, "dist"):
        x_rep = x[:, None, :].expand(-1, y.shape[0], -1)
        y_rep = y[None, :, :].expand(x.shape[0], -1, -1)
        return manifold.dist(x_rep, y_rep)
    return torch.cdist(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))


def embed_for_mmd(manifold: Any, x: torch.Tensor) -> torch.Tensor:
    if isinstance(manifold, FlatTorus):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    if isinstance(manifold, SPD):
        A = manifold.devectorize(x)
        return A.reshape(A.shape[0], -1)
    return x.reshape(x.shape[0], -1)


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, max_points: int = 2048) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0], max_points)
    x = x[:n]
    y = y[:n]
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    positive = d2[d2 > 0]
    bandwidth2 = torch.median(positive).clamp_min(1e-6)
    gamma = 1.0 / bandwidth2
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2))
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2))
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def kth_nonself_distance(dist: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, dist.shape[1])
    return torch.topk(dist, k=k, dim=1, largest=False).values[:, -1]


def nonself_mean_square(dist: torch.Tensor) -> torch.Tensor:
    if dist.shape[0] != dist.shape[1]:
        return dist.pow(2).mean()
    mask = ~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
    return dist[mask].pow(2).mean()


def two_sample_knn_accuracy(
    dist_xx: torch.Tensor,
    dist_ss: torch.Tensor,
    dist_xs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    n = dist_xx.shape[0]
    k = min(k, 2 * n - 1)
    top = torch.cat(
        [
            torch.cat([dist_xx, dist_xs], dim=1),
            torch.cat([dist_xs.transpose(0, 1), dist_ss], dim=1),
        ],
        dim=0,
    )
    top.fill_diagonal_(float("inf"))
    labels = torch.cat(
        [
            torch.ones(n, device=top.device, dtype=torch.long),
            torch.zeros(n, device=top.device, dtype=torch.long),
        ]
    )
    neighbor_idx = top.topk(k=k, dim=1, largest=False).indices
    vote_real = labels[neighbor_idx].float().mean(dim=1) >= 0.5
    return (vote_real.long() == labels).float().mean()


def plot_samples(
    manifold: Any,
    heldout: torch.Tensor,
    samples: torch.Tensor,
    method: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    x2, xlabel, ylabel = project_for_plot(manifold, heldout)
    s2, _, _ = project_for_plot(manifold, samples)
    ax.scatter(x2[:, 0], x2[:, 1], s=2.0, c="black", alpha=0.25, label="heldout")
    ax.scatter(s2[:, 0], s2[:, 1], s=2.4, c="#3569b8", alpha=0.50, label=method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(manifold, FlatTorus):
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.45)
        ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def project_for_plot(manifold: Any, x: torch.Tensor) -> tuple[np.ndarray, str, str]:
    if isinstance(manifold, FlatTorus):
        y = ((x[:, :2] + math.pi) % (2.0 * math.pi) - math.pi) / math.pi * 180.0
        return y.numpy(), "angle 1 (deg)", "angle 2 (deg)"
    if isinstance(manifold, Sphere) and x.shape[-1] == 3:
        lonlat = torch.stack([lonlat_from_cartesian(row) for row in x])
        y = lonlat / math.pi * 180.0
        return y.numpy(), "longitude", "latitude"
    emb = embed_for_mmd(manifold, x)
    centered = emb - emb.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    y = centered @ vh[:2].T
    return y[:, :2].numpy(), "PC1", "PC2"


def cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--rmgm-checkpoint", type=Path, default=None)
    parser.add_argument("--rfm-checkpoint", type=Path, default=None)
    parser.add_argument("--rfm-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rmgm_real_eval"))
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--rfm-nll", action="store_true")
    parser.add_argument("--nll-batch-size", type=int, default=512)
    parser.add_argument("--earth-datadir", type=Path, default=Path("data/earth_data"))
    parser.add_argument("--rna-datadir", type=Path, default=Path("data/rna"))
    parser.add_argument("--top500-datadir", type=Path, default=Path("data/top500"))
    parser.add_argument("--mesh-datadir", type=Path, default=Path("data/mesh"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
