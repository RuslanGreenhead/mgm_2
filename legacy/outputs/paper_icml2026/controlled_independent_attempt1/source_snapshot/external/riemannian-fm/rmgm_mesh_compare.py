"""Evaluate and render bunny/spot mesh MGM comparisons.

This script compares the two mesh generator parameterizations used in the
bunny/spot experiments:

    target | projected local-GMM MGM | spherical-map MGM

It loads saved R-MGM checkpoints, resamples generated points on the original
mesh manifold, recomputes sample metrics, and writes high-resolution figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifm.rmgm import build_generator, constraint_error
from rmgm_train_real import EmpiricalSampler, RealTrainConfig, load_real_dataset, sample_metrics


DATASETS = ("bunny10", "bunny50", "bunny100", "spot10", "spot50", "spot100")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-root", type=Path, required=True)
    parser.add_argument("--sphere-map-root", type=Path, required=True)
    parser.add_argument("--sphere-map-label", default="sphere_map_gmm")
    parser.add_argument("--sphere-map-title", default="Sphere-map GMM")
    parser.add_argument("--output-dir", type=Path, default=Path("../../outputs/rmgm_mesh_bunny_spot_summary"))
    parser.add_argument("--mesh-datadir", default="data/mesh")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rendered = []
    entries = []
    for data_name in args.datasets:
        print(f"rendering {data_name}", flush=True)
        projection_ckpt = latest_checkpoint(args.projection_root, data_name)
        sphere_ckpt = latest_checkpoint(args.sphere_map_root, data_name)
        if projection_ckpt is None or sphere_ckpt is None:
            print(f"missing checkpoint for {data_name}; skipping", flush=True)
            continue

        manifold, target = load_target(data_name, args.mesh_datadir, args.samples, device)
        target_cpu = target.detach().cpu()
        method_samples = {"target": target_cpu}
        for label, ckpt_path in [("projection", projection_ckpt), ("sphere_map", sphere_ckpt)]:
            samples = sample_checkpoint(ckpt_path, manifold, args.samples, device)
            samples_cpu = samples.detach().cpu()
            method_samples[label] = samples_cpu
            eval_n = min(args.eval_samples, args.samples, target_cpu.shape[0])
            metrics = sample_metrics(
                data_name,
                manifold,
                samples_cpu[:eval_n],
                target_cpu[:eval_n],
            )
            metrics.update(
                {
                    "data": data_name,
                    "variant": "projection" if label == "projection" else args.sphere_map_label,
                    "checkpoint": str(ckpt_path.resolve()),
                    "step": int(torch.load(ckpt_path, map_location="cpu").get("step", -1)),
                    "constraint_error": float(
                        constraint_error(manifold, samples_cpu[:eval_n], reduction="mean")
                    ),
                }
            )
            rows.append(metrics)

        single_path = args.output_dir / f"{data_name}_mesh_mgm_comparison.png"
        render_dataset(data_name, manifold, method_samples, single_path, args.sphere_map_title)
        rendered.append(single_path)
        entries.append((data_name, manifold, method_samples))

    if entries:
        grid_path = args.output_dir / "bunny_spot_mesh_mgm_grid.png"
        render_grid(entries, grid_path, args.sphere_map_title)
        print(grid_path.resolve(), flush=True)

    metrics_path = args.output_dir / "mesh_mgm_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report_path = args.output_dir / "mesh_mgm_report.md"
    report_path.write_text(markdown_report(rows, rendered), encoding="utf-8")
    print(metrics_path.resolve(), flush=True)
    print(report_path.resolve(), flush=True)


def latest_checkpoint(root: Path, data_name: str) -> Path | None:
    base = root / data_name / "rmgm"
    candidates = sorted(base.glob("seed*/checkpoints/best.pt"))
    if not candidates:
        candidates = sorted(base.glob("seed*/checkpoints/last.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_target(
    data_name: str, mesh_datadir: str, n: int, device: torch.device
) -> tuple[Any, torch.Tensor]:
    del device
    cfg = RealTrainConfig(
        data=data_name,
        method="rmgm",
        mesh_datadir=mesh_datadir,
        mesh_numeigs=100,
        eval_samples=n,
        device="cpu",
    )
    real_data = load_real_dataset(cfg)
    manifold = real_data["manifold"]
    sampler = EmpiricalSampler(real_data["test"], manifold)
    target = sampler.sample(min(n, len(sampler.data)), torch.device("cpu"), torch.float32)
    return manifold, target


@torch.no_grad()
def sample_checkpoint(
    ckpt_path: Path,
    manifold: Any,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    generator_cfg = {
        "generator": {
            "latent_dim": cfg.get("latent_dim", 16),
            "hidden_dim": cfg.get("hidden_dim", 256),
            "num_layers": cfg.get("num_layers", 4),
            "activation": "silu",
            "generator_type": cfg.get("generator_type", "mlp"),
            "generator_components": cfg.get("generator_components", 4),
            "generator_residual_scale": cfg.get("generator_residual_scale", 0.25),
            "sphere_map_temperature": cfg.get("sphere_map_temperature", 80.0),
            "sphere_map_vertex_stride": cfg.get("sphere_map_vertex_stride", 1),
        }
    }
    generator = build_generator(generator_cfg, manifold, 3).to(device=device, dtype=dtype)
    generator.load_state_dict(ckpt["generator"])
    z = torch.randn(n, cfg.get("latent_dim", 16), device=device, dtype=dtype)
    samples = generator(z).detach().cpu()
    if hasattr(manifold, "to"):
        manifold.to(device=torch.device("cpu"), dtype=torch.float32)
    return samples


def render_dataset(
    data_name: str,
    manifold: Any,
    samples: dict[str, torch.Tensor],
    path: Path,
    sphere_map_title: str,
) -> None:
    methods = comparison_methods(sphere_map_title)
    fig = plt.figure(figsize=(14.0, 4.0))
    for col, (key, title, color) in enumerate(methods, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        draw_mesh_samples(ax, manifold, samples[key], color, title)
    fig.suptitle(data_name, fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.01)
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def render_grid(
    entries: list[tuple[str, Any, dict[str, torch.Tensor]]],
    path: Path,
    sphere_map_title: str,
) -> None:
    methods = comparison_methods(sphere_map_title)
    fig = plt.figure(figsize=(13.2, 2.65 * len(entries)))
    grid = fig.add_gridspec(len(entries), len(methods), wspace=0.01, hspace=0.03)
    for row, (data_name, manifold, samples) in enumerate(entries):
        for col, (key, title, color) in enumerate(methods):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            draw_mesh_samples(
                ax,
                manifold,
                samples[key],
                color,
                title if row == 0 else "",
            )
            if col == 0:
                ax.text2D(
                    -0.05,
                    0.50,
                    data_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.97, bottom=0.01)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def draw_mesh_samples(ax: Any, manifold: Any, points: torch.Tensor, color: str, title: str) -> None:
    v = manifold.v.detach().cpu().numpy()
    f = manifold.f.detach().cpu().numpy()
    arr = points.detach().cpu().numpy()
    ax.plot_trisurf(
        v[:, 0],
        v[:, 1],
        v[:, 2],
        triangles=f,
        color="#d7d7d7",
        linewidth=0.03,
        alpha=0.18,
        shade=False,
    )
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=2.0, c=color, alpha=0.42, linewidths=0)
    ax.set_title(title, fontsize=13)
    set_equal_3d(ax, np.concatenate([v, arr], axis=0))
    ax.view_init(elev=18, azim=-62)
    ax.set_axis_off()


def set_equal_3d(ax: Any, arr: np.ndarray) -> None:
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float((hi - lo).max())
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def comparison_methods(sphere_map_title: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("target", "Target", "black"),
        ("projection", "Projected MGM", "#d9782f"),
        ("sphere_map", sphere_map_title, "#3569b8"),
    )


def markdown_report(rows: list[dict[str, Any]], figures: list[Path]) -> str:
    lines = [
        "# Bunny/Spot mesh MGM comparison",
        "",
        "| data | variant | step | MMD2 | symmetric NN | coverage | constraint |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda x: (x["data"], x["variant"])):
        lines.append(
            "| {data} | {variant} | {step} | {mmd2:.6g} | {symmetric_nn:.6g} | "
            "{coverage_at_radius:.3f} | {constraint_error:.3g} |".format(**row)
        )
    lines.extend(["", "## Figures", ""])
    for path in figures:
        lines.append(f"- `{path.resolve()}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
