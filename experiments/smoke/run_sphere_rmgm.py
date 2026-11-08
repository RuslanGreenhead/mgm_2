#!/usr/bin/env python3
"""Run a small, portable R-MGM/RFM sphere smoke test.

This intentionally uses the historical model implementation while avoiding the
CUDA-bound controlled-suite wrapper. It is a functional test, not a benchmark
or a substitute for the Table 2 Fire experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_RFM = ROOT / "external" / "riemannian-fm"
if str(EXTERNAL_RFM) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_RFM))

from manifm.manifolds.sphere import Sphere  # noqa: E402
from manifm.rmgm import (  # noqa: E402
    build_critic,
    build_generator,
    constraint_error,
    critic_loss,
    generator_loss,
    metric_inner,
)
from rfm_baseline_train import BaseSampler, TrainConfig, rfm_path, sample_rfm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/development/sphere_smoke.json")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, help="Override runtime.max_steps.")
    parser.add_argument("--seed", type=int, help="Override runtime.seed.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else ROOT / path
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    if cfg.get("dataset", {}).get("name") != "synthetic_sphere":
        raise ValueError("This runner only accepts the synthetic_sphere smoke configuration.")
    if cfg.get("output", {}).get("overwrite", False):
        raise ValueError("Smoke runs are intentionally non-overwriting.")
    return cfg


def select_device(requested: str, preference: str) -> torch.device:
    mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch installation.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if preference == "mps_then_cpu" and mps_available else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_target(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Three deterministic local modes on S², independent of paper datasets."""
    centers = torch.tensor(
        [[0.85, 0.20, 0.48], [-0.55, 0.72, 0.42], [-0.18, -0.55, -0.82]], dtype=dtype, device=device
    )
    centers = centers / centers.norm(dim=-1, keepdim=True)
    center = centers[torch.randint(0, len(centers), (n,), device=device)]
    noise = torch.randn(n, 3, device=device, dtype=dtype) * 0.20
    tangent_noise = noise - (noise * center).sum(dim=-1, keepdim=True) * center
    values = center + tangent_noise
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def all_finite(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())


def train_rmgm(manifold: Sphere, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, events: list[dict[str, Any]]):
    method, runtime, dim = cfg["method"], cfg["runtime"], 3
    generator = build_generator(
        {"generator": {"generator_type": method["rmgm_generator"], "generator_components": method["rmgm_components"],
                        "latent_dim": method["latent_dim"], "hidden_dim": method["hidden_dim"], "num_layers": method["num_layers"]}},
        manifold, dim,
    ).to(device=device, dtype=dtype)
    critic = build_critic(
        {"critic": {"hidden_dim": method["hidden_dim"], "num_layers": method["num_layers"],
                     "fourier": method["critic_fourier_features"]}}, manifold, dim,
    ).to(device=device, dtype=dtype)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=method["learning_rate"])
    opt_c = torch.optim.AdamW(critic.parameters(), lr=method["learning_rate"])
    started, last = time.perf_counter(), {}
    for step in range(1, runtime["max_steps"] + 1):
        z = torch.randn(runtime["batch_size"], method["latent_dim"], device=device, dtype=dtype)
        opt_c.zero_grad(set_to_none=True)
        loss_c, c_metrics = critic_loss(manifold, generator, critic, z, synthetic_target(runtime["batch_size"], device, dtype))
        loss_c.backward()
        opt_c.step()
        set_requires_grad(critic, False)
        z = torch.randn(runtime["batch_size"], method["latent_dim"], device=device, dtype=dtype)
        opt_g.zero_grad(set_to_none=True)
        loss_g, g_metrics = generator_loss(manifold, generator, critic, z, synthetic_target(runtime["batch_size"], device, dtype))
        loss_g.backward()
        opt_g.step()
        set_requires_grad(critic, True)
        last = {"step": step, "critic_loss": float(loss_c.detach().cpu()), "generator_loss": float(loss_g.detach().cpu()),
                "target_norm": float(c_metrics["target_norm"].cpu()), "critic_prediction_norm": float(c_metrics["pred_norm"].cpu()),
                "generator_path_norm": float(g_metrics["sample_norm"].cpu())}
        if not all(np.isfinite(value) for value in last.values() if isinstance(value, float)):
            raise FloatingPointError(f"Non-finite R-MGM metrics at step {step}: {last}")
        if step == 1 or step == runtime["max_steps"]:
            events.append({"event": "rmgm_step", **last})
    return generator, critic, {"final": last, "seconds": time.perf_counter() - started,
                                "finite_parameters": all_finite(generator) and all_finite(critic),
                                "generator_class": generator.__class__.__name__, "critic_class": critic.__class__.__name__}


def train_rfm(manifold: Sphere, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, events: list[dict[str, Any]]):
    method, runtime, dim = cfg["method"], cfg["runtime"], 3
    field = build_critic(
        {"critic": {"hidden_dim": method["hidden_dim"], "num_layers": method["num_layers"],
                     "fourier": method["critic_fourier_features"]}}, manifold, dim,
    ).to(device=device, dtype=dtype)
    base, optimizer, started, last = BaseSampler(manifold, dim), torch.optim.AdamW(field.parameters(), lr=method["learning_rate"]), time.perf_counter(), {}
    for step in range(1, runtime["max_steps"] + 1):
        x0 = base.sample(runtime["batch_size"], device, dtype)
        x1 = synthetic_target(runtime["batch_size"], device, dtype)
        t = torch.rand(runtime["batch_size"], 1, device=device, dtype=dtype)
        x_t, velocity = rfm_path(manifold, x0, x1, t)
        optimizer.zero_grad(set_to_none=True)
        prediction = field(t, x_t)
        residual = prediction - velocity
        loss = metric_inner(manifold, x_t, residual, residual).mean() / dim
        loss.backward()
        optimizer.step()
        last = {"step": step, "rfm_loss": float(loss.detach().cpu()),
                "target_norm": float(metric_inner(manifold, x_t, velocity, velocity).mean().detach().cpu()),
                "prediction_norm": float(metric_inner(manifold, x_t, prediction, prediction).mean().detach().cpu())}
        if not all(np.isfinite(value) for value in last.values() if isinstance(value, float)):
            raise FloatingPointError(f"Non-finite RFM metrics at step {step}: {last}")
        if step == 1 or step == runtime["max_steps"]:
            events.append({"event": "rfm_step", **last})
    return field, base, {"final": last, "seconds": time.perf_counter() - started,
                         "finite_parameters": all_finite(field), "field_class": field.__class__.__name__}


def plot_samples(target: torch.Tensor, rmgm: torch.Tensor, rfm: torch.Tensor, path: Path) -> None:
    fig = plt.figure(figsize=(12, 4.2))
    for index, (name, values, color) in enumerate((("Synthetic target", target, "black"), ("R-MGM mixture", rmgm, "#3569b8"), ("RFM baseline", rfm, "#be6a3d")), start=1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.scatter(values[:, 0], values[:, 1], values[:, 2], s=5, c=color, alpha=0.55)
        axis.set(title=name, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        axis.set_box_aspect([1, 1, 1])
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sphere_sample_metrics(samples: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Small diagnostic metrics, not the paper evaluation protocol."""
    x, y = samples.detach().cpu().double(), target.detach().cpu().double()
    reference_distances = torch.cdist(y, y)
    positive = reference_distances[reference_distances > 1e-12]
    bandwidth2 = positive.square().median() if positive.numel() else torch.tensor(1.0, dtype=x.dtype)
    mmd2 = (
        torch.exp(-torch.cdist(x, x).square() / bandwidth2).mean()
        + torch.exp(-torch.cdist(y, y).square() / bandwidth2).mean()
        - 2.0 * torch.exp(-torch.cdist(x, y).square() / bandwidth2).mean()
    )
    chord_minus = (x[:, None, :] - y[None, :, :]).norm(dim=-1)
    chord_plus = (x[:, None, :] + y[None, :, :]).norm(dim=-1)
    geodesic = 2.0 * torch.atan2(chord_minus, chord_plus)
    return {
        "mmd2_biased_chordal": float(mmd2),
        "symmetric_nearest_neighbour_geodesic": float((geodesic.min(1).values.mean() + geodesic.min(0).values.mean()) / 2.0),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_output_dir(cfg: dict[str, Any]) -> Path:
    destination, artifacts = (ROOT / cfg["output"]["root"]).resolve(), (ROOT / "artifacts").resolve()
    if artifacts not in destination.parents:
        raise ValueError("Smoke-test output must be inside artifacts/.")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = destination.parent / f"{destination.name}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def run() -> Path:
    args, cfg = parse_args(), None
    cfg = load_config(args.config)
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("--steps must be positive.")
        cfg["runtime"]["max_steps"] = args.steps
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
    device = select_device(args.device, cfg["runtime"]["device_preference"])
    seed_everything(cfg["runtime"]["seed"])
    output, dtype = make_output_dir(cfg), torch.float32
    with (output / "run.log").open("x", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
        print("R-MGM sphere smoke test")
        print(f"output={output}")
        print(f"device={device}")
        manifold, events = Sphere(), []
        rmgm_generator, _critic, rmgm_summary = train_rmgm(manifold, cfg, device, dtype, events)
        rfm_field, rfm_base, rfm_summary = train_rfm(manifold, cfg, device, dtype, events)
        n_eval = cfg["runtime"]["evaluation_samples"]
        with torch.no_grad():
            target = synthetic_target(n_eval, device, dtype)
            rmgm_samples = rmgm_generator(torch.randn(n_eval, cfg["method"]["latent_dim"], device=device, dtype=dtype))
            rfm_cfg = TrainConfig(experiment="sphere", ode_steps=cfg["method"]["rfm_ode_steps"])
            rfm_samples = sample_rfm(rfm_cfg, manifold, rfm_base, rfm_field, n_eval, device, dtype)
        metrics = {"scope": "functional smoke test only; not a paper-quality experiment or Table 2 reproduction",
                   "device": str(device), "seed": cfg["runtime"]["seed"], "steps_per_method": cfg["runtime"]["max_steps"],
                   "rmgm": rmgm_summary, "rfm": rfm_summary,
                   "sample_constraint_error": {"rmgm_max": float(constraint_error(manifold, rmgm_samples, "max").cpu()),
                                               "rmgm_mean": float(constraint_error(manifold, rmgm_samples, "mean").cpu()),
                                               "rfm_max": float(constraint_error(manifold, rfm_samples, "max").cpu()),
                                               "rfm_mean": float(constraint_error(manifold, rfm_samples, "mean").cpu())},
                   "finite_samples": {"target": bool(torch.isfinite(target).all()), "rmgm": bool(torch.isfinite(rmgm_samples).all()), "rfm": bool(torch.isfinite(rfm_samples).all())},
                   "sample_diagnostics": {"rmgm": sphere_sample_metrics(rmgm_samples, target), "rfm": sphere_sample_metrics(rfm_samples, target)},
                   "events": events}
        if not all(metrics["finite_samples"].values()):
            raise FloatingPointError("Generated non-finite sample values.")
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        (output / "environment.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(), "git_commit": git_commit(),
            "historical_rmgm_source": "external/riemannian-fm/manifm/rmgm",
            "historical_rfm_source": "external/riemannian-fm/rfm_baseline_train.py"}, indent=2) + "\n", encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.savez_compressed(output / "samples.npz", target=target.cpu().numpy(), rmgm=rmgm_samples.cpu().numpy(), rfm=rfm_samples.cpu().numpy())
        plot_samples(target.cpu(), rmgm_samples.cpu(), rfm_samples.cpu(), output / "samples.png")
        print(json.dumps({"event": "completed", "metrics": metrics}, indent=2))
    print(f"Smoke test completed: {output}")
    return output


if __name__ == "__main__":
    run()
