"""R-MGM real-data experiments on Earth event datasets on S^2.

This is a small companion to ``rmgm_train.py`` for the RFM paper's
Earth/climate sphere datasets: volcano, earthquake, flood, and fire. It keeps
the synthetic trainer untouched and replaces ``WrappedSampler`` with a
replacement sampler over real latitude/longitude CSV rows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifm.manifolds.sphere import Sphere
from manifm.rmgm import build_critic, build_generator, constraint_error, critic_loss, generator_loss
from manifm.utils import cartesian_from_latlon
from rmgm_train import maybe_init_local_gmm, rbf_mmd2


EARTH_DATASETS = ("volcano", "earthquake", "flood", "fire")


@dataclass
class RealSphereTrainConfig:
    data: str = "volcano"
    earth_datadir: str = "data/earth_data"
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
    data_seed: int | None = None
    log_every: int = 100
    plot_every: int = 1000
    eval_samples: int = 4096
    metric_samples: int = 2048
    device: str = "cuda"
    output_dir: str = "outputs/rmgm_real_sphere"
    dtype: str = "float32"
    metric_normalize: bool = False
    generator_type: str = "local_gmm"
    generator_components: int = 8
    generator_residual_scale: float = 0.25
    generator_kmeans_init: bool = True
    generator_kmeans_samples: int = 8192
    generator_kmeans_iters: int = 40


class RealSphereSampler:
    """Replacement for WrappedSampler that samples empirical S^2 points."""

    manifold = Sphere()
    dim = 3

    def __init__(
        self,
        data_name: str,
        split: str,
        csv_path: Path,
        latlon_deg: torch.Tensor,
        data: torch.Tensor,
    ) -> None:
        self.data_name = data_name
        self.split = split
        self.csv_path = csv_path
        self.latlon_deg = latlon_deg.contiguous().float()
        self.data = data.contiguous().float()

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def sample(
        self, n: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if len(self) == 0:
            raise ValueError(f"{self.data_name}/{self.split} split is empty")
        values = self.data.to(device=device, dtype=dtype)
        idx = torch.randint(len(self), (n,), device=device)
        return values.index_select(0, idx)

    def all(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.data.to(device=device, dtype=dtype)


def make_real_sphere_experiment(
    data_name: str,
    earth_datadir: str | Path,
    *,
    data_seed: int,
) -> Tuple[Sphere, int, RealSphereSampler, RealSphereSampler, Dict[str, Any]]:
    data_name = normalize_data_name(data_name)
    csv_path, latlon_deg, cartesian = load_earth_csv(data_name, earth_datadir)
    train_idx, val_idx, test_idx = split_indices(len(cartesian), data_seed)

    train = RealSphereSampler(
        data_name,
        "train",
        csv_path,
        latlon_deg[train_idx],
        cartesian[train_idx],
    )
    val = RealSphereSampler(
        data_name,
        "val",
        csv_path,
        latlon_deg[val_idx],
        cartesian[val_idx],
    )
    test = RealSphereSampler(
        data_name,
        "test",
        csv_path,
        latlon_deg[test_idx],
        cartesian[test_idx],
    )
    eval_sampler = test if len(test) > 0 else val
    info = dataset_info(data_name, csv_path, latlon_deg, train, val, test)
    return Sphere(), 3, train, eval_sampler, info


def load_earth_csv(
    data_name: str,
    earth_datadir: str | Path,
) -> Tuple[Path, torch.Tensor, torch.Tensor]:
    data_name = normalize_data_name(data_name)
    csv_path = Path(earth_datadir) / f"{data_name}.csv"
    if not csv_path.exists():
        zip_hint = Path(earth_datadir).parent / "data.zip"
        raise FileNotFoundError(
            f"Missing {csv_path}. Expected RFM Earth CSVs under data/earth_data. "
            f"If only {zip_hint} exists, unzip it into data/ first."
        )

    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    normalized_header = [h.strip().lower() for h in header]
    if normalized_header[:2] != ["latitude", "longitude"]:
        raise ValueError(
            f"{csv_path} must start with LATITUDE,LONGITUDE; got {','.join(header)}"
        )

    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least two columns")
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        raise ValueError(f"{csv_path} contains NaN or infinite coordinates")

    lat = arr[:, 0]
    lon = arr[:, 1]
    if (lat < -90.0).any() or (lat > 90.0).any():
        raise ValueError(f"{csv_path} contains latitude outside [-90, 90] degrees")
    if (lon < -180.0).any() or (lon > 180.0).any():
        raise ValueError(f"{csv_path} contains longitude outside [-180, 180] degrees")

    latlon_deg = torch.tensor(arr, dtype=torch.float32)
    cartesian = cartesian_from_latlon(latlon_deg * (math.pi / 180.0))
    cartesian = cartesian / cartesian.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return csv_path, latlon_deg, cartesian.float()


def normalize_data_name(data_name: str) -> str:
    normalized = data_name.lower().strip()
    if normalized not in EARTH_DATASETS:
        raise ValueError(f"Unknown Earth dataset: {data_name}. Choose one of {EARTH_DATASETS}.")
    return normalized


def split_indices(n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n < 3:
        raise ValueError(f"Need at least 3 rows for train/val/test split, got {n}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = n // 10
    n_test = n // 10
    n_train = n - n_val - n_test
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def dataset_info(
    data_name: str,
    csv_path: Path,
    latlon_deg: torch.Tensor,
    train: RealSphereSampler,
    val: RealSphereSampler,
    test: RealSphereSampler,
) -> Dict[str, Any]:
    return {
        "data": data_name,
        "csv_path": str(csv_path),
        "rows": int(latlon_deg.shape[0]),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "lat_min": float(latlon_deg[:, 0].min()),
        "lat_max": float(latlon_deg[:, 0].max()),
        "lon_min": float(latlon_deg[:, 1].min()),
        "lon_max": float(latlon_deg[:, 1].max()),
    }


def run(cfg: RealSphereTrainConfig) -> Path:
    seed_everything(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    manifold, dim, train_sampler, eval_sampler, info = make_real_sphere_experiment(
        cfg.data, cfg.earth_datadir, data_seed=data_seed
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / cfg.data / f"seed{cfg.seed}_{stamp}"
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
    maybe_init_local_gmm(cfg, manifold, train_sampler, generator, device, dtype)

    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr_g, weight_decay=cfg.weight_decay
    )
    opt_c = torch.optim.AdamW(
        critic.parameters(), lr=cfg.lr_c, weight_decay=cfg.weight_decay
    )

    summary = {
        "config": asdict(cfg),
        "dataset": info,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifold": manifold.__class__.__name__,
        "dim": dim,
        "likelihood_note": (
            "NLL is not reported: implicit one-step MGM has no tractable "
            "RFM-style likelihood without adding a separate density estimator."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Writing real-sphere R-MGM run to: {out_dir}")

    metrics_path = out_dir / "metrics.jsonl"
    latest_log: Dict[str, Any] = {}
    best_mmd2 = float("inf")
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

        set_requires_grad(critic, False)
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device, dtype=dtype)
        x1 = train_sampler.sample(cfg.batch_size, device, dtype)
        opt_g.zero_grad(set_to_none=True)
        loss_g, g_metrics = generator_loss(manifold, generator, critic, z, x1)
        loss_g.backward()
        opt_g.step()
        set_requires_grad(critic, True)

        if step == 1 or step % cfg.log_every == 0:
            latest_log = evaluate_real_sphere(
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
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(latest_log) + "\n")
            print(json.dumps(latest_log))
            if latest_log["mmd2"] < best_mmd2:
                best_mmd2 = latest_log["mmd2"]
                save_checkpoint(ckpt_dir / "best.pt", step, generator, critic, cfg, latest_log)

        if step == 1 or step % cfg.plot_every == 0 or step == cfg.steps:
            save_sample_figure(
                cfg,
                eval_sampler,
                generator,
                device,
                dtype,
                figs_dir / f"samples-{step:06d}.png",
            )
            save_checkpoint(ckpt_dir / "last.pt", step, generator, critic, cfg, latest_log)

    if metrics_path.exists():
        lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            latest_log = json.loads(lines[-1])
    (out_dir / "metrics.json").write_text(json.dumps(latest_log, indent=2), encoding="utf-8")
    return out_dir


@torch.no_grad()
def evaluate_real_sphere(
    cfg: RealSphereTrainConfig,
    manifold: Sphere,
    eval_sampler: RealSphereSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    c_metrics: Dict[str, torch.Tensor],
    g_metrics: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    n = min(cfg.eval_samples, 4096)
    metric_n = min(cfg.metric_samples, n)
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    samples = generator(z)
    data = eval_sampler.sample(n, device, dtype)
    mmd2 = rbf_mmd2(samples, data).item()
    geo = geodesic_metrics(samples[:metric_n], data[:metric_n])
    err = constraint_error(manifold, samples, reduction="mean").item()
    return {
        "step": step,
        "critic_loss": float(c_metrics["critic_loss"].cpu()),
        "generator_loss": float(g_metrics["generator_loss"].cpu()),
        "target_norm": float(c_metrics["target_norm"].cpu()),
        "sample_norm": float(g_metrics["sample_norm"].cpu()),
        "mmd2": mmd2,
        "geodesic_nn_sample_to_data": geo["nn_sample_to_data"],
        "geodesic_nn_data_to_sample": geo["nn_data_to_sample"],
        "geodesic_energy": geo["energy"],
        "constraint_error": err,
        "nll": None,
        "nll_note": "Unavailable for implicit one-step MGM without a density estimator.",
    }


@torch.no_grad()
def geodesic_metrics(samples: torch.Tensor, data: torch.Tensor) -> Dict[str, float]:
    samples = samples / samples.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    data = data / data.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    nn_sample_to_data = pairwise_geodesic_min(samples, data).mean()
    nn_data_to_sample = pairwise_geodesic_min(data, samples).mean()
    energy = (
        2.0 * pairwise_geodesic_mean(samples, data)
        - pairwise_geodesic_mean(samples, samples)
        - pairwise_geodesic_mean(data, data)
    )
    return {
        "nn_sample_to_data": float(nn_sample_to_data.cpu()),
        "nn_data_to_sample": float(nn_data_to_sample.cpu()),
        "energy": float(energy.cpu()),
    }


def pairwise_geodesic_min(x: torch.Tensor, y: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    mins = []
    for x_chunk in x.split(chunk):
        cos = (x_chunk @ y.T).clamp(-1.0, 1.0)
        mins.append(torch.acos(cos).min(dim=1).values)
    return torch.cat(mins, dim=0)


def pairwise_geodesic_mean(x: torch.Tensor, y: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    total = x.new_tensor(0.0)
    count = 0
    for x_chunk in x.split(chunk):
        cos = (x_chunk @ y.T).clamp(-1.0, 1.0)
        dist = torch.acos(cos)
        total = total + dist.sum()
        count += dist.numel()
    return total / max(count, 1)


@torch.no_grad()
def save_sample_figure(
    cfg: RealSphereTrainConfig,
    eval_sampler: RealSphereSampler,
    generator: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    path: Path,
) -> None:
    n = min(cfg.eval_samples, 6000)
    z = torch.randn(n, cfg.latent_dim, device=device, dtype=dtype)
    samples = generator(z).detach().cpu()
    data = eval_sampler.sample(n, device, dtype).detach().cpu()
    plot_real_sphere(data, samples, cfg.data, path)


def plot_real_sphere(data: torch.Tensor, samples: torch.Tensor, data_name: str, path: Path) -> None:
    data_lon, data_lat = lonlat_degrees(data)
    sample_lon, sample_lat = lonlat_degrees(samples)

    fig = plt.figure(figsize=(11.0, 5.2))
    ax_map = fig.add_subplot(121)
    ax_map.scatter(data_lon, data_lat, s=2.2, c="black", alpha=0.25, label="data")
    ax_map.scatter(sample_lon, sample_lat, s=2.8, c="#2f855a", alpha=0.45, label="R-MGM")
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.set_title(f"{data_name}: equirectangular")
    ax_map.grid(alpha=0.18, linewidth=0.6)
    ax_map.legend(loc="lower left", frameon=False)

    ax_sphere = fig.add_subplot(122, projection="3d")
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax_sphere.plot_surface(xs, ys, zs, color="#edf5fa", alpha=0.24, linewidth=0)
    d = data.numpy()
    s = samples.numpy()
    ax_sphere.scatter(d[:, 0], d[:, 1], d[:, 2], s=1.5, c="black", alpha=0.22)
    ax_sphere.scatter(s[:, 0], s[:, 1], s[:, 2], s=1.8, c="#2f855a", alpha=0.50)
    ax_sphere.view_init(elev=20, azim=35)
    ax_sphere.set_box_aspect([1, 1, 1])
    ax_sphere.set_axis_off()
    ax_sphere.set_title("unit sphere")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def lonlat_degrees(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lon = torch.atan2(x[:, 1], x[:, 0]) * (180.0 / math.pi)
    lat = torch.asin(x[:, 2].clamp(-1.0, 1.0)) * (180.0 / math.pi)
    return lon.numpy(), lat.numpy()


def save_checkpoint(
    path: Path,
    step: int,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    cfg: RealSphereTrainConfig,
    latest_log: Dict[str, Any],
) -> None:
    torch.save(
        {
            "step": step,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "config": asdict(cfg),
            "latest_log": latest_log,
        },
        path,
    )


def inspect_data(earth_datadir: str | Path, names: Iterable[str] = EARTH_DATASETS) -> None:
    rows = []
    for name in names:
        try:
            csv_path, latlon_deg, cartesian = load_earth_csv(name, earth_datadir)
            rows.append(
                {
                    "data": name,
                    "csv_path": str(csv_path),
                    "rows": int(latlon_deg.shape[0]),
                    "lat_min": float(latlon_deg[:, 0].min()),
                    "lat_max": float(latlon_deg[:, 0].max()),
                    "lon_min": float(latlon_deg[:, 1].min()),
                    "lon_max": float(latlon_deg[:, 1].max()),
                    "mean_norm": float(cartesian.norm(dim=-1).mean()),
                }
            )
        except Exception as exc:
            rows.append({"data": name, "error": str(exc)})
    print(json.dumps(rows, indent=2))


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Tuple[RealSphereTrainConfig, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="volcano", choices=EARTH_DATASETS)
    parser.add_argument("--earth-datadir", default="data/earth_data")
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
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/rmgm_real_sphere")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--metric-normalize", action="store_true")
    parser.add_argument("--generator-type", default="local_gmm", choices=["mlp", "local_gmm"])
    parser.add_argument("--generator-components", type=int, default=8)
    parser.add_argument("--generator-residual-scale", type=float, default=0.25)
    parser.add_argument("--generator-kmeans-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generator-kmeans-samples", type=int, default=8192)
    parser.add_argument("--generator-kmeans-iters", type=int, default=40)
    parser.add_argument("--inspect-data", action="store_true")
    args = parser.parse_args()
    inspect = args.inspect_data
    delattr(args, "inspect_data")
    return RealSphereTrainConfig(**vars(args)), inspect


if __name__ == "__main__":
    config, should_inspect = parse_args()
    if should_inspect:
        inspect_data(config.earth_datadir)
    else:
        run(config)
