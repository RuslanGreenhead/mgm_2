"""Single-config entrypoint for geometric drift experiments.

Hyperparameters come from configs/<dataset>.yaml. Pass --dataset to pick the
config; --geometry, --method, --seed, and --out-dir are required runtime
choices. Any other CLI flag (e.g. --steps, --temperature) overrides its
YAML counterpart.

Examples
--------
  python main.py --dataset checkerboard_sphere --geometry manifold --method gradient
  python main.py --dataset swissroll_hyperboloid --geometry euclidean --method base \
                 --steps 1000 --out-dir results/swissroll_hyp_eucl_base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from models.backbone import MLPBackbone
from models.drift import GeometricDrift
from tasks import DATASETS, get_manifold, make_dataset, manifold_for_dataset
from train import TrainConfig, train


CONFIG_DIR = Path(__file__).parent / "configs"


def get_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--geometry", choices=["euclidean", "manifold"], default="manifold")
    p.add_argument("--method", choices=["base", "gradient"], default="base")

    # All numeric/string fields below default to None so we can tell whether
    # the user passed them; if not, we fall back to the YAML config.
    p.add_argument("--n-points", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--capture-steps", type=str, default=None,
                   help="comma-separated training steps at which to save samples")
    p.add_argument("--n-eval-samples", type=int, default=None)
    p.add_argument("--sliced-w2-n-proj", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)

    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--max-step", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--config", type=str, default=None,
                   help="path to YAML config; default: configs/<dataset>.yaml")
    p.add_argument("--out-dir", default=None,
                   help="output directory; default: results/<dataset>/<geometry>-<method>/seed<n>")
    return p.parse_args()


def resolve(args) -> Dict[str, Any]:
    """Merge YAML defaults with optional per-mode overrides and CLI flags.

    Precedence (lowest to highest):
        hard-coded default  <  YAML base  <  YAML overrides[<geometry>-<method>]  <  CLI flag.
    """
    cfg_path = Path(args.config) if args.config else CONFIG_DIR / f"{args.dataset}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    raw = load_yaml(cfg_path)

    overrides_block = raw.pop("overrides", {}) or {}
    mode_key = f"{args.geometry}-{args.method}"
    mode_overrides = overrides_block.get(mode_key, {}) or {}

    cfg = {**raw, **mode_overrides}

    def pick(key, cli_value, default=None):
        return cli_value if cli_value is not None else cfg.get(key, default)

    return {
        "n_points":         pick("n_points",         args.n_points,         10000),
        "batch_size":       pick("batch_size",       args.batch_size,       2048),
        "steps":            pick("steps",            args.steps,            2000),
        "lr":               pick("lr",               args.lr,               1e-3),
        "log_every":        pick("log_every",        args.log_every,        100),
        "capture_steps":    pick("capture_steps",    args.capture_steps,    "0,50,200,1000,2000"),
        "n_eval_samples":   pick("n_eval_samples",   args.n_eval_samples,   4096),
        "sliced_w2_n_proj": pick("sliced_w2_n_proj", args.sliced_w2_n_proj, 128),
        "grad_clip":        pick("grad_clip",        args.grad_clip,        1.0),
        "temperature":      pick("temperature",      args.temperature,      0.5),
        "eta":              pick("eta",              args.eta,              1.0),
        "max_step":         pick("max_step",         args.max_step,         5.0),
        "hidden":           pick("hidden",           args.hidden,           256),
        "depth":            pick("depth",            args.depth,            5),
        "device":           pick("device",           args.device,           "auto"),
        "_config_path":     str(cfg_path),
        "_mode_overrides":  dict(mode_overrides),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = resolve(args)
    device = get_device(cfg["device"])
    print(f"device={device}  config={cfg['_config_path']}")
    if cfg["_mode_overrides"]:
        print(f"applied overrides[{args.geometry}-{args.method}]: {cfg['_mode_overrides']}")

    manifold_name = manifold_for_dataset(args.dataset)
    manifold = get_manifold(manifold_name)
    dim_intrinsic = 2  # both S^2 and H^2

    dataset = make_dataset(args.dataset, cfg["n_points"], device="cpu")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    target_pool = dataset.data  # [n_points, ambient_dim]

    dim_ambient = dim_intrinsic + 1
    backbone = MLPBackbone(
        dim_noise=dim_ambient, dim_data=dim_ambient,
        hidden=cfg["hidden"], depth=cfg["depth"],
    ).to(device)
    model = GeometricDrift(
        backbone=backbone,
        manifold=manifold,
        dim_noise=dim_ambient,
        dim_intrinsic=dim_intrinsic,
        geometry=args.geometry,
        method=args.method,
        temperature=cfg["temperature"],
        eta=cfg["eta"],
        max_step=cfg["max_step"],
    ).to(device)

    if args.out_dir is None:
        out_dir = Path("results") / args.dataset / f"{args.geometry}-{args.method}" / f"seed{args.seed}"
    else:
        out_dir = Path(args.out_dir)

    capture_steps = sorted({int(s) for s in str(cfg["capture_steps"]).split(",") if s.strip()})

    train_cfg = TrainConfig(
        steps=cfg["steps"],
        lr=cfg["lr"],
        log_every=cfg["log_every"],
        capture_steps=capture_steps,
        n_eval_samples=cfg["n_eval_samples"],
        sliced_w2_n_proj=cfg["sliced_w2_n_proj"],
        grad_clip=cfg["grad_clip"],
        device=device,
        out_dir=str(out_dir),
    )

    extra_meta = {
        "manifold": manifold_name,
        "geometry": args.geometry,
        "method": args.method,
        "temperature": cfg["temperature"],
        "eta": cfg["eta"],
        "max_step": cfg["max_step"],
        "hidden": cfg["hidden"],
        "depth": cfg["depth"],
        "n_points": cfg["n_points"],
        "batch_size": cfg["batch_size"],
        "seed": args.seed,
        "config_path": cfg["_config_path"],
    }

    print(json.dumps(extra_meta, indent=2))
    train(
        model, loader, target_pool, train_cfg,
        dataset_name=args.dataset, manifold=manifold,
        extra_meta=extra_meta,
    )


if __name__ == "__main__":
    main()
