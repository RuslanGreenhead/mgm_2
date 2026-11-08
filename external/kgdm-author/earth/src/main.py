from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from task.earth_dataset import EarthDataset
from models.backbone_earth import EarthModel
from models.earth_drift import EarthDrift
from src.train import TrainConfig, train


DATASETS = ("earthquake", "volcano", "fire", "flood")
DRIFT_TYPES = ("euclidean", "sphere", "spectral")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(cfg, seed: int):
    data = cfg["data"]
    batch_size = data["batch_size"]
    num_workers = data.get("num_workers", 0)
    pin_memory = data.get("pin_memory", False)
    data_dir = Path(data["data_dir"])
    dataset_file = data["dataset_file"]
    expand_train = data.get("expand_train", True)

    train_ds = EarthDataset(data_dir, dataset_file, split="train", expand_train=expand_train)
    val_ds = EarthDataset(data_dir, dataset_file, split="val", expand_train=False)
    test_ds = EarthDataset(data_dir, dataset_file, split="test", expand_train=False)

    # deterministic shuffling: a seeded generator + per-worker seeding
    gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True, generator=gen,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader, test_loader


def calibrated_kappas(cfg, dataset: str,
                      winners_path: str = "configs/winners.yaml") -> list[float]:
    """Calibrated per-dataset kappa ladder from the spectral formulations."""
    winners = load_yaml(winners_path)
    nu = float(cfg["model"].get("matern_nu", 2.5))
    key = f"spectral-nu{'inf' if math.isinf(nu) else nu}"
    if key not in winners:
        raise KeyError(f"no calibrated kappa ladder for matern_nu={nu} in "
                       f"{winners_path}; pass --matern_kappas")
    return [float(k) for k in winners[key]["per_dataset"][dataset]["matern_kappas"]]


def apply_formulation(cfg, formulation: str, dataset: str,
                      winners_path: str = "configs/winners.yaml"):
    """Apply a formulation's model flags (shared + per-dataset) onto cfg."""
    winners = load_yaml(winners_path)
    if formulation not in winners:
        raise KeyError(f"formulation '{formulation}' not in {winners_path} "
                       f"(has {sorted(winners)})")
    spec = dict(winners[formulation])
    spec.update(spec.pop("per_dataset", {}).get(dataset, {}))
    cfg["model"].update(spec)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--formulation", type=str, default=None,
                   help="key in configs/winners.yaml; applies that "
                        "formulation's model flags for the chosen dataset")
    p.add_argument("--winners", type=str, default="configs/winners.yaml")
    p.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    p.add_argument("--drift_type", type=str, default=None, choices=DRIFT_TYPES)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--n_blocks", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--dist_norm", type=str2bool, default=None)
    p.add_argument("--grad_form", type=str2bool, default=None)
    p.add_argument("--kernel", type=str, default=None, choices=("laplace", "gauss"))
    p.add_argument("--drop_T", type=str2bool, default=None)
    p.add_argument("--max_step", type=float, default=None)
    p.add_argument("--matern_nu", type=float, default=None)  # accepts "inf"
    p.add_argument("--matern_kappas", type=str, default=None,
                   help="comma-separated kappa ladder; default is the "
                        "calibrated per-dataset ladder from winners.yaml")
    p.add_argument("--matern_tables", type=str, default=None)
    p.add_argument("--spec_norm", type=str2bool, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results",
                   help="directory for the final-metrics JSON")
    p.add_argument("--use_wandb", type=str2bool, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.dataset is not None:
        cfg["data"]["dataset_file"] = args.dataset
    dataset_name = cfg["data"]["dataset_file"]

    if args.formulation is not None:
        apply_formulation(cfg, args.formulation, dataset_name, args.winners)

    # explicit CLI flags override config and formulation
    if args.drift_type is not None:
        cfg["model"]["drift_type"] = args.drift_type
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = args.warmup_steps
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = args.eval_every
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = args.hidden_dim
    if args.n_blocks is not None:
        cfg["model"]["n_blocks"] = args.n_blocks
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.dist_norm is not None:
        cfg["model"]["dist_norm"] = args.dist_norm
    if args.grad_form is not None:
        cfg["model"]["grad_form"] = args.grad_form
    if args.kernel is not None:
        cfg["model"]["kernel"] = args.kernel
    if args.drop_T is not None:
        cfg["model"]["drop_T"] = args.drop_T
    if args.max_step is not None:
        cfg["model"]["max_step"] = args.max_step
    if args.matern_nu is not None:
        cfg["model"]["matern_nu"] = args.matern_nu
    if args.matern_kappas is not None:
        cfg["model"]["matern_kappas"] = [float(s) for s in args.matern_kappas.split(",")]
    if args.matern_tables is not None:
        cfg["model"]["matern_tables"] = args.matern_tables
    if args.spec_norm is not None:
        cfg["model"]["spec_norm"] = args.spec_norm
    if args.use_wandb is not None:
        cfg["train"]["use_wandb"] = args.use_wandb

    seed = args.seed if args.seed is not None else cfg["train"].get("seed", 0)
    set_seed(seed)

    device = get_device(cfg["device"])
    print(f"Using device: {device} | seed: {seed}")

    m = cfg["model"]
    drift_type = m.get("drift_type", "sphere")
    if drift_type == "spectral":
        if m.get("matern_kappas") is None:
            m["matern_kappas"] = calibrated_kappas(cfg, dataset_name)
        # build any missing kernel lookup tables before the model is created
        from src.matern_tables import ensure_tables
        nu = float(m.get("matern_nu", 2.5))
        ensure_tables(m.get("matern_tables", "matern_tables.pt"),
                      [(nu, k) for k in m["matern_kappas"]])

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed)

    backbone = EarthModel(
        input_dim=m.get("dim_noise", 3),
        hidden_dim=m.get("hidden_dim", 128),
        n_blocks=m.get("n_blocks", 4),
        output_dim=3,
    ).to(device)

    model = EarthDrift(
        backbone=backbone,
        dim_noise=m.get("dim_noise", 3),
        device=device,
        drift_type=drift_type,
        temps=tuple(m.get("temps", (0.02, 0.05, 0.2))),
        dist_norm=m.get("dist_norm", True),
        grad_form=m.get("grad_form", True),
        kernel=m.get("kernel", "laplace"),
        drop_T=m.get("drop_T", True),
        max_step=m.get("max_step", 1.0),
        matern_nu=float(m.get("matern_nu", 2.5)),
        matern_kappas=tuple(m.get("matern_kappas") or ()),
        matern_tables=m.get("matern_tables", "matern_tables.pt"),
        spec_norm=m.get("spec_norm", False),
    ).to(device)

    print(f"Dataset: {dataset_name} | drift_type: {drift_type} | "
          f"hidden_dim: {m.get('hidden_dim', 128)} | n_blocks: {m.get('n_blocks', 4)}")
    print(f"temps={model.temps} dist_norm={model.dist_norm} "
          f"grad_form={model.grad_form} kernel={model.kernel} "
          f"drop_T={model.drop_T} max_step={model.max_step} "
          f"matern_nu={m.get('matern_nu')} "
          f"matern_kappas={m.get('matern_kappas')} "
          f"spec_norm={m.get('spec_norm', False)} seed={seed} "
          f"batch_size={cfg['data']['batch_size']}")

    t = cfg["train"]
    run_name = (args.run_name or t.get("run_name")
                or f"{args.formulation or drift_type}_{dataset_name}_s{seed}")
    cfg_train = TrainConfig(
        project=t.get("project", "kgd-earth"),
        entity=t.get("entity"),
        run_name=run_name,
        epochs=t["epochs"],
        lr=float(t["lr"]),
        warmup_steps=t.get("warmup_steps", 2500),
        log_every=t.get("log_every", 50),
        eval_every=t.get("eval_every", 500),
        device=device,
        batch_size=cfg["data"]["batch_size"],
        grad_clip=t.get("grad_clip", 1.0),
        weight_decay=t.get("weight_decay", 1e-2),
        ema_decay=t.get("ema_decay", 0.999),
        mmd_gamma=t.get("mmd_gamma", 1.0),
        selection_metric=t.get("selection_metric", "mmd"),
        selection_mode=t.get("selection_mode", "min"),
        n_eval_repeats=t.get("n_eval_repeats", 3),
        early_stopping_patience=t.get("early_stopping_patience", 20),
        use_wandb=t.get("use_wandb", False),
    )

    result = train(model, train_loader, val_loader, test_loader, cfg_train)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "formulation": args.formulation,
            "dataset": dataset_name,
            "seed": seed,
            "model": {k: v for k, v in m.items()},
            "best_step": result["best_step"],
            "best_epoch": result["best_epoch"],
            "test_mean": result["test_mean"],
            "test_std": result["test_std"],
        }, f, indent=2)
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
