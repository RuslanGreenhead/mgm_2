"""Single-run entry point for the promoter-DNA experiments.

Resolution order for every field (last wins):

    configs/config.yaml   ->   --formulation recipe   ->   explicit CLI flag

A `--formulation` pins the static geometry of a paper cell (the booleans
`sphere / simplex / spectral / grad_form`, from the registry below) and loads
its tuned per-cell scalars (`lr`, `t_list`, or the spectral `kappa_list / nu /
lmax / eta`, plus `best_kmer_min_epoch`) from configs/winners.yaml. Any field can
still be overridden on the CLI.

Examples
--------
    # reproduce the headline winner (sphere + kernel-gradient drift)
    python -m src.main --formulation sphere-grad

    # the spectral Matern-kernel cell, 3 seeds
    for s in 0 1 2; do python -m src.main --formulation spectral --seed $s; done

    # an ad-hoc cell straight from config.yaml + CLI
    python -m src.main --sphere true --grad_form true --lr 1.1e-4 \
        --t_list "0.01,0.04,0.15" --sei_batch_tau 0.1 --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.train import run

REPO = Path(__file__).resolve().parents[1]

# Paper cells. Each entry pins ONLY the static geometry booleans; the tuned
# scalars (lr, t_list / kappa_list+nu+lmax+eta, best_kmer_min_epoch) live in
# configs/winners.yaml. The six geometry cells share the multi-tau sequence
# drift; `spectral` swaps in the spherical-harmonic Matern kernel score.
FORMULATIONS_BY_KEY = {
    "ambient-nograd": {"sphere": False, "simplex": False, "spectral": False, "grad_form": False},
    "ambient-grad":   {"sphere": False, "simplex": False, "spectral": False, "grad_form": True},
    "simplex-nograd": {"sphere": False, "simplex": True,  "spectral": False, "grad_form": False},
    "simplex-grad":   {"sphere": False, "simplex": True,  "spectral": False, "grad_form": True},
    "sphere-nograd":  {"sphere": True,  "simplex": False, "spectral": False, "grad_form": False},
    "sphere-grad":    {"sphere": True,  "simplex": False, "spectral": False, "grad_form": True},
    "spectral":       {"sphere": False, "simplex": False, "spectral": True,  "grad_form": False},
}

# winners.yaml scalar keys that map straight onto cfg fields.
WINNER_SCALARS = ("lr", "t_list", "kappa", "kappa_list", "nu", "lmax", "eta",
                  "best_kmer_min_epoch")


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
    p.add_argument("--config", default=str(REPO / "configs" / "config.yaml"))
    p.add_argument("--winners", default=str(REPO / "configs" / "winners.yaml"))
    p.add_argument("--formulation", default=None, choices=sorted(FORMULATIONS_BY_KEY),
                   help="load a paper cell (geometry + tuned scalars).")

    # geometry (optional bool overrides; None -> take from formulation/config)
    for flag in ("sphere", "simplex", "spectral", "grad_form", "dist_norm"):
        p.add_argument(f"--{flag}", type=str2bool, default=None)

    # drift scalars
    p.add_argument("--t_list", default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--kappa_list", default=None)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)

    # optimization / schedule
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--val_batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--betas", default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--ema_decay", type=float, default=None)

    # conditioning / eval / objective
    p.add_argument("--sei_batch_tau", type=float, default=None)
    p.add_argument("--eval_every_epochs", type=int, default=None)
    p.add_argument("--test_eval_every_epochs", type=int, default=None)
    p.add_argument("--score_kmer_floor", type=float, default=None)
    p.add_argument("--best_kmer_min_epoch", type=int, default=None)

    # bookkeeping
    p.add_argument("--data_dir", default=None)
    p.add_argument("--logs_root", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strict_determinism", type=str2bool, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_mode", default=None, choices=("online", "offline", "disabled"))
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def _flatten(cfg_yaml):
    """configs/config.yaml is grouped (data/model/train); flatten to one dict
    keyed by the cfg field names train.run() reads."""
    flat = {}
    for section in ("data", "model", "train", "wandb"):
        flat.update(cfg_yaml.get(section, {}) or {})
    return flat


def build_cfg(args):
    cfg_yaml = yaml.safe_load(Path(args.config).read_text())
    flat = _flatten(cfg_yaml)

    # 1) formulation: static geometry from the registry + tuned scalars from winners
    if args.formulation is not None:
        flat.update(FORMULATIONS_BY_KEY[args.formulation])
        winners = yaml.safe_load(Path(args.winners).read_text()) or {}
        cell = winners.get(args.formulation)
        if cell is None:
            raise SystemExit(f"{args.winners} has no entry for '{args.formulation}'")
        for k in WINNER_SCALARS:
            if k in cell and cell[k] is not None:
                flat[k] = cell[k]
        flat.setdefault("run_name", args.formulation)

    # 2) explicit CLI flags win over everything
    for k, v in vars(args).items():
        if k in ("config", "winners", "formulation"):
            continue
        if v is not None:
            flat[k] = v

    cfg = SimpleNamespace(**flat)
    _validate(cfg)
    return cfg


def _validate(cfg):
    required = ("data_dir", "epochs", "batch", "lr", "sphere", "simplex",
                "spectral", "grad_form", "t_list", "sei_batch_tau",
                "score_kmer_floor", "seed")
    missing = [k for k in required if getattr(cfg, k, None) is None]
    if missing:
        raise SystemExit(f"missing required fields: {missing} "
                         "(set them in configs/config.yaml or pass via CLI)")
    n_geom = sum(bool(getattr(cfg, g)) for g in ("simplex", "sphere", "spectral"))
    if n_geom > 1:
        raise SystemExit("at most one of simplex/sphere/spectral may be true")
    if cfg.grad_form and cfg.spectral:
        raise SystemExit("grad_form does not apply to the spectral kernel cell")


def main():
    args = parse_args()
    cfg = build_cfg(args)
    geom = ("spectral" if cfg.spectral else "sphere" if cfg.sphere
            else "simplex" if cfg.simplex else "ambient")
    print(f"=== {args.formulation or '(custom)'} | geom={geom} "
          f"grad_form={cfg.grad_form} | lr={cfg.lr} tau={cfg.sei_batch_tau} "
          f"epochs={cfg.epochs} seed={cfg.seed} ===")
    run(cfg)


if __name__ == "__main__":
    main()
