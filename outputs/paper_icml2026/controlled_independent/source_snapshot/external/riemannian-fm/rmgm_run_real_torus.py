"""Run the protein/RNA torus real-data benchmark block."""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

import torch

from rmgm_train_real import RealTrainConfig, run


TORUS_DATASETS = ("general", "glycine", "proline", "prepro")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="../../outputs/rmgm_real_torus")
    parser.add_argument("--datasets", nargs="*", default=list(TORUS_DATASETS))
    parser.add_argument("--methods", nargs="*", choices=("rmgm", "rfm"), default=["rmgm", "rfm"])
    args = parser.parse_args()

    base = RealTrainConfig(
        data="general",
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        eval_samples=args.eval_samples,
        ode_steps=args.ode_steps,
        latent_dim=16,
        log_every=max(1, args.steps // 4),
        plot_every=max(1, args.steps // 2),
        output_dir=args.output_dir,
        device="cuda",
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for data in args.datasets:
        if "rmgm" in args.methods:
            rmgm_cfg = replace(
                base,
                data=data,
                method="rmgm",
                critic_steps=1,
                lr_g=1e-4,
                lr_c=2e-4,
                torus_generator="gmm",
                torus_components=64 if data == "general" else 32,
                torus_residual_scale=0.18,
                torus_kmeans_init=True,
                torus_kmeans_samples=65536 if data == "general" else 16384,
            )
            print(f"\n=== {data} / R-MGM ===")
            run(rmgm_cfg)
            _clear_cuda()

        if "rfm" in args.methods:
            rfm_cfg = replace(base, data=data, method="rfm", lr=2e-4)
            print(f"\n=== {data} / RFM sample baseline ===")
            run(rfm_cfg)
            _clear_cuda()


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
