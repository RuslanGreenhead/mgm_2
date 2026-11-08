"""Render sample/data plots from a saved R-MGM generator checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from manifm.rmgm import build_generator
from rmgm_train import TrainConfig, make_experiment, save_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-samples", type=int, default=6000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg_dict["eval_samples"] = args.eval_samples
    cfg = TrainConfig(**cfg_dict)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    dtype = getattr(torch, cfg.dtype)
    manifold, dim, data_sampler = make_experiment(cfg.experiment)

    generator_cfg = {
        "generator": {
            "latent_dim": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "activation": "silu",
            "generator_type": cfg.generator_type,
            "generator_components": cfg.generator_components,
            "generator_residual_scale": cfg.generator_residual_scale,
            "torus_map": cfg.torus_map,
            "torus_generator": cfg.torus_generator,
            "torus_components": cfg.torus_components,
            "torus_residual_scale": cfg.torus_residual_scale,
            "torus_mixture_init": cfg.torus_mixture_init,
        }
    }
    generator = build_generator(generator_cfg, manifold, dim).to(device=device, dtype=dtype)
    generator.load_state_dict(ckpt["generator"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_plots(
        cfg,
        manifold,
        data_sampler,
        generator,
        device,
        dtype,
        out,
        int(ckpt["step"]),
    )
    print(out / f"samples-{int(ckpt['step']):06d}.png")


if __name__ == "__main__":
    main()
