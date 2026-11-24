"""Exploratory conditional denoising initialization for the DNA generator."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmgm.dna.data import PromoterArrays
from rmgm.dna.geometry import sphere_exp, tangent, target_sphere
from rmgm.dna.networks import Generator


def warmup_step(generator: Generator, optimizer: torch.optim.Optimizer,
                labels: torch.Tensor, signal: torch.Tensor,
                rng: torch.Generator, grad_clip: float) -> float:
    """Denoise tangent-perturbed real DNA using the existing conditional net."""
    clean = target_sphere(labels)
    displacement = tangent(clean, torch.randn(clean.shape, device=clean.device, generator=rng))
    displacement = displacement / displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 0.1 + 1.4 * torch.rand((len(labels), 1, 1), device=clean.device, generator=rng)
    corrupted = sphere_exp(clean, displacement * angle)
    predicted = generator(corrupted, signal)
    # Chordal squared distance is smooth at exact matches on the sphere.
    loss = (predicted - clean).square().sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite generator warmup loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
    if not all(torch.isfinite(p.grad).all() for p in generator.parameters() if p.grad is not None):
        raise FloatingPointError("Non-finite generator warmup gradient")
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    config = json.loads(args.config.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(seed + 3000)
    generator = Generator(int(config["width"]), int(config["blocks"])).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=float(config["lr_g"]), weight_decay=0)
    dataset = PromoterArrays(args.data, "train", int(config["train_shift"]), seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps({"config": config, "steps": args.steps,
        "objective": "conditional spherical denoising; chordal squared distance",
        "code_commit": os.environ.get("RMGM_GIT_COMMIT")}, indent=2) + "\n")
    started = time.monotonic()
    step = 0
    epoch = 0
    with (args.output / "history.jsonl").open("w") as history:
        while step < args.steps:
            dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True,
                                drop_last=True, num_workers=int(config.get("num_workers", 0)))
            for labels, signal in loader:
                loss = warmup_step(generator, optimizer, labels.to(device), signal.to(device),
                                   rng, float(config["grad_clip"]))
                step += 1
                if step % 50 == 0 or step == args.steps:
                    event = {"phase": "generator_warmup", "step": step,
                             "seconds": time.monotonic() - started, "loss": loss}
                    history.write(json.dumps(event) + "\n")
                    history.flush()
                    print(event, flush=True)
                if step == args.steps:
                    break
            epoch += 1
    checkpoint = args.output / "generator_warmup.pt"
    torch.save({"generator": generator.state_dict(), "config": config, "steps": step}, checkpoint)
    print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
