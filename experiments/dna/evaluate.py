"""Evaluate a selected checkpoint without starting or resuming training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.dna.train import ROOT, choose_device, evaluate, make_loader
from rmgm.dna.data import verify_prepared
from rmgm.dna.evaluation import KGDMSeiMetric
from rmgm.dna.method import PromoterRMGM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/promoters_rmf10")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sei-data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = choose_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    verify_prepared(args.data)
    model = PromoterRMGM(config, device)
    model.load_state_dict(state["method"])
    rng = torch.Generator(device=device).manual_seed(int(config["seed"]) + 2000)
    sei = KGDMSeiMetric(args.sei_data_dir, ROOT / "external/kgdm-author/dna") if args.sei_data_dir else None
    result = {"checkpoint": str(args.checkpoint), "split": args.split,
              "metrics": evaluate(model, make_loader(args.data, args.split, config), rng, sei,
                                  sei_batch_size=int(config.get("sei_batch_size", 1)))}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
