"""Train, resume, and evaluate conditional promoter R-MGM on any PyTorch device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmgm.dna.data import PromoterArrays, verify_prepared
from rmgm.dna.evaluation import KGDMSeiMetric, kmer_counts, kmer_pearson, one_hot_with_unknown
from rmgm.dna.method import PromoterRMGM


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config["seed"] < 0 or config["batch_size"] < 1:
        raise ValueError("Invalid seed or batch size")
    return config


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def make_loader(directory: Path, split: str, config: dict, epoch: int = 0) -> DataLoader:
    shift = int(config.get("train_shift", 10)) if split == "train" else 0
    dataset = PromoterArrays(directory, split, shift, int(config["seed"]))
    dataset.set_epoch(epoch)
    shuffle_rng = torch.Generator().manual_seed(int(config["seed"]) + epoch * 1009)
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=split == "train",
                      num_workers=int(config.get("num_workers", 0)), pin_memory=False,
                      drop_last=split == "train", generator=shuffle_rng)


def evaluate(model: PromoterRMGM, loader: DataLoader, rng: torch.Generator,
             sei: KGDMSeiMetric | None = None) -> dict:
    true_counts = np.zeros(4096, dtype=np.int64)
    generated_counts = np.zeros(4096, dtype=np.int64)
    sei_sum = 0.0
    examples = 0
    for labels, signal in loader:
        generated = model.sample(signal, rng).square().argmax(dim=-1)
        true_counts += kmer_counts(labels.numpy())
        generated_counts += kmer_counts(generated.cpu().numpy())
        if sei is not None:
            predicted = one_hot_with_unknown(generated).to(model.device)
            reference = one_hot_with_unknown(labels).to(model.device)
            errors = sei.squared_error(predicted, reference)
            sei_sum += float(errors.sum())
        examples += len(labels)
    result = {"examples": examples, "kmer_pearson": kmer_pearson(true_counts, generated_counts)}
    if sei is not None:
        result["sei_h3k4me3_mse"] = sei_sum / examples
    return result


def save_checkpoint(path: Path, model: PromoterRMGM, epoch: int, next_batch: int, config: dict,
                    rng: torch.Generator, best_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    torch.save({"method": model.state_dict(), "epoch": epoch, "next_batch": next_batch, "config": config,
                "rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "best_score": best_score}, staging)
    os.replace(staging, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dna/rmgm.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/promoters_rmf10")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dna/default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="Override the configured seed for repeated runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Short local smoke run")
    parser.add_argument("--eval-sei", action="store_true")
    parser.add_argument("--test", action="store_true", help="Evaluate the selected checkpoint on chr8/9")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    seed_all(int(config["seed"]))
    device = choose_device(args.device)
    rng = torch.Generator(device=device).manual_seed(int(config["seed"]) + 1000)
    model = PromoterRMGM(config, device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = verify_prepared(args.data)
    (args.output / "run.json").write_text(json.dumps({"config": config, "data_manifest": manifest,
                                                       "git_commit": git_commit(), "device": str(device)}, indent=2) + "\n")
    start_epoch, start_batch, best_score = 0, 0, float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("seed", "batch_size", "width", "critic_width", "blocks", "train_shift"):
            if state["config"].get(key) != config.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        model.load_state_dict(state["method"])
        start_epoch = state["epoch"]
        start_batch = state.get("next_batch", 0)
        best_score = state["best_score"]
        rng.set_state(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    valid_loader = make_loader(args.data, "valid", config)
    sei = None
    if args.eval_sei:
        sei = KGDMSeiMetric(Path(config["sei_data_dir"]), ROOT / "external/kgdm-author/dna")
    started = time.monotonic()
    with (args.output / "history.jsonl").open("a") as history:
        for epoch in range(start_epoch, int(config["epochs"])):
            train_loader = make_loader(args.data, "train", config, epoch)
            for batch_index, (labels, signal) in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                result = model.update(labels, signal, rng)
                if model.iteration % int(config.get("log_every", 50)) == 0:
                    event = {"epoch": epoch, "step": model.iteration, "seconds": time.monotonic() - started, **result}
                    history.write(json.dumps(event) + "\n")
                    history.flush()
                    print(event, flush=True)
                if model.iteration % int(config.get("checkpoint_every_steps", 1000)) == 0:
                    save_checkpoint(args.output / "last.pt", model, epoch, batch_index + 1,
                                    config, rng, best_score)
                if args.max_steps and model.iteration >= args.max_steps:
                    break
            # The holdout is consulted for checkpoint selection; test remains sealed.
            if epoch % int(config.get("eval_every_epochs", 1)) == 0 or args.max_steps:
                metrics = evaluate(model, valid_loader, rng, sei)
                event = {"epoch": epoch, "step": model.iteration, "split": "valid", **metrics}
                history.write(json.dumps(event) + "\n")
                history.flush()
                print(event, flush=True)
                score = -metrics["sei_h3k4me3_mse"] if sei else metrics["kmer_pearson"]
                if score > best_score:
                    best_score = score
                    save_checkpoint(args.output / "best.pt", model, epoch + 1, 0, config, rng, best_score)
            save_checkpoint(args.output / "last.pt", model, epoch + 1, 0, config, rng, best_score)
            if args.max_steps and model.iteration >= args.max_steps:
                break
    if args.test:
        selected = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(selected["method"])
        test_rng = torch.Generator(device=device).manual_seed(int(config["seed"]) + 2000)
        metrics = evaluate(model, make_loader(args.data, "test", config), test_rng, sei)
        (args.output / "test.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print({"split": "test", **metrics})


if __name__ == "__main__":
    main()
