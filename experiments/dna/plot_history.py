"""Plot R-MGM DNA training losses and validation metrics from history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.history.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    events = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()]
    training = [event for event in events if "critic_loss" in event]
    validation = [event for event in events if event.get("split") == "valid"]
    if not training:
        raise ValueError("No training loss events found")

    steps = np.array([event["step"] for event in training], dtype=np.int64)
    critic = np.array([event["critic_loss"] for event in training], dtype=float)
    generator = np.array([event.get("generator_loss", np.nan) for event in training], dtype=float)
    epochs = np.array([event["epoch"] + 1 for event in validation], dtype=np.int64)
    kmer = np.array([event["kmer_pearson"] for event in validation], dtype=float)
    sei = np.array([event.get("sei_h3k4me3_mse", np.nan) for event in validation], dtype=float)
    np.savez_compressed(output / "curves.npz", steps=steps, critic_loss=critic,
                        generator_loss=generator, epochs=epochs,
                        kmer_pearson=kmer, sei_h3k4me3_mse=sei)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(steps, critic, label="Critic loss", linewidth=1.2)
    ax.plot(steps, generator, label="Generator loss", linewidth=1.2)
    ax.set(xlabel="Training step", ylabel="Logged loss", title="R-MGM training losses")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "losses.png", dpi=180)
    plt.close(fig)

    if len(validation):
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        axes[0].plot(epochs, kmer, marker="o", markersize=3)
        axes[0].set(ylabel="6-mer Pearson ↑", title="Validation metrics")
        axes[1].plot(epochs, sei, marker="o", markersize=3)
        axes[1].set(xlabel="Completed epoch", ylabel="Sei-MSE ↓")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.savefig(output / "validation.png", dpi=180)
        plt.close(fig)
    print(f"Saved plots and curves to {output}")


if __name__ == "__main__":
    main()
