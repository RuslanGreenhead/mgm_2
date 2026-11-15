"""Make a compact contact sheet from selected experiment figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("items", nargs="+", help="Pairs of label=path")
    args = parser.parse_args()

    parsed = []
    for item in args.items:
        label, path = item.split("=", 1)
        parsed.append((label, Path(path)))

    cols = 2
    rows = (len(parsed) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 9.5))
    axes = axes.reshape(-1)
    for ax, (label, path) in zip(axes, parsed):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=13)
        ax.axis("off")
    for ax in axes[len(parsed) :]:
        ax.axis("off")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
