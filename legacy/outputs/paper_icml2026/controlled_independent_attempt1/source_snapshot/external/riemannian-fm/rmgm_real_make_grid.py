"""Build contact sheets from real-data comparison summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_EARTH_ROWS = ["volcano", "earthquake", "flood", "fire"]
DEFAULT_METHODS = ["rmgm", "rfm"]
METHOD_LABELS = {"rmgm": "R-MGM", "rfm": "RFM"}


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    by_key = {(row["data"], row["method"]): row for row in summary}

    fig, axes = plt.subplots(
        len(args.rows),
        len(args.methods),
        figsize=(6.7 * len(args.methods), 2.75 * len(args.rows)),
        squeeze=False,
    )
    for i, data in enumerate(args.rows):
        for j, method in enumerate(args.methods):
            ax = axes[i, j]
            rec = by_key.get((data, method))
            if rec is None:
                ax.axis("off")
                continue
            png = choose_png(Path(rec[args.run_key]), int(rec[args.step_key]))
            if png is not None and png.exists():
                ax.imshow(mpimg.imread(png))
            ax.set_axis_off()
            ax.set_title(
                (
                    f"{data} / {METHOD_LABELS.get(method, method)}\n"
                    f"MMD2={rec['best_mmd2']:.4g}, symNN={rec['best_symmetric_nn']:.4g}"
                ),
                fontsize=11,
            )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(args.output.resolve())


def choose_png(run_dir: Path, desired_step: int) -> Path | None:
    figs_dir = run_dir / "figs"
    figs = sorted(figs_dir.glob("*.png"))
    if not figs:
        return None
    return min(figs, key=lambda path: abs(step_from_png(path) - desired_step))


def step_from_png(path: Path) -> int:
    match = re.search(r"-(\d{6})\.png$", path.name)
    return int(match.group(1)) if match else -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", default=DEFAULT_EARTH_ROWS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--run-key", default="best_mmd2_run")
    parser.add_argument("--step-key", default="best_mmd2_step")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main()
