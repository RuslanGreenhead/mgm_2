"""Collect best/final metrics from R-MGM/RFM experiment directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for metrics in root.rglob("metrics.jsonl"):
        yield metrics.parent


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: Path) -> dict:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        raise ValueError(f"No metrics rows in {run_dir}")
    best = min(rows, key=lambda row: row.get("mmd2", float("inf")))
    final = rows[-1]
    figs = sorted((run_dir / "figs").glob("samples-*.png"))
    best_step = int(best["step"])
    best_fig = nearest_figure(figs, best_step)
    return {
        "run_dir": str(run_dir),
        "experiment": run_dir.parent.name,
        "family": run_dir.parent.parent.name,
        "best_step": best_step,
        "best_mmd2": best.get("mmd2"),
        "best_constraint_error": best.get("constraint_error"),
        "final_step": final.get("step"),
        "final_mmd2": final.get("mmd2"),
        "final_constraint_error": final.get("constraint_error"),
        "best_fig": str(best_fig) if best_fig else None,
    }


def nearest_figure(figs: list[Path], step: int) -> Path | None:
    if not figs:
        return None

    def fig_step(path: Path) -> int:
        return int(path.stem.split("-")[-1])

    return min(figs, key=lambda path: abs(fig_step(path) - step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        rows.extend(summarize_run(run_dir) for run_dir in iter_run_dirs(root))
    rows.sort(key=lambda row: (row["experiment"], row["family"], row["best_mmd2"]))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    header = [
        "experiment",
        "family",
        "best_mmd2",
        "best_step",
        "final_mmd2",
        "constraint",
        "best_fig",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["experiment"]),
                    str(row["family"]),
                    f"{row['best_mmd2']:.6g}",
                    str(row["best_step"]),
                    f"{row['final_mmd2']:.6g}",
                    f"{row['final_constraint_error']:.3g}",
                    str(row["best_fig"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
