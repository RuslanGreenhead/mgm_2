"""Aggregate independent seed test metrics without selecting on the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for directory in args.runs:
        result = json.loads((directory / "test.json").read_text())
        metrics = result.get("metrics", result)
        config = json.loads((directory / "run.json").read_text())["config"]
        records.append({"directory": str(directory), "seed": config["seed"], **metrics})
    if len({row["seed"] for row in records}) != len(records):
        raise ValueError("Seed values must be distinct")
    names = sorted(set.intersection(*(set(row) for row in records)) - {"directory", "seed", "examples"})
    summary = {name: {"mean": float(np.mean([row[name] for row in records])),
                      "sample_std": float(np.std([row[name] for row in records], ddof=1)) if len(records) > 1 else None}
               for name in names}
    output = {"runs": records, "summary": summary}
    print(json.dumps(output, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
