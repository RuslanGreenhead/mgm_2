"""Summarize real-data R-MGM/RFM sample metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows = []
    for root in args.roots:
        rows.extend(scan_root(root))
    if not rows:
        raise SystemExit("No metrics.jsonl files found.")

    summary = summarize(rows)
    print_table(summary)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


def scan_root(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in root.rglob("metrics.jsonl"):
        parts = metrics_path.relative_to(root).parts
        data = parts[0] if len(parts) >= 4 else "unknown"
        method = parts[1] if len(parts) >= 4 else "unknown"
        run_dir = metrics_path.parent
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["data"] = data
            row["method"] = method
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["data"], row["method"]), []).append(row)

    summary = []
    for (data, method), group in sorted(groups.items()):
        mmd_group = [row for row in group if finite_metric(row, "mmd2")]
        nn_group = [row for row in group if finite_metric(row, "symmetric_nn")]
        if not mmd_group or not nn_group:
            continue
        best_mmd = min(mmd_group, key=lambda r: r.get("mmd2", float("inf")))
        best_nn = min(nn_group, key=lambda r: r.get("symmetric_nn", float("inf")))
        final = max(nn_group, key=lambda r: r.get("step", -1))
        summary.append(
            {
                "data": data,
                "method": method,
                "best_mmd2": best_mmd.get("mmd2"),
                "best_mmd2_step": best_mmd.get("step"),
                "best_mmd2_run": best_mmd.get("run_dir"),
                "best_symmetric_nn": best_nn.get("symmetric_nn"),
                "best_symmetric_nn_step": best_nn.get("step"),
                "best_symmetric_nn_run": best_nn.get("run_dir"),
                "final_mmd2": final.get("mmd2"),
                "final_symmetric_nn": final.get("symmetric_nn"),
                "final_step": final.get("step"),
            }
        )
    return summary


def finite_metric(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def print_table(summary: list[dict[str, Any]]) -> None:
    headers = [
        "data",
        "method",
        "best_mmd2",
        "mmd_step",
        "best_sym_nn",
        "nn_step",
        "final_mmd2",
        "final_sym_nn",
    ]
    print("\t".join(headers))
    for row in summary:
        print(
            "\t".join(
                [
                    str(row["data"]),
                    str(row["method"]),
                    fmt(row["best_mmd2"]),
                    str(row["best_mmd2_step"]),
                    fmt(row["best_symmetric_nn"]),
                    str(row["best_symmetric_nn_step"]),
                    fmt(row["final_mmd2"]),
                    fmt(row["final_symmetric_nn"]),
                ]
            )
        )


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
