"""Collect metrics and figures from real Earth benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_DATASETS = ("volcano", "earthquake", "flood", "fire")
DEFAULT_METHODS = ("rmgm", "rfm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="../../outputs/rmgm_real_earth")
    parser.add_argument("--summary-dir", default="../../outputs/rmgm_summary")
    parser.add_argument("--name", default="real_earth")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--select", choices=("last", "best_mmd2", "best_symmetric_nn"), default="best_mmd2")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths: dict[tuple[str, str], Path] = {}
    datasets = tuple(args.datasets)
    methods = tuple(args.methods)
    for data in datasets:
        for method in methods:
            run_dir = latest_run_dir(runs_root / data / method)
            if run_dir is None:
                rows.append({"data": data, "method": method, "status": "missing"})
                continue
            metrics = selected_metrics(run_dir, args.select)
            rows.append(
                {
                    "data": data,
                    "method": method,
                    "selection": args.select,
                    "run_dir": str(run_dir.resolve()),
                    **metrics,
                }
            )
            fig = selected_png(run_dir / "figs", data, method, metrics.get("step"))
            if fig is not None:
                image_paths[(data, method)] = fig

    out_json = summary_dir / f"{args.name}_metrics.json"
    out_md = summary_dir / f"{args.name}_metrics.md"
    out_png = summary_dir / f"{args.name}_contact_sheet.png"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out_md.write_text(markdown_table(rows, args.name), encoding="utf-8")
    write_contact_sheet(image_paths, out_png, datasets, methods)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_png}")


def latest_run_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_png(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = sorted(path.glob("*.png"))
    return candidates[-1] if candidates else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_metrics(run_dir: Path, selection: str) -> dict[str, Any]:
    if selection == "last":
        return read_json(run_dir / "metrics.json")

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return read_json(run_dir / "metrics.json")
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return {}
    key = "mmd2" if selection == "best_mmd2" else "symmetric_nn"
    return min(rows, key=lambda row: row.get(key, float("inf")))


def selected_png(path: Path, data: str, method: str, step: Any) -> Path | None:
    if path.exists() and isinstance(step, int):
        exact = path / f"{data}-{method}-{step:06d}.png"
        if exact.exists():
            return exact
    return latest_png(path)


def markdown_table(rows: list[dict[str, Any]], name: str) -> str:
    lines = [
        f"# {name.replace('_', ' ').title()} Benchmark",
        "",
        "Selection: best evaluated checkpoint by MMD2.",
        "",
        "| dataset | method | step | MMD2 | sym. NN | coverage | constraint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") == "missing":
            lines.append(f"| {row['data']} | {row['method']} | missing | | | | |")
            continue
        lines.append(
            "| {data} | {method} | {step} | {mmd2:.6g} | {symmetric_nn:.6g} | "
            "{coverage_at_radius:.3f} | {constraint_error:.3g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Metrics are computed against the held-out 10% split. Lower MMD2 and",
            "symmetric NN are better; higher coverage is better. R-MGM does not",
            "produce an exact likelihood, so this table is sample-quality based.",
            "",
        ]
    )
    return "\n".join(lines)


def write_contact_sheet(
    image_paths: dict[tuple[str, str], Path],
    out_path: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(
        len(datasets),
        len(methods),
        figsize=(5.5 * len(methods), 4.2 * len(datasets)),
        squeeze=False,
    )
    for i, data in enumerate(datasets):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            ax.axis("off")
            path = image_paths.get((data, method))
            ax.set_title(f"{data} / {method.upper()}")
            if path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            ax.imshow(mpimg.imread(path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
