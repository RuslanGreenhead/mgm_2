"""Make large side-by-side comparison panels from selected real-run figures."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()

    rows = json.loads(args.metrics.read_text(encoding="utf-8"))
    by_data: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_data.setdefault(row["data"], {})[row["method"]] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for data, methods in by_data.items():
        if "rmgm" not in methods or "rfm" not in methods:
            continue
        out_path = args.output_dir / f"{data}_rmgm_vs_rfm.png"
        write_pair(data, methods["rmgm"], methods["rfm"], out_path, args.dpi)
        written.append(out_path)

    print("\n".join(str(path.resolve()) for path in written))


def write_pair(data: str, rmgm: dict[str, Any], rfm: dict[str, Any], out_path: Path, dpi: int) -> None:
    rmgm_png = choose_png(Path(rmgm["run_dir"]) / "figs", data, "rmgm", int(rmgm["step"]))
    rfm_png = choose_png(Path(rfm["run_dir"]) / "figs", data, "rfm", int(rfm["step"]))
    if rmgm_png is None or rfm_png is None:
        raise FileNotFoundError(f"Missing comparison PNG for {data}")

    left = mpimg.imread(rmgm_png)
    right = mpimg.imread(rfm_png)
    aspect = max(left.shape[0] / left.shape[1], right.shape[0] / right.shape[1])
    fig_w = 18.0
    fig_h = max(6.2, fig_w * aspect / 2.0 + 1.1)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), squeeze=False)
    for ax, img, row, label in [
        (axes[0, 0], left, rmgm, "R-MGM"),
        (axes[0, 1], right, rfm, "RFM"),
    ]:
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            (
                f"{data} / {label}\n"
                f"MMD2={row['mmd2']:.4g}, symNN={row['symmetric_nn']:.4g}, "
                f"coverage={row['coverage_at_radius']:.3f}, step={row['step']}"
            ),
            fontsize=15,
            pad=10,
        )
    fig.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def choose_png(figs_dir: Path, data: str, method: str, step: int) -> Path | None:
    exact = figs_dir / f"{data}-{method}-{step:06d}.png"
    if exact.exists():
        return exact
    candidates = sorted(figs_dir.glob("*.png"))
    if not candidates:
        return None
    return min(candidates, key=lambda path: abs(step_from_png(path) - step))


def step_from_png(path: Path) -> int:
    match = re.search(r"-(\d{6})\.png$", path.name)
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    main()
