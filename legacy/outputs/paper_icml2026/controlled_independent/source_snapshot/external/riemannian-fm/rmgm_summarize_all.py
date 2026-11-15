"""Summarize completed R-MGM/RFM pilot runs and build contact sheets."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SUMMARY_DIR = OUTPUTS / "rmgm_summary"


ROOTS = [
    ("earth_real", OUTPUTS / "rmgm_real_earth"),
    ("torus_real", OUTPUTS / "rmgm_real_torus"),
    ("extra_pilot", OUTPUTS / "rmgm_extra_pilot"),
    ("scaling_pilot", OUTPUTS / "rmgm_scaling_pilot"),
]


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for family, root in ROOTS:
        runs.extend(load_runs(family, root))

    best = select_best_runs(runs)
    payload = {"runs": runs, "best": best}
    (SUMMARY_DIR / "final_reproduction_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (SUMMARY_DIR / "final_reproduction_status.md").write_text(
        render_markdown(best), encoding="utf-8"
    )
    build_contact_sheet(
        [row for row in best if row["family"] in {"extra_pilot", "scaling_pilot"}],
        SUMMARY_DIR / "extra_scaling_contact_sheet.png",
    )
    build_contact_sheet(
        [row for row in best if row["family"] in {"earth_real", "torus_real"}],
        SUMMARY_DIR / "real_data_contact_sheet_latest.png",
    )
    print(SUMMARY_DIR / "final_reproduction_status.md")
    print(SUMMARY_DIR / "extra_scaling_contact_sheet.png")
    print(SUMMARY_DIR / "real_data_contact_sheet_latest.png")


def load_runs(family: str, root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for metrics_json in root.rglob("metrics.json"):
        run_dir = metrics_json.parent
        config_path = run_dir / "config.json"
        jsonl_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not jsonl_path.exists():
            continue
        try:
            config_doc = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        if not metrics:
            continue

        cfg = config_doc.get("config", {})
        final = metrics[-1]
        best_mmd = min(metrics, key=lambda item: item.get("mmd2", float("inf")))
        best_nn = min(metrics, key=lambda item: item.get("symmetric_nn", float("inf")))
        best_cov = max(metrics, key=lambda item: item.get("coverage_at_radius", float("-inf")))
        dim = int(config_doc.get("dim", cfg.get("wrapped_dim") or 0) or 0)
        data = str(cfg.get("data", final.get("data", run_dir.parents[1].name)))
        method = str(cfg.get("method", final.get("method", run_dir.parents[0].name)))
        plot_path = latest_plot(run_dir)
        rows.append(
            {
                "family": family,
                "data": data,
                "method": method,
                "dim": dim,
                "steps": int(cfg.get("steps", final.get("step", 0)) or 0),
                "batch_size": int(cfg.get("batch_size", 0) or 0),
                "ode_steps": int(cfg.get("ode_steps", 0) or 0),
                "generator_type": cfg.get("generator_type", ""),
                "torus_generator": cfg.get("torus_generator", ""),
                "components": cfg.get("generator_components", cfg.get("torus_components", "")),
                "best_mmd2": float(best_mmd.get("mmd2", math.nan)),
                "best_mmd_step": int(best_mmd.get("step", 0) or 0),
                "final_mmd2": float(final.get("mmd2", math.nan)),
                "best_symmetric_nn": float(best_nn.get("symmetric_nn", math.nan)),
                "final_symmetric_nn": float(final.get("symmetric_nn", math.nan)),
                "best_coverage": float(best_cov.get("coverage_at_radius", math.nan)),
                "final_coverage": float(final.get("coverage_at_radius", math.nan)),
                "constraint_error": float(final.get("constraint_error", math.nan)),
                "run_dir": str(run_dir),
                "plot": str(plot_path) if plot_path else "",
                "mesh_note": config_doc.get("mesh_note"),
                "wrapped": config_doc.get("wrapped"),
            }
        )
    return rows


def select_best_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        key = (row["family"], row["data"], row["dim"], row["method"])
        grouped[key].append(row)
    best = [min(rows, key=lambda row: row["best_mmd2"]) for rows in grouped.values()]
    return sorted(best, key=lambda row: (row["family"], row["data"], row["dim"], row["method"]))


def latest_plot(run_dir: Path) -> Path | None:
    figs = run_dir / "figs"
    if not figs.exists():
        return None
    plots = sorted(figs.glob("*.png"), key=lambda p: p.stat().st_mtime)
    return plots[-1] if plots else None


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Final R-MGM / RFM Reproduction Status",
        "",
        "This report aggregates completed local GPU runs only. Lower is better for `mmd2` and `symmetric_nn`; higher is better for `coverage_at_radius`.",
        "",
        "Important caveat: official RFM author-budget training is audited but not completed here. Earth/Torus author configs require 1,000,000 steps; mesh/appendix official configs require 50,000-100,000 steps. The RFM rows below are the shared sample-quality baseline runs from `rmgm_train_real.py`, not the full Hydra likelihood runs from `train.py`.",
        "",
    ]
    for family in ["earth_real", "torus_real", "extra_pilot", "scaling_pilot"]:
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            continue
        lines.append(f"## {family}")
        lines.append("")
        lines.append(
            "| data | dim | method | steps | best mmd2 | final mmd2 | best symNN | final symNN | best coverage | constraint | run |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in subset:
            lines.append(
                "| {data} | {dim} | {method} | {steps} | {best_mmd2:.6g} | {final_mmd2:.6g} | {best_symmetric_nn:.6g} | {final_symmetric_nn:.6g} | {best_coverage:.6g} | {constraint_error:.3g} | `{run}` |".format(
                    data=row["data"],
                    dim=row["dim"],
                    method=row["method"],
                    steps=row["steps"],
                    best_mmd2=row["best_mmd2"],
                    final_mmd2=row["final_mmd2"],
                    best_symmetric_nn=row["best_symmetric_nn"],
                    final_symmetric_nn=row["final_symmetric_nn"],
                    best_coverage=row["best_coverage"],
                    constraint_error=row["constraint_error"],
                    run=Path(row["run_dir"]).name,
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Official RFM Audit",
            "",
            "- Resolved author configs are in `outputs/rmgm_summary/original_rfm_config_audit.md`.",
            "- Mesh dependency/data status is in `outputs/rmgm_summary/mesh_pipeline_status.md`.",
            "- Extra-geometry command audit is in `outputs/rmgm_summary/extra_geometry_plan.md`.",
            "- Boundary mesh arrays generated locally: `bunny_simple.npy`, `maze_3x3v2.npz`, `maze_4x4v2.npz`.",
            "- Full official RFM mesh smoke through `train.py` reached validation but was stopped after 10 minutes because validation over the full 10k split dominates even at one training step.",
            "",
        ]
    )
    return "\n".join(lines)


def build_contact_sheet(rows: list[dict[str, Any]], out_path: Path) -> None:
    if Image is None:
        return
    items = [(row, Path(row["plot"])) for row in rows if row.get("plot") and Path(row["plot"]).exists()]
    if not items:
        return
    thumb_w, thumb_h = 420, 320
    pad = 20
    label_h = 54
    cols = 3
    rows_count = math.ceil(len(items) / cols)
    canvas = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, rows_count * (thumb_h + label_h) + (rows_count + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (row, image_path) in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + label_h + pad)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            ox = x + (thumb_w - img.width) // 2
            oy = y + label_h + (thumb_h - img.height) // 2
            canvas.paste(img, (ox, oy))
        label = f"{row['data']} d={row['dim']} {row['method']} | best MMD {row['best_mmd2']:.3g}"
        draw.text((x, y), label, fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


if __name__ == "__main__":
    main()
