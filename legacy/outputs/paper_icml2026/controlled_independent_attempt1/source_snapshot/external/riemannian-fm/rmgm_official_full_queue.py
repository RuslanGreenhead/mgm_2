"""Run full-budget official RFM experiments and compare to current R-MGM pilots.

The queue executes the original ``train.py`` from a clean worktree. It only
passes non-algorithmic overrides for local paths, run directories, seed, and
wandb disabling. Iteration counts come from the experiment YAML files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCHED_RFM_REPO = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_REPO = PROJECT_ROOT / "external" / "riemannian-fm-official-clean"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rfm_official_full"
DEFAULT_PYTHON = PROJECT_ROOT / ".conda" / "rmgm" / "python.exe"
DEFAULT_CONDA_ENV = PROJECT_ROOT / ".conda" / "rmgm"
DATA_ROOT = PROJECT_ROOT / "external" / "riemannian-fm" / "data"


@dataclass(frozen=True)
class Job:
    label: str
    experiment: str
    group: str
    expected_steps: int
    overrides: dict[str, str] = field(default_factory=dict)
    required_files: tuple[Path, ...] = ()


def available_jobs() -> list[Job]:
    mesh = DATA_ROOT / "mesh"
    return [
        Job("wrapped_spd_d3", "wrapped_spd", "appendix_highdim", 50_000, {"wrapped.dim": "3"}),
        Job("wrapped_spd_d6", "wrapped_spd", "appendix_highdim", 50_000, {"wrapped.dim": "6"}),
        Job("wrapped_spd_d10", "wrapped_spd", "appendix_highdim", 50_000, {"wrapped.dim": "10"}),
        Job("wrapped_torus_d5", "wrapped_torus", "appendix_highdim", 50_000, {"wrapped.dim": "5"}),
        Job("wrapped_torus_d10", "wrapped_torus", "appendix_highdim", 50_000, {"wrapped.dim": "10"}),
        Job("wrapped_torus_d20", "wrapped_torus", "appendix_highdim", 50_000, {"wrapped.dim": "20"}),
        Job("hyperbolic", "hyperbolic", "appendix", 100_000),
        Job("simple_bunny", "simple_bunny", "mesh_boundary", 50_000, required_files=(mesh / "bunny_simple.npy",)),
        Job("maze3v2", "maze3v2", "mesh_boundary", 100_000, required_files=(mesh / "maze_3x3v2.npz",)),
        Job("maze4v2", "maze4v2", "mesh_boundary", 100_000, required_files=(mesh / "maze_4x4v2.npz",)),
        Job("volcano", "volcano", "earth_real", 1_000_000),
        Job("earthquake", "earthquake", "earth_real", 1_000_000),
        Job("fire", "fire", "earth_real", 1_000_000),
        Job("flood", "flood", "earth_real", 1_000_000),
        Job("general", "general", "torus_real", 1_000_000),
        Job("glycine", "glycine", "torus_real", 1_000_000),
        Job("proline", "proline", "torus_real", 1_000_000),
        Job("prepro", "prepro", "torus_real", 1_000_000),
        Job("rna", "rna", "torus_real", 1_000_000),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", default=str(DEFAULT_OFFICIAL_REPO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    output_root = Path(args.output_root).resolve()
    python = Path(args.python).resolve()
    status_path = output_root / f"official_rfm_seed{args.seed}_status.jsonl"
    manifest_path = output_root / f"official_rfm_seed{args.seed}_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    selected = available_jobs()
    if args.only:
        labels = set(args.only)
        selected = [job for job in selected if job.label in labels or job.experiment in labels]

    manifest = {
        "created_at": now(),
        "official_repo": str(official_repo),
        "output_root": str(output_root),
        "python": str(python),
        "seed": args.seed,
        "jobs": [job_to_dict(job, args.seed, output_root) for job in selected],
        "notes": [
            "Iteration counts are inherited from the original YAML configs.",
            "Overrides are limited to seed, local datadirs, hydra.run.dir, wandb disabling, and optional wrapped.dim variants.",
            "Runs are serialized on one GPU to avoid memory contention.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)

    for job in selected:
        run_one(job, args, official_repo, output_root, python, status_path)
        write_comparison(output_root, args.seed)

    write_status(status_path, {"event": "queue_finished", "time": now()})
    write_comparison(output_root, args.seed)


def run_one(
    job: Job,
    args: argparse.Namespace,
    official_repo: Path,
    output_root: Path,
    python: Path,
    status_path: Path,
) -> None:
    run_dir = output_root / f"seed{args.seed}" / job.label / "train"
    eval_dir = output_root / f"seed{args.seed}" / job.label / "official_eval"
    log_path = output_root / f"seed{args.seed}" / job.label / "train.log"
    missing = [str(path) for path in job.required_files if not path.exists()]
    if missing:
        write_status(
            status_path,
            {
                "event": "skipped_missing_data",
                "time": now(),
                "label": job.label,
                "experiment": job.experiment,
                "missing": missing,
            },
        )
        return

    metrics_path = run_dir / "metrics.json"
    complete = metrics_path.exists() and not args.force
    if complete:
        write_status(status_path, completed_event(job, run_dir, skipped=True))
        maybe_evaluate(job, args, official_repo, output_root, python, status_path)
        return

    cmd = train_command(job, args.seed, run_dir, python)
    write_status(
        status_path,
        {
            "event": "dry_run" if args.dry_run else "started",
            "time": now(),
            "label": job.label,
            "experiment": job.experiment,
            "expected_steps": job.expected_steps,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "cmd": cmd[1:],
        },
    )
    if args.dry_run:
        return

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = run_env()
    started = datetime.now(timezone.utc)
    with log_path.open("ab") as log_file:
        log_file.write(("\n\n===== START " + now() + " =====\n").encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=official_repo,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    event = completed_event(job, run_dir, skipped=False)
    event.update(
        {
            "event": "completed" if return_code == 0 else "failed",
            "time": now(),
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "log": str(log_path),
        }
    )
    write_status(status_path, event)
    if return_code != 0 and elapsed < 5.0:
        write_status(
            status_path,
            {
                "event": "queue_aborted_after_fast_failure",
                "time": now(),
                "label": job.label,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "log": str(log_path),
            },
        )
        raise SystemExit(return_code)
    if return_code == 0:
        maybe_evaluate(job, args, official_repo, output_root, python, status_path)


def train_command(job: Job, seed: int, run_dir: Path, python: Path) -> list[str]:
    args = [
        str(python),
        "-B",
        "train.py",
        f"experiment={job.experiment}",
        f"seed={seed}",
        "use_wandb=False",
        f"earth_datadir={slash(DATA_ROOT / 'earth_data')}",
        f"top500_datadir={slash(DATA_ROOT / 'top500')}",
        f"rna_datadir={slash(DATA_ROOT / 'rna')}",
        f"mesh_datadir={slash(DATA_ROOT / 'mesh')}",
        f"hydra.run.dir={slash(run_dir)}",
    ]
    for key, value in job.overrides.items():
        args.append(f"{key}={value}")
    return args


def maybe_evaluate(
    job: Job,
    args: argparse.Namespace,
    official_repo: Path,
    output_root: Path,
    python: Path,
    status_path: Path,
) -> None:
    run_dir = output_root / f"seed{args.seed}" / job.label / "train"
    eval_dir = output_root / f"seed{args.seed}" / job.label / "official_eval"
    metrics_path = eval_dir / "sample_metrics.json"
    if metrics_path.exists() and not args.force:
        write_status(
            status_path,
            {
                "event": "evaluation_skipped_existing",
                "time": now(),
                "label": job.label,
                "eval_metrics": str(metrics_path),
            },
        )
        return
    ckpt = best_checkpoint(run_dir)
    if ckpt is None:
        write_status(
            status_path,
            {
                "event": "evaluation_skipped_no_checkpoint",
                "time": now(),
                "label": job.label,
                "run_dir": str(run_dir),
            },
        )
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_log = output_root / f"seed{args.seed}" / job.label / "eval.log"
    cmd = [
        str(python),
        "-B",
        str(PATCHED_RFM_REPO / "rmgm_eval_official_rfm.py"),
        "--official-repo",
        str(official_repo),
        "--checkpoint",
        str(ckpt),
        "--output",
        str(eval_dir),
        "--eval-samples",
        str(args.eval_samples),
        "--metric-samples",
        str(args.metric_samples),
        "--sample-batch-size",
        str(args.sample_batch_size),
    ]
    write_status(
        status_path,
        {
            "event": "evaluation_started",
            "time": now(),
            "label": job.label,
            "checkpoint": str(ckpt),
            "cmd": cmd[1:],
        },
    )
    env = run_env()
    with eval_log.open("ab") as log_file:
        log_file.write(("\n\n===== EVAL START " + now() + " =====\n").encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=PATCHED_RFM_REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return_code = proc.wait()
    write_status(
        status_path,
        {
            "event": "evaluation_completed" if return_code == 0 else "evaluation_failed",
            "time": now(),
            "label": job.label,
            "return_code": return_code,
            "eval_metrics": str(metrics_path),
            "eval_log": str(eval_log),
        },
    )


def best_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    nonlast = sorted(
        [path for path in ckpt_dir.glob("*.ckpt") if path.name != "last.ckpt"],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if nonlast:
        return nonlast[0]
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def completed_event(job: Job, run_dir: Path, skipped: bool) -> dict[str, Any]:
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metrics = {"parse_error": str(exc)}
    return {
        "event": "completed_existing" if skipped else "completed",
        "time": now(),
        "label": job.label,
        "experiment": job.experiment,
        "expected_steps": job.expected_steps,
        "run_dir": str(run_dir),
        "metrics_json": str(metrics_path) if metrics_path.exists() else "",
        "official_metrics": metrics,
    }


def write_comparison(output_root: Path, seed: int) -> None:
    current_path = PROJECT_ROOT / "outputs" / "rmgm_summary" / "final_reproduction_status.json"
    rows = []
    if current_path.exists():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
            for row in current.get("best", []):
                if row.get("method") == "rmgm":
                    rows.append({"source": "current_rmgm", **row})
        except Exception:
            pass

    official_rows = []
    for metrics_path in (output_root / f"seed{seed}").glob("*/official_eval/sample_metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = metrics_path.parents[1].name
        official_rows.append(
            {
                "label": label,
                "data": metrics.get("data"),
                "dim": metrics.get("dim"),
                "mmd2": metrics.get("mmd2"),
                "symmetric_nn": metrics.get("symmetric_nn"),
                "coverage_at_radius": metrics.get("coverage_at_radius"),
                "constraint_error": metrics.get("constraint_error"),
                "plot": str(metrics_path.parent / "official_rfm_samples.png"),
                "checkpoint": metrics.get("checkpoint"),
                "global_step": metrics.get("global_step"),
            }
        )

    current_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            current_by_key[(str(row["data"]), int(row["dim"]))] = row
        except Exception:
            continue

    lines = [
        "# Official RFM Full-Budget vs Current R-MGM",
        "",
        f"Updated: {now()}",
        "",
        "Official RFM rows are completed full-budget `train.py` runs from the clean RFM checkout. Current R-MGM rows are the best local pilot runs already present in `outputs/rmgm_summary/final_reproduction_status.json`.",
        "",
        "| label | data | dim | RFM step | official RFM mmd2 | current R-MGM best mmd2 | official RFM symNN | current R-MGM best symNN | official coverage | current R-MGM coverage | plot |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(official_rows, key=lambda item: item["label"]):
        cur = current_by_key.get((str(row["data"]), int(row["dim"] or 0)), {})
        lines.append(
            "| {label} | {data} | {dim} | {step} | {rfm_mmd} | {mgm_mmd} | {rfm_nn} | {mgm_nn} | {rfm_cov} | {mgm_cov} | {plot} |".format(
                label=row["label"],
                data=row["data"],
                dim=row["dim"],
                step=row.get("global_step", ""),
                rfm_mmd=fmt(row.get("mmd2")),
                mgm_mmd=fmt(cur.get("best_mmd2")),
                rfm_nn=fmt(row.get("symmetric_nn")),
                mgm_nn=fmt(cur.get("best_symmetric_nn")),
                rfm_cov=fmt(row.get("coverage_at_radius")),
                mgm_cov=fmt(cur.get("best_coverage")),
                plot=markdown_link("png", row.get("plot")),
            )
        )
    if not official_rows:
        lines.append("| _none completed yet_ | | | | | | | | | | |")
    out = output_root / f"seed{seed}" / "current_mgm_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_link(label: str, path: str | None) -> str:
    if not path:
        return ""
    return f"[{label}](<{path}>)"


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return ""


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GEOMSTATS_BACKEND"] = "pytorch"
    env["USER"] = env.get("USER") or "Alexander"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PATH"] = conda_path_prefix() + os.pathsep + env.get("PATH", "")
    return env


def conda_path_prefix() -> str:
    parts = [
        DEFAULT_CONDA_ENV,
        DEFAULT_CONDA_ENV / "Library" / "mingw-w64" / "bin",
        DEFAULT_CONDA_ENV / "Library" / "usr" / "bin",
        DEFAULT_CONDA_ENV / "Library" / "bin",
        DEFAULT_CONDA_ENV / "Scripts",
        DEFAULT_CONDA_ENV / "bin",
        DEFAULT_CONDA_ENV / "DLLs",
    ]
    return os.pathsep.join(str(path) for path in parts if path.exists())


def write_status(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False))
    sys.stdout.flush()


def job_to_dict(job: Job, seed: int, output_root: Path) -> dict[str, Any]:
    return {
        "label": job.label,
        "experiment": job.experiment,
        "group": job.group,
        "seed": seed,
        "expected_steps": job.expected_steps,
        "overrides": job.overrides,
        "required_files": [str(path) for path in job.required_files],
        "run_dir": str(output_root / f"seed{seed}" / job.label / "train"),
    }


def slash(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
