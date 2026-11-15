"""Start the official RFM full-budget queue as a detached background process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = PROJECT_ROOT / ".conda" / "rmgm" / "python.exe"
DEFAULT_QUEUE = PROJECT_ROOT / "external" / "riemannian-fm" / "rmgm_official_full_queue.py"
DEFAULT_OFFICIAL_REPO = PROJECT_ROOT / "external" / "riemannian-fm-official-clean"
DEFAULT_WORKDIR = PROJECT_ROOT / "external" / "riemannian-fm"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rfm_official_full"
DEFAULT_CONDA_ENV = PROJECT_ROOT / ".conda" / "rmgm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--official-repo", default=str(DEFAULT_OFFICIAL_REPO))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--metric-samples", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "queue_stdout.log"
    stderr_path = output_root / "queue_stderr.log"
    pid_path = output_root / "queue_process.json"

    cmd = [
        str(Path(args.python).resolve()),
        "-B",
        str(Path(args.queue).resolve()),
        "--official-repo",
        str(Path(args.official_repo).resolve()),
        "--output-root",
        str(output_root),
        "--seed",
        str(args.seed),
        "--eval-samples",
        str(args.eval_samples),
        "--metric-samples",
        str(args.metric_samples),
        "--sample-batch-size",
        str(args.sample_batch_size),
    ]
    if args.only:
        cmd.append("--only")
        cmd.extend(args.only)

    env = os.environ.copy()
    env["GEOMSTATS_BACKEND"] = "pytorch"
    env["USER"] = env.get("USER") or "Alexander"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PATH"] = conda_path_prefix() + os.pathsep + env.get("PATH", "")

    creationflags = 0

    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(args.workdir).resolve()),
        env=env,
        stdout=stdout,
        stderr=stderr,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
    payload = {
        "pid": proc.pid,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cmd": cmd,
        "cwd": str(Path(args.workdir).resolve()),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    pid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


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


if __name__ == "__main__":
    main()
