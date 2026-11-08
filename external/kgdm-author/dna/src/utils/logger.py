"""
Unbuffered, file-first logging for training and smoke runs.

Two loggers per run:
  - `run.log`    : human-readable text via stdlib `logging`. FlushFileHandler
                   flushes after every record so a `tail -f` (or post-mortem
                   read) sees everything that has been emitted, even mid-step.
  - `metrics.jsonl`: one JSON object per step. Line-buffered (`buffering=1`)
                   so each `write` flushes immediately. Easy to parse later
                   with `pandas.read_json(path, lines=True)` for plotting.

Console output (stdout) is mirrored so slurm `.out` files still show
progress, but the file logs are the source of truth — slurm `.out` is
heavily buffered and unreliable for live diagnosis.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def make_run_dir(root: str | Path, run_name: str) -> Path:
    """Create `<root>/<run_name>_<YYYYmmdd-HHMMSS>/` and return the path."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(
    logfile: str | Path,
    name: str = "promoter",
    level: int = logging.INFO,
    to_stdout: bool = True,
) -> logging.Logger:
    """Configure a named logger writing to `logfile` (line-flushed) and stdout."""
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent across re-imports
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = FlushFileHandler(logfile, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.propagate = False
    return logger


class JsonlWriter:
    """Append-mode, line-buffered JSON-lines writer.

    Each `write(**fields)` emits one JSON object and flushes (since we open
    with `buffering=1`). Safe to interleave with the text logger.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)

    def write(self, **fields) -> None:
        self.f.write(json.dumps(fields, sort_keys=True) + "\n")

    def close(self) -> None:
        try:
            self.f.flush()
        finally:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
