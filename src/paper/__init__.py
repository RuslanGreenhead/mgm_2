"""Compatibility import path for the relocated 2026 research code."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "legacy" / "paper")]
