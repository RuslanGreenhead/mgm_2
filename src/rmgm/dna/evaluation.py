"""Published promoter metrics on matched, held-out conditions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def kmer_counts(labels: np.ndarray, k: int = 6) -> np.ndarray:
    """Pooled base-4 counts; windows containing unknown label 4 are omitted."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("Expected [N,L] integer DNA labels")
    windows = np.lib.stride_tricks.sliding_window_view(labels, k, axis=1)
    valid = np.all(windows < 4, axis=-1)
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    ids = (windows[valid].astype(np.int64) * powers).sum(axis=-1)
    return np.bincount(ids, minlength=4 ** k).astype(np.int64)


def kmer_pearson(real_counts: np.ndarray, generated_counts: np.ndarray) -> float:
    real_counts, generated_counts = np.asarray(real_counts), np.asarray(generated_counts)
    if real_counts.shape != generated_counts.shape or real_counts.sum() == 0 or generated_counts.sum() == 0:
        raise ValueError("Nonempty matching count vectors required")
    mask = (real_counts + generated_counts) > 0
    x = real_counts[mask] / real_counts.sum()
    y = generated_counts[mask] / generated_counts.sum()
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        raise ValueError("Pearson correlation is undefined for constant frequencies")
    return float(np.corrcoef(x, y)[0, 1])


def one_hot_with_unknown(labels: torch.Tensor) -> torch.Tensor:
    result = F.one_hot(labels.long().clamp_max(3), 4).float()
    return torch.where((labels == 4)[..., None], torch.full_like(result, 0.25), result)


class KGDMSeiMetric:
    """Use the vendored KGDM Sei implementation to preserve its H3K4me3 metric."""

    def __init__(self, data_directory: str | Path, source_directory: str | Path):
        source_directory = Path(source_directory).resolve()
        if not (source_directory / "src/eval/sei_eval.py").is_file():
            raise FileNotFoundError(source_directory / "src/eval/sei_eval.py")
        if str(source_directory) not in sys.path:
            sys.path.insert(0, str(source_directory))
        from src.eval.sei_eval import SeiEval  # type: ignore
        self.evaluator = SeiEval(data_dir=str(data_directory))

    @torch.no_grad()
    def squared_error(self, predicted: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
        return np.asarray(self.evaluator.eval_sp_mse(predicted, reference))
