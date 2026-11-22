"""Compact prepared promoter arrays and paired sequence/signal cropping."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


LENGTH = 1024
FLANK = 10


def verify_prepared(directory: str | Path) -> dict:
    """Check prepared array hashes before a run or cluster transfer."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["processed_files"].items():
        digest = hashlib.sha256()
        with (directory / name).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Prepared file checksum mismatch: {name}")
    return manifest


class PromoterArrays(Dataset):
    """Read memory-mapped arrays; a train crop shifts both modalities together."""

    def __init__(self, directory: str | Path, split: str, shift: int = 0, seed: int = 0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.split = split
        self.shift = shift
        if shift < 0 or shift > self.manifest["flank"]:
            raise ValueError("shift must fit in the prepared flank")
        if split != "train" and shift:
            raise ValueError("validation and test crops must be centered")
        self.labels = np.load(self.directory / f"{split}_labels.npy", mmap_mode="r")
        self.signal = np.load(self.directory / f"{split}_signal.npy", mmap_mode="r")
        if self.labels.shape != self.signal.shape or self.labels.shape[1] != LENGTH + 2 * self.manifest["flank"]:
            raise ValueError("Prepared array shapes are inconsistent")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        # Stateless per-record shift makes a resumed epoch reproduce its crops.
        crop_seed = (self.seed * 1000003 + self.epoch * 9176 + index * 6361) & ((1 << 64) - 1)
        delta = random.Random(crop_seed).randint(-self.shift, self.shift) if self.shift else 0
        start = self.manifest["flank"] + delta
        labels = np.array(self.labels[index, start:start + LENGTH], dtype=np.int64)
        signal = np.array(self.signal[index, start:start + LENGTH], dtype=np.float32)
        return torch.from_numpy(labels), torch.from_numpy(signal[:, None])
