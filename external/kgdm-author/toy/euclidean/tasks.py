from typing import Optional

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll


def _checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    x1 = rng.uniform(-2.0, 2.0, size=n)
    x2_ = rng.uniform(0.0, 1.0, size=n) - rng.integers(0, 2, size=n) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.stack([x1, x2], axis=1) * 2.0


def _swiss_roll_2d(n: int, rng: np.random.Generator, noise: float = 0.5) -> np.ndarray:
    seed = int(rng.integers(0, 2**31 - 1))
    pts, _ = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    xy = pts[:, [0, 2]]
    return xy / 7.5


_SAMPLERS = {
    "checkerboard": _checkerboard,
    "swissroll": _swiss_roll_2d,
}


class Task:
    def __init__(self, name: str):
        if name not in _SAMPLERS:
            raise ValueError(f"unknown task '{name}', expected one of {list(_SAMPLERS)}")
        self.name = name
        self.dim = 2
        self._sampler = _SAMPLERS[name]

    def sample(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        x = self._sampler(n, rng)
        return torch.from_numpy(x).float()


def get_task(name: str) -> Task:
    return Task(name)


def available_tasks() -> list[str]:
    return list(_SAMPLERS)
