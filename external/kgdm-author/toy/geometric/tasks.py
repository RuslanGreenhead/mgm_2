"""Target distributions on S^2 and H^2.

Four combinations:
  checkerboard_sphere     — 4x4 black-square checkerboard in [-1,1]^2 wrapped to S^2.
  checkerboard_hyperboloid — same checkerboard wrapped to H^2 via exp-map at origin.
  swissroll_sphere        — spiral parametrized natively on S^2.
  swissroll_hyperboloid   — spiral parametrized natively on H^2.
"""
from __future__ import annotations

import math
import torch
from torch.utils import data

from manifolds.sphere import SphereManifold
from manifolds.hyperboloid import HyperboloidManifold


def checkerboard_2d(batch_size: int, device: str = "cpu", num_square: int = 4) -> torch.Tensor:
    """Points in [-1, 1]^2 sampled only on black squares of a num_square × num_square grid."""
    x = torch.rand(batch_size, 2, device=device) * num_square
    cell_x = torch.floor(x[:, 0]).to(torch.int)
    cell_y = torch.floor(x[:, 1]).to(torch.int)
    is_white = ((cell_x + cell_y) % 2 == 0)

    can_shift_right = (cell_x < num_square - 1)
    x[:, 0] = x[:, 0] + (is_white & can_shift_right).float()
    can_shift_down = (cell_y < num_square - 1)
    x[:, 1] = x[:, 1] + (is_white & ~can_shift_right & can_shift_down).float()
    is_corner = is_white & ~can_shift_right & ~can_shift_down
    x[:, 0] = x[:, 0] - (num_square - 1) * is_corner.float()

    return (x / num_square * 2.0 - 1.0).float()


def swiss_roll_sphere(
    batch_size: int,
    device: str = "cpu",
    n_turns: float = 3.0,
    width: float = 0.1,
    z_min: float = -0.9,
    z_max: float = 0.9,
) -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    z_center = z_max + (z_min - z_max) * u
    theta = 2.0 * math.pi * n_turns * u
    z = z_center + width * (torch.rand(batch_size, device=device) - 0.5)
    z = torch.clamp(z, -0.999, 0.999)
    r_xy = torch.sqrt(1.0 - z**2)
    return torch.stack([r_xy * torch.cos(theta), r_xy * torch.sin(theta), z], dim=1).float()


def swiss_roll_hyperboloid(
    batch_size: int,
    device: str = "cpu",
    n_turns: float = 3.0,
    r_max: float = 1.5,
    width: float = 0.05,
) -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    r_center = r_max * u
    theta = 2.0 * math.pi * n_turns * u
    r = (r_center + width * (torch.rand(batch_size, device=device) - 0.5)).clamp(min=0.0)
    sinh_r = torch.sinh(r)
    return torch.stack(
        [torch.cosh(r), sinh_r * torch.cos(theta), sinh_r * torch.sin(theta)], dim=1
    ).float()


class _TensorDataset(data.Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.data = tensor

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


DATASETS = ("checkerboard_sphere", "checkerboard_hyperboloid", "swissroll_sphere", "swissroll_hyperboloid")
MANIFOLDS = ("sphere", "hyperboloid")


def make_dataset(name: str, n: int, device: str = "cpu") -> data.Dataset:
    if name == "checkerboard_sphere":
        return _TensorDataset(SphereManifold().wrap(checkerboard_2d(n, device)))
    if name == "checkerboard_hyperboloid":
        return _TensorDataset(HyperboloidManifold().wrap(checkerboard_2d(n, device)))
    if name == "swissroll_sphere":
        return _TensorDataset(swiss_roll_sphere(n, device))
    if name == "swissroll_hyperboloid":
        return _TensorDataset(swiss_roll_hyperboloid(n, device))
    raise ValueError(f"unknown dataset: {name!r}; pick one of {DATASETS}")


def manifold_for_dataset(name: str) -> str:
    if name.endswith("_sphere"):
        return "sphere"
    if name.endswith("_hyperboloid"):
        return "hyperboloid"
    raise ValueError(name)


def get_manifold(name: str):
    if name == "sphere":
        return SphereManifold()
    if name == "hyperboloid":
        return HyperboloidManifold()
    raise ValueError(f"unknown manifold: {name!r}; pick one of {MANIFOLDS}")
