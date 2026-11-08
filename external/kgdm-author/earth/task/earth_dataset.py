from __future__ import annotations
from typing import Tuple
from pathlib import Path
import os
from csv import reader
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split


def _load_csv(filename: str | os.PathLike) -> np.ndarray:
    with open(filename, "r") as f:
        rows = list(reader(f))
    return np.array(rows[1:], dtype=np.float32)


def cartesian_from_latlon_deg(x: torch.Tensor) -> torch.Tensor:
    """
    Convert latitude/longitude in degrees to 3D Cartesian coordinates.

    x: [..., 2] where x[..., 0] = lat, x[..., 1] = lon
    returns: [..., 3]
    """
    assert x.shape[-1] == 2
    lat = torch.deg2rad(x[..., 0])
    lon = torch.deg2rad(x[..., 1])
    xx = torch.cos(lat) * torch.cos(lon)
    yy = torch.cos(lat) * torch.sin(lon)
    zz = torch.sin(lat)
    return torch.stack([xx, yy, zz], dim=-1)


class ExpandDataset(Dataset):
    def __init__(self, dataset: Dataset, expand_factor: int = 1):
        self.dataset = dataset
        self.expand_factor = int(expand_factor)

    def __len__(self) -> int:
        return len(self.dataset) * self.expand_factor

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]


class EarthPoints(Dataset):
    def __init__(self, csv_path: str | os.PathLike, return_latlon: bool = False):
        arr = _load_csv(csv_path)
        latlon = torch.tensor(arr, dtype=torch.float32)
        xyz = cartesian_from_latlon_deg(latlon)
        self.latlon = latlon
        self.xyz = xyz
        self.return_latlon = return_latlon

    def __len__(self) -> int:
        return len(self.xyz)

    def __getitem__(self, idx: int):
        out = {"x": self.xyz[idx]}
        if self.return_latlon:
            out["latlon"] = self.latlon[idx]
        return out


class EarthDataset(Dataset):
    """
    Earth event dataset with the same split convention as RMF:
    80% train, 10% val, 10% test, fixed seed 42.

    Expected directory structure:
        data/earth_data/
            earthquake.csv
            volcano.csv
            fire.csv
            flood.csv
    """

    expand_factors = {
        "volcano": 1550,
        "earthquake": 210,
        "fire": 100,
        "flood": 260,
    }

    def __init__(
        self,
        data_dir: str | os.PathLike = "data/earth_data",
        dataset_file: str = "earthquake",
        split: str = "train",
        expand_train: bool = True,
    ):
        assert split in {"train", "val", "test", "valid"}
        split = "val" if split == "valid" else split

        csv_path = Path(data_dir) / f"{dataset_file}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find Earth dataset CSV at: {csv_path}")

        base = EarthPoints(csv_path)
        n_val = int(0.1 * len(base))
        n_test = int(0.1 * len(base))
        n_train = len(base) - n_val - n_test

        train_ds, val_ds, test_ds = random_split(
            base,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            if expand_train:
                expand = self.expand_factors.get(dataset_file, 1)
                self.data = ExpandDataset(train_ds, expand)
            else:
                self.data = train_ds
        elif split == "val":
            self.data = val_ds
        else:
            self.data = test_ds

        self.dataset_file = dataset_file
        self.data_dir = str(data_dir)
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def spherical_geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Stable geodesic distance on the sphere.
    a, b: [..., 3]
    returns: [...]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    cos = torch.sum(a * b, dim=-1)
    perp = a - cos.unsqueeze(-1) * b
    sin = torch.sqrt(torch.sum(perp**2, dim=-1) + 1e-8)
    theta = torch.atan2(sin, cos)
    return theta


def pairwise_spherical_geodesic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 3]
    y: [M, 3]
    returns: [N, M]
    """
    return spherical_geodesic_distance(x[:, None, :], y[None, :, :])


def compute_mmd(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> Tuple[float, float]:
    """
    Spherical RBF-kernel MMD using geodesic distance.
    x, y: [N, 3]
    """
    assert x.shape[0] == y.shape[0]

    with torch.no_grad():
        xx = pairwise_spherical_geodesic(x, x).pow(2)
        yy = pairwise_spherical_geodesic(y, y).pow(2)
        xy = pairwise_spherical_geodesic(x, y).pow(2)

        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)

        mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        mmd = float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())
        return mmd, float(mmd2.item())


@torch.no_grad()
def compute_nn_1_accuracy(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    1-NN two-sample test accuracy using geodesic distance.

    Pool real and generated samples, classify each point by the label of its nearest
    neighbor excluding itself. If distributions match well, accuracy should be near 50%.

    x: [N, 3] generated
    y: [N, 3] real
    """
    n = x.shape[0]
    assert y.shape[0] == n

    z = torch.cat([x, y], dim=0)  # [2N, 3]
    labels = torch.cat(
        [torch.zeros(n, device=z.device), torch.ones(n, device=z.device)], dim=0
    )  # 0=fake, 1=real

    D = pairwise_spherical_geodesic(z, z)  # [2N, 2N]
    D.fill_diagonal_(float("inf"))
    nn_idx = D.argmin(dim=1)
    pred = labels[nn_idx]
    acc = (pred == labels).float().mean()
    return float(acc.item())




def sinkhorn_uniform(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 200,
) -> float:
    """
    Entropic Sinkhorn OT cost for two uniform empirical measures.

    cost: [N, N]
    returns: scalar float
    """
    assert cost.ndim == 2 and cost.shape[0] == cost.shape[1]
    n = cost.shape[0]
    device = cost.device
    dtype = cost.dtype

    a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    b = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

    K = torch.exp(-cost / epsilon).clamp_min(1e-30)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        Kv = K @ v
        u = a / Kv.clamp_min(1e-30)

        KTu = K.t() @ u
        v = b / KTu.clamp_min(1e-30)

    P = u[:, None] * K * v[None, :]
    ot_cost = torch.sum(P * cost)
    return float(ot_cost.item())


@torch.no_grad()
def compute_sinkhorn_geo(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 200,
    p: float = 1.0,
) -> float:
    """
    Sinkhorn OT on the sphere using geodesic cost.

    x, y: [N, 3]
    p:
      - 1.0 -> cost = d
      - 2.0 -> cost = d^2
    """
    d = pairwise_spherical_geodesic(x, y)
    if p == 1.0:
        cost = d
    elif p == 2.0:
        cost = d.pow(2)
    else:
        cost = d.pow(p)
    return sinkhorn_uniform(cost, epsilon=epsilon, n_iters=n_iters)


@torch.no_grad()
def evaluate_earth_model(
    model: nn.Module,
    dataloader,
    device: str,
    gamma: float = 1.0,
    return_samples: bool = False,
    compute_nn1_metric: bool = True,
    compute_sinkhorn_metric: bool = True,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 200,
    sinkhorn_p: float = 1.0,
):
    model.eval()

    reals = []
    for batch in dataloader:
        reals.append(batch["x"].float().to(device))
    real_x = torch.cat(reals, dim=0)

    gen_x = model.sample(
        batch_size=real_x.shape[0],
        mode="sample",
        device=real_x.device,
    )

    metrics = {}

    mmd, mmd2 = compute_mmd(gen_x, real_x, gamma=gamma)
    metrics["mmd"] = mmd
    metrics["mmd2"] = mmd2

    if compute_nn1_metric:
        metrics["nn1_acc"] = compute_nn_1_accuracy(gen_x, real_x)

    if compute_sinkhorn_metric:
        metrics["sinkhorn_geo"] = compute_sinkhorn_geo(
            gen_x,
            real_x,
            epsilon=sinkhorn_epsilon,
            n_iters=sinkhorn_iters,
            p=sinkhorn_p,
        )

    model.train()

    if return_samples:
        return metrics, {
            "real_x": real_x.detach().cpu(),
            "gen_x": gen_x.detach().cpu(),
        }
    return metrics
