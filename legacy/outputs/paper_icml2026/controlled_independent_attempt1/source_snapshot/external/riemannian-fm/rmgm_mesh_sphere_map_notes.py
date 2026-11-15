"""Practical sphere-map baseline notes for bunny/spot mesh R-MGM runs.

This file is intentionally a lightweight research note plus a smoke-test
utility, not a long training script. It proposes a minimal baseline for the
closed genus-0 mesh experiments in Flow Matching on General Geometries:

    1. Map mesh target samples x to directions u = Psi(x) in S^2.
    2. Train the existing sphere R-MGM generator on those u samples.
    3. Push generated sphere samples back to the mesh with Phi(u).
    4. Evaluate and visualize samples on the original mesh.

The baseline is useful for bunny10/50/100 and spot10/50/100. It is not a good
fit for maze3v2/maze4v2 because those meshes have boundaries and are not
sphere-like.

Why radial Phi?
---------------
Without adding heavy parameterization libraries, the simplest usable map is a
radial map from an interior center c:

    Psi(x) = normalize(x - c)
    Phi(u) = first intersection of ray c + t u with the triangle mesh.

For bunny/spot this is a practical baseline, but not a theorem-grade bijection:
thin parts, concavities, legs/ears, and any non-star-shaped region can create
folds or hidden surface patches. That caveat must be reported in results.

Minimal experiment today
------------------------
Use this file to smoke-test the map, then train S^2 R-MGM on Psi(data):

    python rmgm_mesh_sphere_map_notes.py \
        --obj data/mesh/bunny_simp.obj \
        --data data/mesh/bunny_simple.npy \
        --out ../../outputs/mesh_sphere_map_smoke/bunny_simple

For bunny10/50/100 and spot10/50/100, first create the official target samples:

    python rmgm_prepare_mesh_data.py --targets bunny10 bunny50 bunny100 spot10 spot50 spot100

The actual MGM training hook should be a small follow-up script:

    - load MeshDataset data x;
    - transform target to u = Psi(x);
    - use Sphere(), SphereGenerator, TangentCritic, critic_loss/generator_loss;
    - save generated u samples;
    - report Phi(u) as mesh samples.

This avoids differentiating through Phi. A stronger variant can put Phi inside
the generator and train with the existing mesh local midpoint surrogate, but
then the nearest triangle / ray hit selection is only piecewise differentiable.

Before training, sweep the small map choices on held-out target samples:

    center_mode in {bbox, mean}
    choose in {first, last}

Pick the setting with the best roundtrip Chamfer/L2 and report the hit
fraction. This matters because bunny/spot are sphere-like topologically, but
not perfectly star-shaped from one center.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


Choice = Literal["first", "last"]


@dataclass
class RadialSphereMeshMap:
    """Radial S^2 <-> mesh map based only on vertices, faces, and ray hits."""

    vertices: torch.Tensor
    faces: torch.Tensor
    center: torch.Tensor
    eps: float = 1e-8

    @classmethod
    def from_obj(
        cls,
        obj_path: str | Path,
        *,
        center_mode: Literal["bbox", "mean"] = "bbox",
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "RadialSphereMeshMap":
        vertices_np, faces_np = read_obj_vertices_faces(obj_path)
        vertices = torch.as_tensor(vertices_np, dtype=dtype, device=device)
        faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
        if center_mode == "bbox":
            center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
        elif center_mode == "mean":
            center = vertices.mean(dim=0)
        else:
            raise ValueError(f"Unknown center_mode: {center_mode}")
        return cls(vertices=vertices, faces=faces, center=center)

    def mesh_to_sphere(self, x: torch.Tensor) -> torch.Tensor:
        """Map mesh/ambient points to S^2 directions."""

        x = x.to(device=self.vertices.device, dtype=self.vertices.dtype)
        return normalize(x - self.center, eps=self.eps)

    def sphere_to_mesh(
        self,
        u: torch.Tensor,
        *,
        choose: Choice = "first",
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map S^2 directions to mesh points by radial ray intersection.

        Returns:
            points: mapped mesh points, with nearest-direction vertex fallback
                when no robust ray hit is found.
            hit_mask: boolean mask indicating true ray-triangle intersections.
        """

        u = normalize(u.to(device=self.vertices.device, dtype=self.vertices.dtype))
        return radial_ray_intersections(
            self.center,
            u,
            self.vertices,
            self.faces,
            choose=choose,
            chunk_size=chunk_size,
            eps=self.eps,
        )


def read_obj_vertices_faces(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read v/f records from a triangle OBJ without igl/trimesh."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z, *_ = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                idx = [int(tok.split("/")[0]) - 1 for tok in tokens]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    root = idx[0]
                    for i in range(1, len(idx) - 1):
                        faces.append([root, idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"No vertices/faces found in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def radial_ray_intersections(
    origin: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    choose: Choice = "first",
    chunk_size: int = 128,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Moller-Trumbore ray/triangle intersections in chunks."""

    tri = vertices[faces]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    tvec = origin.reshape(1, 3) - v0
    vertex_dirs = normalize(vertices - origin.reshape(1, 3), eps=eps)

    points = []
    masks = []
    for dirs in torch.split(directions, chunk_size):
        pvec = torch.cross(dirs[:, None, :].expand(-1, e2.shape[0], -1), e2[None], dim=-1)
        det = (e1[None] * pvec).sum(dim=-1)
        valid_det = det.abs() > eps
        safe_det = torch.where(valid_det, det, torch.ones_like(det))
        inv_det = torch.where(valid_det, 1.0 / safe_det, torch.zeros_like(det))

        u_bary = (tvec[None] * pvec).sum(dim=-1) * inv_det
        qvec = torch.cross(tvec[None].expand(dirs.shape[0], -1, -1), e1[None], dim=-1)
        v_bary = (dirs[:, None, :] * qvec).sum(dim=-1) * inv_det
        t = (e2[None] * qvec).sum(dim=-1) * inv_det

        hit = valid_det & (u_bary >= -eps) & (v_bary >= -eps)
        hit = hit & (u_bary + v_bary <= 1.0 + eps) & (t > eps)
        if choose == "first":
            score = torch.where(hit, t, torch.full_like(t, float("inf")))
            best_t, _ = score.min(dim=1)
        elif choose == "last":
            score = torch.where(hit, t, torch.full_like(t, -float("inf")))
            best_t, _ = score.max(dim=1)
        else:
            raise ValueError(f"Unknown choose: {choose}")

        found = torch.isfinite(best_t)
        mapped = origin.reshape(1, 3) + best_t[:, None] * dirs
        if (~found).any():
            fallback_idx = (dirs[~found] @ vertex_dirs.T).argmax(dim=1)
            mapped[~found] = vertices[fallback_idx]
        points.append(mapped)
        masks.append(found)

    return torch.cat(points, dim=0), torch.cat(masks, dim=0)


@torch.no_grad()
def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, max_points: int = 2048) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0], max_points)
    x = x[:n]
    y = y[:n]
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    positive = d2[d2 > 0]
    bandwidth2 = torch.median(positive).clamp_min(1e-6)
    gamma = 1.0 / bandwidth2
    return (
        torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
        + torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
        - 2.0 * torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    )


@torch.no_grad()
def symmetric_chamfer(x: torch.Tensor, y: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0], max_points)
    dist = torch.cdist(x[:n], y[:n])
    return dist.min(dim=0).values.mean() + dist.min(dim=1).values.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test radial S^2 -> mesh map.")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--center-mode", choices=("bbox", "mean"), default="bbox")
    parser.add_argument("--choose", choices=("first", "last"), default="first")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sphere_map = RadialSphereMeshMap.from_obj(args.obj, center_mode=args.center_mode)
    data = torch.as_tensor(np.load(args.data), dtype=sphere_map.vertices.dtype)
    data = data[: args.n]
    u = sphere_map.mesh_to_sphere(data)
    recon, hit = sphere_map.sphere_to_mesh(u, choose=args.choose)

    metrics = {
        "n": int(data.shape[0]),
        "ray_hit_fraction": float(hit.float().mean()),
        "roundtrip_mmd2": float(rbf_mmd2(recon, data)),
        "roundtrip_symmetric_chamfer": float(symmetric_chamfer(recon, data)),
        "roundtrip_l2_mean": float((recon - data).norm(dim=-1).mean()),
        "roundtrip_l2_q90": float(torch.quantile((recon - data).norm(dim=-1), 0.90)),
        "center_mode": args.center_mode,
        "choose": args.choose,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_smoke_plot(args.out / "sphere_map_roundtrip.png", data, recon, u)
    print(json.dumps(metrics, indent=2))


def save_smoke_plot(path: Path, data: torch.Tensor, recon: torch.Tensor, u: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 4.2), dpi=180)
    titles = ["mesh data", "Phi(Psi(data))", "Psi(data) on S^2"]
    arrays = [data.cpu().numpy(), recon.cpu().numpy(), u.cpu().numpy()]
    colors = ["black", "#3569b8", "#b85c38"]
    for i, (title, arr, color) in enumerate(zip(titles, arrays, colors), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=1.4, color=color, alpha=0.35, linewidths=0)
        ax.set_title(title)
        set_equal_3d(ax, arr)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def set_equal_3d(ax, arr: np.ndarray) -> None:
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float((hi - lo).max())
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    main()
