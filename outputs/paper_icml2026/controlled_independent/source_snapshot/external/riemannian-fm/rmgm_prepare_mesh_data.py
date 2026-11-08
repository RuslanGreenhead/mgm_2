"""Prepare mesh datasets for the RFM/R-MGM experiments.

The upstream ``data/synthesize_mesh_data.py`` assumes it is executed from the
``data`` directory and writes through relative ``mesh/...`` paths. In the Codex
workspace sandbox those relative writes can be denied on Windows, while
absolute writes from the project root are allowed. This wrapper keeps the same
generation logic but resolves every input/output path explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import igl
import numpy as np
import torch

from manifm.dist import GaussianMM
from manifm.manifolds.mesh import Mesh, sample_simplex_uniform


ROOT = Path(__file__).resolve().parent
MESH_DIR = ROOT / "data" / "mesh"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["boundary"],
        choices=[
            "all",
            "boundary",
            "simple_bunny",
            "bunny10",
            "bunny50",
            "bunny100",
            "spot10",
            "spot50",
            "spot100",
            "maze3v2",
            "maze4v2",
        ],
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--eigfn-samples", type=int, default=500000)
    parser.add_argument("--upsample", type=int, default=3)
    args = parser.parse_args()

    targets = set(args.targets)
    if "all" in targets:
        targets = {
            "simple_bunny",
            "bunny10",
            "bunny50",
            "bunny100",
            "spot10",
            "spot50",
            "spot100",
            "maze3v2",
            "maze4v2",
        }
    if "boundary" in targets:
        targets.update({"simple_bunny", "maze3v2", "maze4v2"})

    if "simple_bunny" in targets:
        create_simple_bunny(replace=args.replace)
    for obj, idx, name in [
        ("bunny", 9, "bunny10"),
        ("bunny", 49, "bunny50"),
        ("bunny", 99, "bunny100"),
        ("spot", 9, "spot10"),
        ("spot", 49, "spot50"),
        ("spot", 99, "spot100"),
    ]:
        if name in targets:
            create_eigfn(
                obj,
                idx,
                nsamples=args.eigfn_samples,
                upsample=args.upsample,
                replace=args.replace,
            )
    if "maze3v2" in targets:
        gen_mazev2(3, replace=args.replace)
    if "maze4v2" in targets:
        gen_mazev2(4, replace=args.replace)


def create_simple_bunny(replace: bool = False) -> None:
    out = MESH_DIR / "bunny_simple.npy"
    if _valid_npy(out) and not replace:
        print(f"{out} exists. Skipping.")
        return

    np.random.seed(777)
    v, f = igl.read_triangle_mesh(str(MESH_DIR / "bunny_simp.obj"))
    vi = v[234]
    x = np.random.randn(200000, 3) * 10.0 + vi
    x = _signed_distance_result(x, v, f)[2]
    with out.open("wb") as fh:
        np.save(fh, x.astype(np.float32))
    print(f"Saved {out}.")


def create_eigfn(
    obj: str,
    idx: int,
    nsamples: int = 500000,
    upsample: int = 3,
    replace: bool = False,
) -> None:
    out = MESH_DIR / f"{obj}_eigfn{idx:03d}.npy"
    if _valid_npy(out) and not replace:
        print(f"{out} exists. Skipping.")
        return

    np.random.seed(777)
    v, f = igl.read_triangle_mesh(str(MESH_DIR / f"{obj}_simp.obj"))
    v_t, f_t = torch.tensor(v), torch.tensor(f)
    mesh = Mesh(v_t, f_t, numeigs=idx + 20, upsample=upsample)

    vals = mesh.eigfns[:, idx].clamp(min=0.0)
    vals = torch.mean(vals[mesh.f], dim=1)
    vals = vals * mesh.areas

    f_idx = torch.multinomial(vals, nsamples, replacement=True)
    barycoords = sample_simplex_uniform(
        2, (nsamples,), dtype=mesh.v.dtype, device=mesh.v.device
    )
    samples = torch.sum(mesh.v[mesh.f[f_idx]] * barycoords[..., None], axis=1)
    samples = samples.cpu().detach().numpy()
    with out.open("wb") as fh:
        np.save(fh, samples.astype(np.float32))
    print(f"Saved {out}.")


def gen_mazev2(n: int, replace: bool = False) -> None:
    out = MESH_DIR / f"maze_{n}x{n}v2.npz"
    if _valid_npz(out) and not replace:
        print(f"{out} exists. Skipping.")
        return

    v, f = igl.read_triangle_mesh(str(MESH_DIR / f"maze_{n}x{n}.obj"))

    def reject_outside_mesh(data: np.ndarray) -> np.ndarray:
        data = np.concatenate([data, np.zeros_like(data[:, 0:1])], axis=1)
        dist = _signed_distance_result(data, v, f)[0]
        data = data[dist == 0]
        return data

    np.random.seed(777)
    torch.manual_seed(777)

    total = 100000
    std = 0.02 * 32 / (n * 10 + 2)
    offset = (1 / (n * 10 + 2) * 5 + 0.5) if n % 2 == 0 else 0.5

    x0 = np.random.randn(int(10 * total), 2)
    x0 = (x0 * std + offset) * n
    x0 = reject_outside_mesh(x0)
    z0 = x0.shape[0] / (10 * total)
    x0 = x0[:total]

    if n == 3:
        gmm = GaussianMM([[1 / 6, 1 / 6], [5 / 6, 5 / 6]], 0.02)
    elif n == 4:
        gmm = GaussianMM(
            [[1 / 8, 1 / 8], [1 / 8, 7 / 8], [7 / 8, 1 / 8]],
            0.02 * 32 / (n * 10 + 2),
        )
    else:
        raise ValueError(f"Unsupported maze size: {n}")

    x1 = n * gmm.sample(total).detach().cpu().numpy()
    x1 = reject_outside_mesh(x1)
    z1 = x1.shape[0] / (10 * total)
    x1 = x1[:total]

    with out.open("wb") as fh:
        np.savez(fh, x0=x0, x1=x1, Z0=z0, Z1=z1, std=std)
    print(f"Saved {out}.")


def _valid_npy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        np.load(path, mmap_mode="r")
        return True
    except Exception:
        return False


def _valid_npz(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return {"x0", "x1", "Z0", "Z1"}.issubset(data.files)
    except Exception:
        return False


def _signed_distance_result(points: np.ndarray, v: np.ndarray, f: np.ndarray):
    result = igl.signed_distance(points, v, f)
    if len(result) == 3:
        return result
    if len(result) == 4:
        dist, face_index, closest_points, _normals = result
        return dist, face_index, closest_points
    raise ValueError(f"Unexpected igl.signed_distance return arity: {len(result)}")


if __name__ == "__main__":
    main()
