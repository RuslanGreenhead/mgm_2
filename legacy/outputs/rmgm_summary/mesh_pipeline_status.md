# Mesh Pipeline Status for RFM Paper Experiments

Checked on 2026-08-01 from project root:

```powershell
E:\Codex Projects\Riemmanian Midpoint Generative Models
```

No mesh data generation was launched in this audit. Only dependency imports,
file existence, and lightweight `igl.read_triangle_mesh` checks were run.

## Code paths

- Data synthesis: `external/riemannian-fm/data/synthesize_mesh_data.py`
- Dataset routing: `external/riemannian-fm/manifm/datasets.py`
- Mesh manifold/eigensolver code: `external/riemannian-fm/manifm/manifolds/mesh.py`
- VTK export helpers: `external/riemannian-fm/manifm/mesh_utils.py`
- Default RFM mesh datadir key: `mesh_datadir` in `external/riemannian-fm/configs/train.yaml`

## Dependency status

Local env checked:

```powershell
.\.conda\rmgm\python.exe
```

Current status in this env:

| Dependency | Import checked | Local status | Why it matters |
|---|---:|---|---|
| `igl` | `import igl` | OK, conda package `igl 2.6.1`, Python dist metadata `libigl 2.6.1` | Required by synthesis, mesh loading, eigensolvers, projection, signed distance. |
| `pyevtk` | `import pyevtk` | OK, `pyevtk 1.7.0` | Package used for VTK output. |
| `evtk` | `import evtk` | OK | Legacy import path used by `manifm/mesh_utils.py`: `from evtk import hl, vtk`. |
| `torch`, `numpy`, `scipy`, `sklearn` | imports checked | OK | Required by synthesis and mesh manifold preprocessing. |

Clean-env install commands:

```powershell
conda install -p ".conda\rmgm" -c conda-forge igl pyevtk
```

Alternative pip route for an activated env:

```powershell
.\.conda\rmgm\python.exe -m pip install libigl pyevtk
```

Notes:

- The PyPI package name for the Python bindings is `libigl`, but the import name is `igl`.
- The conda-forge package name is `igl`.
- The RFM `environment.yml` lists `igl` through conda channels and `pyevtk` under pip.

## Existing source meshes

All required source mesh files for the requested experiments are present under:

```powershell
external\riemannian-fm\data\mesh
```

Lightweight `igl.read_triangle_mesh` check:

| Mesh file | Status | Vertices | Faces | Used by |
|---|---:|---:|---:|---|
| `bunny_simp.obj` | present/read OK | 2502 | 5000 | `simple_bunny`, `bunny10`, `bunny50`, `bunny100` |
| `spot_simp.obj` | present/read OK | 2930 | 5856 | `spot10`, `spot50`, `spot100` |
| `maze_3x3.obj` | present/read OK | 801 | 1408 | `maze3v2` |
| `maze_4x4.obj` | present/read OK | 1431 | 2528 | `maze4v2` |
| `bunny.obj`, `bunny.off` | present | not re-read here | not re-read here | only needed if `bunny_simp.obj` must be recreated |

## Required generated files

All generated targets below are currently missing.

| `cfg.data` | Loader class | Required data file | Required mesh file | Scale in loader | Generator |
|---|---|---|---|---:|---|
| `simple_bunny` | `SimpleBunny` | `bunny_simple.npy` | `bunny_simp.obj` | `1/250` | `create_simple_bunny(replace=False)` |
| `bunny10` | `Bunny10` | `bunny_eigfn009.npy` | `bunny_simp.obj` | `1/250` | `create_eigfn("bunny", 9, upsample=3, replace=False)` |
| `bunny50` | `Bunny50` | `bunny_eigfn049.npy` | `bunny_simp.obj` | `1/250` | `create_eigfn("bunny", 49, upsample=3, replace=False)` |
| `bunny100` | `Bunny100` | `bunny_eigfn099.npy` | `bunny_simp.obj` | `1/250` | `create_eigfn("bunny", 99, upsample=3, replace=False)` |
| `spot10` | `Spot10` | `spot_eigfn009.npy` | `spot_simp.obj` | `1.0` | `create_eigfn("spot", 9, upsample=3, replace=False)` |
| `spot50` | `Spot50` | `spot_eigfn049.npy` | `spot_simp.obj` | `1.0` | `create_eigfn("spot", 49, upsample=3, replace=False)` |
| `spot100` | `Spot100` | `spot_eigfn099.npy` | `spot_simp.obj` | `1.0` | `create_eigfn("spot", 99, upsample=3, replace=False)` |
| `maze3v2` | `Maze3v2` | `maze_3x3v2.npz` | `maze_3x3.obj` | `1/3` | `gen_mazev2(3, replace=False)` |
| `maze4v2` | `Maze4v2` | `maze_4x4v2.npz` | `maze_4x4.obj` | `1/4` | `gen_mazev2(4, replace=False)` |

Expected generated contents:

- `bunny_simple.npy`: float32 array of projected bunny points, generated from 200000 Gaussian proposals around one bunny vertex.
- `*_eigfn*.npy`: float32 array of 500000 mesh surface samples, sampled from the positive part of the selected mesh eigenfunction after `upsample=3`.
- `maze_3x3v2.npz`, `maze_4x4v2.npz`: keys `x0`, `x1`, `Z0`, `Z1`, `std`; points are 3D because synthesis pads planar maze samples with `z=0`.

## Safe generation plan

Run only after confirming `igl` imports in the same env that will run synthesis.
These commands are heavy: bunny/spot eigfn generation computes sparse mesh
eigenfunctions and samples 500000 points per file; each maze v2 generator uses
signed-distance rejection over large proposal batches.

From the RFM data directory:

```powershell
cd "E:\Codex Projects\Riemmanian Midpoint Generative Models\external\riemannian-fm\data"
$env:PYTHONPATH = (Resolve-Path ..).Path
& "..\..\..\.conda\rmgm\python.exe" -c "from synthesize_mesh_data import create_simple_bunny; create_simple_bunny(replace=False)"
& "..\..\..\.conda\rmgm\python.exe" synthesize_mesh_data.py
```

Why two commands:

- `synthesize_mesh_data.py` main generates `bunny_eigfn009/049/099.npy`, `spot_eigfn009/049/099.npy`, `maze_3x3v2.npz`, and `maze_4x4v2.npz`.
- Its main block does not call `create_simple_bunny`, so `bunny_simple.npy` must be generated separately.
- `replace=False` makes existing targets skip-safe.

Verification after generation:

```powershell
cd "E:\Codex Projects\Riemmanian Midpoint Generative Models"
& ".\.conda\rmgm\python.exe" -c "from pathlib import Path; root=Path(r'external/riemannian-fm/data/mesh'); files=['bunny_simple.npy','bunny_eigfn009.npy','bunny_eigfn049.npy','bunny_eigfn099.npy','spot_eigfn009.npy','spot_eigfn049.npy','spot_eigfn099.npy','maze_3x3v2.npz','maze_4x4v2.npz']; print('\n'.join(f'{f}: {root.joinpath(f).exists()}' for f in files))"
```

## Running RFM mesh experiments after data exists

Use an absolute `mesh_datadir` override because Hydra changes into the run
directory and the default points at `/checkpoint/${USER}/data/mesh`.

Example:

```powershell
cd "E:\Codex Projects\Riemmanian Midpoint Generative Models\external\riemannian-fm"
& "..\..\.conda\rmgm\python.exe" -B train.py experiment=bunny10 seed=0 use_wandb=False "mesh_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/mesh" "hydra/launcher=basic"
```

Swap `experiment=bunny10` for:

```text
simple_bunny, bunny50, bunny100, spot10, spot50, spot100, maze3v2, maze4v2
```

## Boundary and R-MGM notes

- `Mesh` has a `dirichlet_bc` constructor argument and can use `igl.boundary_facets`, but `get_manifold(cfg)` currently passes only `numeigs`, `metric`, and `temp`.
- Current RFM configs therefore use the default `dirichlet_bc=False`. This is not a blocker for reproducing the existing configs, but it is a decision point if explicit Dirichlet boundary conditions are required for maze/manifold-with-boundary ablations.
- The current real-data R-MGM trainer rejects callable mesh manifolds, and `manifm/rmgm/geometry.py` intentionally raises `NotImplementedError` for Mesh midpoint paths. This blocks mesh-valued R-MGM training, but not official RFM mesh training/evaluation.

## Exact blockers

1. Generated mesh dataset arrays are absent:
   `bunny_simple.npy`, `bunny_eigfn009.npy`, `bunny_eigfn049.npy`,
   `bunny_eigfn099.npy`, `spot_eigfn009.npy`, `spot_eigfn049.npy`,
   `spot_eigfn099.npy`, `maze_3x3v2.npz`, `maze_4x4v2.npz`.
2. Heavy synthesis still needs to be run deliberately after choosing a compute window.
3. On any clean env, missing `igl` blocks synthesis and mesh dataset loading.
4. On any clean env, missing `pyevtk`/`evtk` blocks mesh VTK visualization when `visualize=True`.
5. Mesh-valued R-MGM training is not wired yet because mesh midpoint paths are not implemented in the R-MGM geometry layer.
