# Real RFM Experiment Map

Scope: read-only audit of the official `external/riemannian-fm` code for moving from synthetic R-MGM checks to the real experiments used by Flow Matching on General Geometries. No long training runs were launched.

## Code Paths

- Main entrypoint: `external/riemannian-fm/train.py`
- Evaluation entrypoint: `external/riemannian-fm/evaluate.py`
- Global config: `external/riemannian-fm/configs/train.yaml`
- Experiment configs: `external/riemannian-fm/configs/experiment/*.yaml`
- Dataset routing: `external/riemannian-fm/manifm/datasets.py`
- RFM Lightning module, likelihoods, plots: `external/riemannian-fm/manifm/model_pl.py`
- Mesh data synthesis: `external/riemannian-fm/data/synthesize_mesh_data.py`

## Experiment Inventory

| Group | Configs / `cfg.data` | Manifold | Required data | Local status |
|---|---|---|---|---|
| Earth events | `volcano`, `earthquake`, `fire`, `flood` | `Sphere`, dim 3 | `data/earth_data/{volcano,earthquake,fire,flood}.csv` | Present |
| Protein torsion | `general`, `glycine`, `proline`, `prepro` | `FlatTorus`, dim 2 | `data/top500/aggregated_angles.tsv` filtered by amino type | Missing; must download PDBs and run `get_torsion_angle.py` |
| RNA torsion | `rna` | `FlatTorus`, dim 7 | `data/rna/aggregated_angles.tsv` | Missing; must download PDBs and run `get_torsion_angles.py` |
| Wrapped torus | `wrapped_torus` | `FlatTorus`, dim 5 | Synthetic `Wrapped` dataset in code | Ready, not external real data |
| Wrapped SPD | `wrapped_spd` | `SPD`, dim 3 vectorized | Synthetic `Wrapped` dataset centered at vectorized `2I` | Ready, not external real data |
| Hyperbolic | `hyperbolic` | `PoincareBall`, dim 2 | Synthetic paired wrapped-normal data in code | Ready, not external real data |
| Bunny / Spot mesh | `simple_bunny`, `bunny10/50/100`, `spot10/50/100` | `Mesh`, dim 3 | OBJ plus generated `.npy`: `bunny_simple.npy`, `bunny_eigfn009/049/099.npy`, `spot_eigfn009/049/099.npy` | OBJ present, `.npy` missing, `igl` missing |
| Maze mesh | `maze3v2`, `maze4v2` | `Mesh`, paired dim 3 | `maze_3x3v2.npz`, `maze_4x4v2.npz`, plus OBJ files | OBJ present, `.npz` missing, `igl` missing |

Earth CSV sample counts, excluding header: `volcano` 827, `earthquake` 6120, `fire` 12809, `flood` 4875.

## Metrics In Official Code

The original RFM code mainly reports likelihood:

- `train/loss`: RFM vector-field regression loss, not NLL.
- `val/loss`: validation NLL, computed as `-mean(compute_exact_loglikelihood(x))`.
- `val/loss_best`: best validation NLL, used for checkpoint selection.
- `test/loss`: test NLL.
- `nfe`: number of function evaluations during likelihood ODE.
- `integ_error`: projection/integration error after reverse ODE.
- `frac_within_manifold`: only for SPD, fraction passing the integration-error mask.

Not present as reported metrics in the official code:

- No bits/dim. `normalize_loglik=True` divides log-likelihood by dimension, but does not convert to bits/dim.
- No explicit MMD, FID-like score, coverage, or constraint score.
- No spectral-distance metric. Mesh configs use biharmonic/diffusion/commutetime distances to define mesh geometry and paths, not as a final reported score.

Figures:

- Earth: density contours on a world map plus data points.
- 2D torus: log-likelihood contours in angle space plus data points.
- Mesh: VTK density and trajectory outputs.
- SPD / Poincare: sample and trajectory plots.

## Baseline RFM Commands

Run from:

```powershell
cd "E:\Codex Projects\Riemmanian Midpoint Generative Models\external\riemannian-fm"
```

Use absolute datadirs because Hydra changes into the output directory (`hydra.job.chdir=True`).

Single-seed Earth RFM baseline:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B train.py experiment=volcano seed=0 use_wandb=False "earth_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/earth_data"
```

Local smoke only, not a paper run:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B train.py experiment=volcano seed=0 use_wandb=False optim.num_iterations=10 optim.batch_size=64 optim.val_batch_size=64 val_every=10 "earth_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/earth_data" hydra.run.dir=../../outputs/rfm_real_smoke/volcano
```

Multi-seed RFM baseline locally:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B train.py -m experiment=volcano seed=0,1,2,3,4 use_wandb=False "hydra/launcher=basic" "earth_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/earth_data"
```

Evaluate a trained checkpoint:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B evaluate.py nll "<path-to-checkpoint.ckpt>"
```

Force paper-style visualization:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B evaluate.py visualize "<path-to-checkpoint.ckpt>"
```

## R-MGM Transfer Plan

The current `rmgm_train.py` is synthetic-first: it constructs samplers inside `make_experiment()` and does not use `manifm.datasets.get_loaders`. To move to the real RFM setups:

1. Add a real-data mode that builds an OmegaConf config from `configs/train.yaml` + `configs/experiment/<name>.yaml`.
2. Call `get_loaders(cfg)` and `get_manifold(cfg)`.
3. Wrap the train split as a sampler with `sample(n, device, dtype)` so the existing MGM loss can consume empirical target samples.
4. Start with non-pair, closed-form manifolds: Earth/Sphere first, then protein/RNA/FlatTorus after preprocessing.
5. Add sample metrics common to RFM and R-MGM: MMD in ambient/geometric coordinates, geodesic nearest-neighbor distance, coverage/precision-style radius metrics, and constraint error.
6. Keep official RFM `test/loss` NLL for RFM. R-MGM currently has no exact likelihood because it is a one-step implicit generator, so direct NLL comparison requires new density machinery or a separate approximate estimator.

Recommended first real experiments:

- `volcano`: smallest real Earth dataset, quickest debug.
- `flood` or `fire`: larger Earth dataset for more meaningful sample quality.
- `earthquake`: also real Earth, moderate size.
- Protein `general/proline/glycine/prepro`: after `aggregated_angles.tsv` exists.

Defer:

- Mesh bunny/spot/maze until `igl` is installed and mesh `.npy/.npz` files are generated.
- Mesh R-MGM until `manifm.rmgm.geometry` supports `Mesh.solve_path`.

## Practical Blockers

- `data/top500/aggregated_angles.tsv` is absent.
- `data/rna/aggregated_angles.tsv` is absent.
- Mesh `.npy/.npz` targets are absent.
- `igl` is absent in `.conda\rmgm`; mesh loaders fail without it.
- `Bio` / Biopython is absent in `.conda\rmgm`; protein/RNA preprocessing will fail until installed.
- `evtk` / VTK export dependency is absent; mesh visualization may fail.
- Many configs set `use_wandb: True`; local runs should override `use_wandb=False`.
- Default datadirs in `configs/train.yaml` point to `/checkpoint/${USER}/data/...`, not this Windows workspace.
- Original `train.py` hardcodes `accelerator="gpu"` and assumes a CUDA GPU.
- Original RFM checkpoints are Lightning `.ckpt` tied to Hydra `.hydra/config.yaml`; current R-MGM checkpoints are standalone `.pt`.

