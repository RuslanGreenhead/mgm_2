# Extra Geometry And High-D Scaling Plan

Scope: audit `configs/experiment/wrapped_torus.yaml`,
`configs/experiment/wrapped_spd.yaml`, `configs/experiment/hyperbolic.yaml`,
`rmgm_train.py`, `rfm_baseline_train.py`, and `rmgm_train_real.py`. No long
training runs were launched for this pass.

## Files Added

- `external/riemannian-fm/rmgm_run_extra_geometries.py`
- `outputs/rmgm_summary/extra_geometry_plan.md`

Both are disjoint new files; existing training/config files were not edited.

## Config Audit

| Config | Official RFM data/manifold | Dim from YAML | Local notes |
|---|---|---:|---|
| `wrapped_torus.yaml` | `Wrapped(FlatTorus)` | 5 | Official Hydra path honors `wrapped.dim`. It also sets `model.d_model=5000`, `num_iterations=50000`, `use_wandb=True`; local GPU runs should override `use_wandb=False` and use smoke/pilot overrides first. |
| `wrapped_spd.yaml` | `Wrapped(SPD)` | 3 | Official path is vectorized SPD(2), because triangular SPD vector dim 3 maps to 2x2 matrices. High-D SPD scaling can use triangular vector dims only: 6 for SPD(3), 10 for SPD(4), 15 for SPD(5). |
| `hyperbolic.yaml` | `HyperbolicDatasetPair(PoincareBall)` | 2 | Official and standalone paths are 2D only. There is no YAML dim knob for higher-dimensional hyperbolic scaling. |

## Code Support Matrix

| Path | Supported now | Important limitation |
|---|---|---|
| `rmgm_train.py` | Synthetic `torus2d`/`wrapped_torus`, `spd`/`wrapped_spd`, `hyperbolic`, `sphere`; writes `metrics.jsonl`, `metrics.json`, checkpoints, and sample PNGs. | `make_experiment()` hard-codes `wrapped_torus` to 2D, `wrapped_spd` to vector dim 3, and `hyperbolic` to 2D. It does not read Hydra YAML and has no `--wrapped-dim`. |
| `rfm_baseline_train.py` | Mirrors `rmgm_train.py`; supports full RFM-style sample baseline via `--ode-steps 128` and same-NFE via `--ode-steps 1`. | Same hard-coded dimensions, because it imports `make_experiment()` from `rmgm_train.py`. |
| `rmgm_train_real.py` | Real Earth/Sphere, Top500/RNA FlatTorus, and direct simple-manifold datasets through `_get_dataset`; reports shared sample metrics for R-MGM and lightweight RFM. | It does not compose Hydra experiment YAML. Its internal `wrapped=DotConfig(dim=2, ...)` makes `wrapped_torus` 2D and makes `wrapped_spd` unsafe; avoid it for wrapped SPD scaling. It is sample-quality RFM, not official likelihood RFM. |
| `train.py` | Official Hydra RFM path; honors `configs/experiment/*.yaml` and `wrapped.dim`; computes validation/test likelihood metrics. | RFM only, GPU hard-coded, no R-MGM, and no shared MMD/coverage metrics unless added separately. |

## Recommended GPU Command Entry

Run all helper commands from:

```powershell
cd "E:\Codex Projects\Riemmanian Midpoint Generative Models\external\riemannian-fm"
```

Print the plan without training:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_run_extra_geometries.py --profile pilot --format markdown
```

Smoke-launch only the first appendix command:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_run_extra_geometries.py --section appendix --profile smoke --limit 1 --run
```

Pilot appendix block, when ready:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_run_extra_geometries.py --section appendix --profile pilot --run
```

Official high-D RFM smoke/pilot block, when ready:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_run_extra_geometries.py --section official-rfm-highd --profile smoke --run
```

Collect standalone appendix metrics after runs:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_run_extra_geometries.py --section metrics --run
```

## What To Run

Appendix geometries already supported by standalone R-MGM/RFM:

- `torus2d`: R-MGM with torus GMM head and k-means init; RFM 128 NFE; RFM 1 NFE.
- `spd`: R-MGM with `--metric-normalize`; RFM 128 NFE; RFM 1 NFE.
- `hyperbolic`: R-MGM with local GMM head and k-means init; RFM 128 NFE; RFM 1 NFE.

High-dimensional scaling supported now only for official RFM:

- `wrapped_torus`, dim 5 from YAML.
- `wrapped_torus`, dim 10 and 20 via `wrapped.dim` override.
- `wrapped_spd`, vector dims 3, 6, 10 via `wrapped.dim` override.
- `hyperbolic`, dim 2 only.

High-dimensional R-MGM/RFM-sample comparison is not fully wired yet because the
standalone scripts do not expose wrapped dimensions. Minimal future extension:
add `--wrapped-dim`, `--wrapped-scale`, and, for SPD, validate triangular vector
dims before constructing centers.

## Metrics Needed

Common R-MGM/RFM sample-quality table:

- `mmd2` in natural embeddings: torus uses `[cos(theta), sin(theta)]`; SPD and
  hyperbolic use flattened ambient coordinates.
- `constraint_error` for all generated samples.
- `nn_sample_to_data`, `nn_data_to_sample`, `symmetric_nn`, and
  `coverage_at_radius` for held-out or synthetic reference samples.
- Best and final step by `mmd2`; include seed and evaluation sample count.

RFM official likelihood table:

- `val/loss_best`, `test/loss`, `train/loss`.
- `nfe` and `integ_error`.
- `frac_within_manifold` for SPD.
- Record `div_mode`, `eval_projx`, `local_coords`, and whether log likelihood is
  normalized by dimension.

Scaling table:

- Geometry, dimension, method, seed, steps, batch size, hidden dim, layers.
- Wall time, approximate samples/sec or steps/sec.
- Peak GPU memory if captured externally.
- Final/best metric row and checkpoint path.

## Visualizations Needed

- Torus 2D: existing degree scatter overlay is enough for appendix.
- Torus high-D: pairwise marginal grids for selected angle pairs plus circular
  mean/covariance heatmaps; avoid judging high-D from only the first two coords.
- SPD: existing 2x2 cone slice for dim 3; for higher vector dims add eigenvalue
  spectra or log-eigen histograms and a 2x2 principal submatrix cone slice.
- Hyperbolic: Poincare disk samples/trajs plus radial histogram.
- Summary figures: method grid per geometry, training curves for MMD/NN/loss,
  and a compact contact sheet of best samples.

## Current Local Evidence

Existing synthetic best MMD2 values already in `outputs/rmgm_summary`:

| Geometry | Best R-MGM | Full RFM | Same-NFE RFM |
|---|---:|---:|---:|
| `torus2d` | 0.000151 | 0.002385 | 0.013952 |
| `spd` | 0.003491 | 0.004529 | 0.007526 |
| `hyperbolic` | 0.000452 | 0.002294 | 0.004861 |

These are 2D torus, vector dim 3 SPD, and 2D hyperbolic; they should not be
described as high-dimensional wrapped scaling.
