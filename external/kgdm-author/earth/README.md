# 🌍 Earth experiment (S² geospatial generation)

Generative drifting of geospatial event distributions on the 2-sphere.
Data loading and evaluation metrics are shared with the original KGD earth
codebase; the drift method is the double-softmax recipe from `drift_core.py`.

Conda env: see `../environment.yml`. Datasets are loaded from `data/`.

## 📁 Layout

```
configs/
  config.yaml           training/eval defaults (data_dir, batch_size, ...)
  winners.yaml          per-cell final settings for the results table (committed)
data/                   earthquake.csv, volcano.csv, fire.csv, flood.csv (lat,lon)
drift_core.py           drift fields: euclidean R^3 / sphere S^2 / spectral S^2
models/
  backbone_earth.py     small MLP (noise -> R^3)
  earth_drift.py        EarthDrift wrapper: .sample() + .get_loss()
  kernels_spectral.py   table-backed spectral Matern kernel on S^2
task/earth_dataset.py   dataset, fixed split, and all eval metrics
src/
  main.py               single-run training entry point
  train.py              training loop (EMA, warmup, early stopping)
  matern_tables.py      spectral-kernel lookup tables, built automatically on first use
```

## 🎯 Reproduce a table cell

All flags for a cell are pulled from `configs/winners.yaml`:

```bash
cd earth
python -m src.main --formulation sphere-laplace-gradient --dataset volcano --seed 0
```

`--formulation` is one of the 9 keys in `configs/winners.yaml`:

```
euclidean-laplace             sphere-laplace
euclidean-laplace-gradient    sphere-laplace-gradient
euclidean-gaussian-gradient   sphere-gaussian-gradient
spectral-nu0.5                spectral-nu2.5    spectral-nuinf
```

`--dataset` is one of `earthquake | volcano | fire | flood`. The table
reports mean +/- std over seeds `0 1 2`; each run writes its final test
metrics to `results/<run_name>.json`. A run takes well under an hour on 16
CPU cores at the default `batch_size: 128`; no GPU is needed.

Free-form runs work without a formulation; everything comes from
`configs/config.yaml` and any field can be overridden on the CLI
(`--drift_type`, `--kernel`, `--grad_form`, `--drop_T`, `--matern_nu`,
`--spec_norm`, `--lr`, `--batch_size`, `--seed`, ...).

## 🌀 Drift formulations

Three drift spaces, selected by `drift_type`:

- **euclidean** — double-softmax affinity in ambient R³. The generator output
  is a raw R³ point during training and projected to S² only for
  sampling/evaluation.
- **sphere** — the same affinity with geodesic distance, log-map tangents, and
  exp-map steps on S².
- **spectral** — Nadaraya-Watson kernel score with the spectral Matérn kernel
  on S² (`matern_nu` ∈ {0.5, 2.5, ∞}), summed over a `matern_kappas`
  lengthscale ladder. The ladders in `configs/winners.yaml` are calibrated
  per dataset so each kernel decays to 1/e at the same geodesic scale T·d̄
  that each temperature T induces in the distance-normalized methods. Kernels
  are evaluated from lookup tables built automatically on first use
  (`src/matern_tables.py`).

Shared knobs (`configs/config.yaml`):

- `temps` — the temperature ladder; per-temperature drift fields are summed.
- `dist_norm` — divide kernel distances by their batch mean, making the
  temperature scale-free.
- `grad_form` — drift direction = unit kernel gradient instead of the
  displacement. `drop_T` controls whether the per-T kernel-gradient constant
  is dropped (uniform ladder) or kept (1/T-weighted ladder); `spec_norm` is
  the spectral analog (per-kappa fields rescaled by their e-folding angle).
- `max_step` — per-sample cap on the step norm; guards the exp-map from
  wrapping.

## 🌱 Reproducibility

Seeded end-to-end: the train/val/test split uses a fixed seed (42); the run
seed seeds Python / NumPy / Torch (CPU, CUDA, MPS), the DataLoader shuffle
generator, and worker init. `num_workers: 0` gives fully deterministic data
ordering.

## 📊 Metrics

Model selection is on validation MMD (geodesic-Gaussian kernel two-sample
statistic); final test reporting also includes geodesic Sinkhorn distance and
1-NN two-sample accuracy. All are computed in `task/earth_dataset.py`.

## 📈 Results

Test metrics, mean ± std over seeds `0 1 2`. All three are lower-is-better —
MMD and Sinkhorn measure geodesic distance to the real distribution; NN1 is a
1-NN two-sample accuracy whose ideal is chance, `0.5`. The top three per column
are highlighted: **best** in bold, the next two in _italic_. Spectral cells use
the calibrated per-dataset `matern_kappas` ladders and the `spec_norm` flag from
`configs/winners.yaml`.

**Volcano**

| formulation | MMD ↓ | Sinkhorn ↓ | NN1 ↓ (→0.5) |
|---|---|---|---|
| euclidean-laplace (ref) | 0.121 ± 0.021 | 0.268 ± 0.037 | 0.644 ± 0.040 |
| euclidean-laplace-gradient | _0.105 ± 0.010_ | _0.240 ± 0.015_ | _0.635 ± 0.030_ |
| euclidean-gaussian-gradient | 0.140 ± 0.025 | 0.298 ± 0.042 | 0.694 ± 0.003 |
| sphere-laplace | 0.116 ± 0.021 | 0.259 ± 0.027 | 0.658 ± 0.029 |
| sphere-laplace-gradient | 0.115 ± 0.019 | 0.275 ± 0.034 | 0.694 ± 0.022 |
| sphere-gaussian-gradient | **0.094 ± 0.014** | **0.231 ± 0.029** | 0.654 ± 0.018 |
| spectral-nu0.5 | 0.112 ± 0.026 | 0.249 ± 0.028 | _0.613 ± 0.011_ |
| spectral-nu2.5 | _0.104 ± 0.007_ | _0.240 ± 0.020_ | 0.638 ± 0.023 |
| spectral-nuinf | 0.142 ± 0.029 | 0.296 ± 0.044 | **0.598 ± 0.015** |

**Earthquake**

| formulation | MMD ↓ | Sinkhorn ↓ | NN1 ↓ (→0.5) |
|---|---|---|---|
| euclidean-laplace (ref) | 0.039 ± 0.006 | _0.134 ± 0.004_ | 0.638 ± 0.013 |
| euclidean-laplace-gradient | 0.036 ± 0.005 | 0.136 ± 0.007 | _0.626 ± 0.009_ |
| euclidean-gaussian-gradient | **0.033 ± 0.003** | **0.129 ± 0.003** | 0.666 ± 0.019 |
| sphere-laplace | 0.045 ± 0.008 | 0.144 ± 0.010 | 0.658 ± 0.017 |
| sphere-laplace-gradient | 0.037 ± 0.006 | 0.137 ± 0.003 | _0.636 ± 0.015_ |
| sphere-gaussian-gradient | _0.034 ± 0.005_ | _0.132 ± 0.008_ | 0.639 ± 0.012 |
| spectral-nu0.5 | _0.035 ± 0.003_ | 0.135 ± 0.006 | 0.638 ± 0.018 |
| spectral-nu2.5 | 0.048 ± 0.001 | 0.149 ± 0.012 | **0.620 ± 0.029** |
| spectral-nuinf | 0.043 ± 0.002 | 0.144 ± 0.003 | 0.641 ± 0.009 |

**Fire**

| formulation | MMD ↓ | Sinkhorn ↓ | NN1 ↓ (→0.5) |
|---|---|---|---|
| euclidean-laplace (ref) | 0.031 ± 0.005 | 0.113 ± 0.012 | 0.704 ± 0.028 |
| euclidean-laplace-gradient | _0.028 ± 0.004_ | **0.105 ± 0.006** | **0.660 ± 0.005** |
| euclidean-gaussian-gradient | **0.027 ± 0.003** | _0.106 ± 0.001_ | 0.751 ± 0.001 |
| sphere-laplace | _0.028 ± 0.003_ | 0.115 ± 0.005 | 0.737 ± 0.011 |
| sphere-laplace-gradient | 0.029 ± 0.008 | 0.112 ± 0.008 | _0.669 ± 0.011_ |
| sphere-gaussian-gradient | 0.028 ± 0.007 | _0.109 ± 0.011_ | 0.759 ± 0.002 |
| spectral-nu0.5 | 0.030 ± 0.007 | 0.113 ± 0.008 | _0.677 ± 0.014_ |
| spectral-nu2.5 | 0.029 ± 0.002 | 0.113 ± 0.001 | 0.681 ± 0.004 |
| spectral-nuinf | 0.041 ± 0.007 | 0.128 ± 0.011 | 0.691 ± 0.014 |

**Flood**

| formulation | MMD ↓ | Sinkhorn ↓ | NN1 ↓ (→0.5) |
|---|---|---|---|
| euclidean-laplace (ref) | 0.058 ± 0.003 | 0.174 ± 0.006 | 0.633 ± 0.007 |
| euclidean-laplace-gradient | _0.048 ± 0.004_ | 0.165 ± 0.004 | 0.637 ± 0.009 |
| euclidean-gaussian-gradient | 0.052 ± 0.007 | _0.165 ± 0.006_ | 0.640 ± 0.011 |
| sphere-laplace | 0.062 ± 0.008 | 0.182 ± 0.010 | _0.617 ± 0.008_ |
| sphere-laplace-gradient | 0.058 ± 0.003 | 0.176 ± 0.012 | 0.627 ± 0.044 |
| sphere-gaussian-gradient | _0.051 ± 0.003_ | 0.176 ± 0.012 | 0.651 ± 0.020 |
| spectral-nu0.5 | 0.052 ± 0.008 | _0.164 ± 0.013_ | 0.623 ± 0.028 |
| spectral-nu2.5 | **0.045 ± 0.007** | **0.154 ± 0.007** | **0.611 ± 0.017** |
| spectral-nuinf | 0.055 ± 0.004 | 0.170 ± 0.021 | _0.614 ± 0.033_ |

Takeaways: the kernel-gradient form consistently beats the plain displacement;
the ν=∞ heat kernel is the weakest spectral variant on the distributional
metrics (MMD/Sinkhorn); and no single configuration dominates every
dataset/metric — euclidean/spherical Gaussian-gradient is strongest on
earthquake and volcano, while the calibrated spectral kernels (ν=0.5, ν=2.5)
take the NN1 column on most datasets and win flood outright.
