# TEST: frozen configurations, validation-selected checkpoints

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

## Earthquake

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| MGM | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| MGM | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| MGM | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| MGM | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| MGM | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| MGM | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| MGM | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |
| KGDM | unspecified | 46b998681a6e | Biased kernel MMD2_V (primary) | 0.0022231 (0.0018333) | 0.0021717 (0.0013508) | 0.0020026 (0.0014045) | 0.0021325 | 0.00011537 |
| KGDM | unspecified | 46b998681a6e | Symmetric NN distance | 0.048356 (0.0028322) | 0.050451 (0.0028887) | 0.049421 (0.00073598) | 0.049409 | 0.0010473 |
| KGDM | unspecified | 46b998681a6e | MinimumMatchingDistance (directed NN) | 0.03963 (0.0021879) | 0.039779 (0.00088258) | 0.039083 (0.0014358) | 0.039497 | 0.00036662 |
| KGDM | unspecified | 46b998681a6e | Radius coverage | 0.17429 (0.013703) | 0.14815 (0.013897) | 0.14052 (0.013961) | 0.15432 | 0.017711 |
| KGDM | unspecified | 46b998681a6e | NN occupancy coverage | 0.49346 (0.011783) | 0.49237 (0.0061862) | 0.48693 (0.0028301) | 0.49092 | 0.0035017 |
| KGDM | unspecified | 46b998681a6e | 1-NNA (ideal near 0.5) | 0.68083 (0.0079351) | 0.69009 (0.014346) | 0.68655 (0.009747) | 0.68582 | 0.0046722 |
| KGDM | unspecified | 46b998681a6e | Mean constraint residual | 3.2386e-08 (8.5115e-10) | 3.1954e-08 (5.6662e-10) | 3.2457e-08 (1.3846e-09) | 3.2266e-08 | 2.7224e-10 |
| KGDM | unspecified | 46b998681a6e | Recorded training seconds | 366.47 (0) | 357.1 (0) | 352.62 (0) | 358.73 | 7.0649 |
| MeanFlow | rmf_mt | 57e0297c5365 | Biased kernel MMD2_V (primary) | 0.0035934 (0.0008128) | 0.0025827 (0.0001591) | 0.0022651 (0.00098445) | 0.0028137 | 0.0006936 |
| MeanFlow | rmf_mt | 57e0297c5365 | Symmetric NN distance | 0.076908 (0.0014776) | 0.069393 (0.0020245) | 0.069748 (0.0036167) | 0.072016 | 0.0042398 |
| MeanFlow | rmf_mt | 57e0297c5365 | MinimumMatchingDistance (directed NN) | 0.053724 (0.0015476) | 0.050464 (0.0021111) | 0.051232 (0.0023331) | 0.051807 | 0.0017044 |
| MeanFlow | rmf_mt | 57e0297c5365 | Radius coverage | 0.08061 (0.0061862) | 0.12255 (0.012969) | 0.081155 (0.010631) | 0.094771 | 0.024058 |
| MeanFlow | rmf_mt | 57e0297c5365 | NN occupancy coverage | 0.39434 (0.016367) | 0.42375 (0.0089993) | 0.40196 (0.010204) | 0.40668 | 0.015264 |
| MeanFlow | rmf_mt | 57e0297c5365 | 1-NNA (ideal near 0.5) | 0.78377 (0.0096783) | 0.75354 (0.015444) | 0.77015 (0.014983) | 0.76915 | 0.015139 |
| MeanFlow | rmf_mt | 57e0297c5365 | Mean constraint residual | 3.1829e-08 (8.0812e-10) | 3.1312e-08 (3.3082e-10) | 3.2337e-08 (5.6229e-10) | 3.1826e-08 | 5.1221e-10 |
| MeanFlow | rmf_mt | 57e0297c5365 | Recorded training seconds | 500.95 (0) | 505.07 (0) | 486.72 (0) | 497.58 | 9.6283 |

## Configuration Ledger

### KGDM / unspecified / final_v1 / 46b998681a6e

```json
{
  "config_id": "46b998681a6e0fabb58c8be05fe7981e6b01638cc294cd58e50754fcf2d71669",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "kernel": "gauss",
    "latent_dim": 16,
    "lr": 0.0002,
    "step_size": 1.0,
    "width": 128
  },
  "resolved_config": {
    "bandwidth_scale": 1.0,
    "batch_size": 256,
    "betas": [
      0.9,
      0.95
    ],
    "depth": 3,
    "dist_norm": true,
    "drop_T": true,
    "ema_decay": 0.999,
    "eps": 1e-08,
    "grad_clip": 1.0,
    "kernel": "gauss",
    "latent_dim": 16,
    "lr": 0.0002,
    "max_step": 1.0,
    "sample_ema": true,
    "step_size": 1.0,
    "temps": [
      0.02,
      0.05,
      0.2
    ],
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "width": 128
  }
}
```

### MeanFlow / rmf_mt / final_v1 / 57e0297c5365

```json
{
  "config_id": "57e0297c536567d5ad4c8afeb67bb30015b157f2b2cfb520c579a8448f654022",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "latent_dim": 16,
    "lr": 0.0002,
    "variant": "rmf_mt",
    "width": 128
  },
  "resolved_config": {
    "batch_size": 256,
    "depth": 3,
    "grad_clip": null,
    "latent_dim": 16,
    "lr": 0.0002,
    "pcgrad_reduction": "mean",
    "prop_equal_times": 0.75,
    "time_distribution": "uniform",
    "torus_fourier": 1,
    "variant": "rmf_mt",
    "weight_decay": 0.01,
    "width": 128
  }
}
```

