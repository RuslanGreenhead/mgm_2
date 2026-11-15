# TEST: frozen configurations, validation-selected checkpoints

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

## Volcano

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | unspecified | 461f24fd75e8 | Biased kernel MMD2_V (primary) | 0.014827 (0.0049123) | 0.012811 (0.0043029) | 0.011767 (0.0071815) | 0.013135 | 0.0015554 |
| MGM | unspecified | 461f24fd75e8 | Symmetric NN distance | 0.18415 (0.0051485) | 0.19004 (0.014334) | 0.15191 (0.013942) | 0.17537 | 0.020524 |
| MGM | unspecified | 461f24fd75e8 | MinimumMatchingDistance (directed NN) | 0.14332 (0.0079909) | 0.15099 (0.013897) | 0.11926 (0.011159) | 0.13786 | 0.016558 |
| MGM | unspecified | 461f24fd75e8 | Radius coverage | 0 (0) | 0.012195 (0.021123) | 0.012195 (0.021123) | 0.0081301 | 0.0070409 |
| MGM | unspecified | 461f24fd75e8 | NN occupancy coverage | 0.47154 (0.025386) | 0.47561 (0.02439) | 0.5 (0.012195) | 0.48238 | 0.01539 |
| MGM | unspecified | 461f24fd75e8 | 1-NNA (ideal near 0.5) | 0.74797 (0.0035204) | 0.74187 (0.036754) | 0.69512 (0.047623) | 0.72832 | 0.028911 |
| MGM | unspecified | 461f24fd75e8 | Mean constraint residual | 3.0955e-08 (2.1366e-09) | 3.2824e-08 (2.057e-09) | 3.0906e-08 (2.0396e-09) | 3.1562e-08 | 1.0936e-09 |
| MGM | unspecified | 461f24fd75e8 | Recorded training seconds | 935.04 (0) | 924.68 (0) | 925.51 (0) | 928.41 | 5.7554 |
| RWFlow | unspecified | d58e75450893 | Biased kernel MMD2_V (primary) | 0.012754 (0.0080628) | 0.012632 (0.0086808) | 0.012224 (0.0073796) | 0.012537 | 0.0002771 |
| RWFlow | unspecified | d58e75450893 | Symmetric NN distance | 0.14282 (0.0058578) | 0.13467 (0.003226) | 0.14612 (0.0082617) | 0.14121 | 0.0058943 |
| RWFlow | unspecified | d58e75450893 | MinimumMatchingDistance (directed NN) | 0.12502 (0.0087966) | 0.12175 (0.0046979) | 0.1286 (0.01531) | 0.12512 | 0.003428 |
| RWFlow | unspecified | d58e75450893 | Radius coverage | 0.02439 (0.012195) | 0.01626 (0.0070409) | 0.01626 (0.018628) | 0.01897 | 0.0046939 |
| RWFlow | unspecified | d58e75450893 | NN occupancy coverage | 0.49187 (0.0070409) | 0.50813 (0.025386) | 0.46341 (0.032265) | 0.4878 | 0.022633 |
| RWFlow | unspecified | d58e75450893 | 1-NNA (ideal near 0.5) | 0.73374 (0.030079) | 0.71341 (0.026579) | 0.72764 (0.062283) | 0.72493 | 0.01043 |
| RWFlow | unspecified | d58e75450893 | Mean constraint residual | 3.4606e-08 (2.2764e-09) | 3.1251e-08 (2.2591e-09) | 3.1205e-08 (2.4615e-09) | 3.2354e-08 | 1.9505e-09 |
| RWFlow | unspecified | d58e75450893 | Recorded training seconds | 413.3 (0) | 410.86 (0) | 383.56 (0) | 402.57 | 16.508 |
| KGDM | unspecified | 46b998681a6e | Biased kernel MMD2_V (primary) | 0.01561 (0.0044666) | 0.013869 (0.0054608) | 0.012222 (0.0045267) | 0.0139 | 0.0016943 |
| KGDM | unspecified | 46b998681a6e | Symmetric NN distance | 0.11461 (0.0031565) | 0.1168 (0.011418) | 0.10165 (0.0044312) | 0.11102 | 0.0081904 |
| KGDM | unspecified | 46b998681a6e | MinimumMatchingDistance (directed NN) | 0.10654 (0.0081448) | 0.10606 (0.016831) | 0.09036 (0.0010232) | 0.10099 | 0.0092059 |
| KGDM | unspecified | 46b998681a6e | Radius coverage | 0.04065 (0.03069) | 0.036585 (0.032265) | 0.02439 (0.032265) | 0.033875 | 0.0084621 |
| KGDM | unspecified | 46b998681a6e | NN occupancy coverage | 0.5 (0.032265) | 0.5122 (0) | 0.48374 (0.04617) | 0.49864 | 0.014276 |
| KGDM | unspecified | 46b998681a6e | 1-NNA (ideal near 0.5) | 0.66057 (0.056656) | 0.67683 (0.079268) | 0.68496 (0.049662) | 0.67412 | 0.012419 |
| KGDM | unspecified | 46b998681a6e | Mean constraint residual | 3.0945e-08 (4.3171e-09) | 3.2067e-08 (1.3918e-09) | 3.3994e-08 (7.4335e-10) | 3.2335e-08 | 1.5423e-09 |
| KGDM | unspecified | 46b998681a6e | Recorded training seconds | 357.83 (0) | 357.64 (0) | 531.52 (0) | 415.67 | 100.33 |
| MeanFlow | rmf_mt | 57e0297c5365 | Biased kernel MMD2_V (primary) | 0.012187 (0.00090292) | 0.012032 (0.0055792) | 0.011652 (0.0019345) | 0.011957 | 0.00027519 |
| MeanFlow | rmf_mt | 57e0297c5365 | Symmetric NN distance | 0.15251 (0.0089517) | 0.16768 (0.0098901) | 0.1387 (0.0040588) | 0.15296 | 0.014495 |
| MeanFlow | rmf_mt | 57e0297c5365 | MinimumMatchingDistance (directed NN) | 0.13666 (0.01501) | 0.15374 (0.014185) | 0.12224 (0.0086908) | 0.13755 | 0.015768 |
| MeanFlow | rmf_mt | 57e0297c5365 | Radius coverage | 0 (0) | 0.0081301 (0.014082) | 0.036585 (0.032265) | 0.014905 | 0.019211 |
| MeanFlow | rmf_mt | 57e0297c5365 | NN occupancy coverage | 0.46748 (0.025386) | 0.46341 (0.02439) | 0.5 (0.021123) | 0.47696 | 0.020052 |
| MeanFlow | rmf_mt | 57e0297c5365 | 1-NNA (ideal near 0.5) | 0.76829 (0.038079) | 0.77033 (0.015345) | 0.74593 (0.014082) | 0.76152 | 0.013533 |
| MeanFlow | rmf_mt | 57e0297c5365 | Mean constraint residual | 3.3852e-08 (3.1127e-09) | 3.3442e-08 (2.4939e-09) | 3.1084e-08 (3.0914e-09) | 3.2793e-08 | 1.494e-09 |
| MeanFlow | rmf_mt | 57e0297c5365 | Recorded training seconds | 867.27 (0) | 620.86 (0) | 497.52 (0) | 661.89 | 188.26 |

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

### MGM / unspecified / final_v1 / 461f24fd75e8

```json
{
  "config_id": "461f24fd75e82b9a1b254cf0ccb7863ad9814c19deaaaf40b272820be8977893",
  "config": {
    "batch_size": 256,
    "critic_steps": 10,
    "critic_warmup": 500,
    "critic_width": 256,
    "depth": 3,
    "harmonics": 4,
    "latent_dim": 16,
    "lr": 0.0002,
    "lr_c": 0.0004,
    "noise": 0.0,
    "width": 128
  },
  "resolved_config": {
    "batch_size": 256,
    "critic_steps": 10,
    "critic_warmup": 500,
    "critic_width": 256,
    "depth": 3,
    "harmonics": 4,
    "latent_dim": 16,
    "lr": 0.0002,
    "lr_c": 0.0004,
    "noise": 0.0,
    "width": 128
  }
}
```

### RWFlow / unspecified / final_v1 / d58e75450893

```json
{
  "config_id": "d58e75450893eb0a0ad5074ba4b2e697190efdd4cb4ebcbd73e40e1817627f3b",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "epsilon": 0.3,
    "flow_step": 0.2,
    "latent_dim": 16,
    "lr": 0.0002,
    "sinkhorn_iters": 128,
    "sinkhorn_solver": "graph64",
    "width": 128
  },
  "resolved_config": {
    "batch_size": 256,
    "cost": "chordal",
    "cost_scale": 1.0,
    "depth": 3,
    "epsilon": 0.3,
    "flow_step": 0.2,
    "grad_clip": 1.0,
    "latent_dim": 16,
    "lr": 0.0002,
    "self_batch_size": null,
    "sinkhorn_iters": 128,
    "sinkhorn_solver": "graph64",
    "sinkhorn_tol": null,
    "source_batch_size": null,
    "weight_decay": 0.01,
    "width": 128
  }
}
```

