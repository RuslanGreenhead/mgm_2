# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

## General

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | unspecified | 4b4c7a1b47aa | Biased kernel MMD2_V (primary) | 0.010072 (--) | -- | -- | 0.010072 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Symmetric NN distance | 0.11144 (--) | -- | -- | 0.11144 | -- |
| MGM | unspecified | 4b4c7a1b47aa | MinimumMatchingDistance (directed NN) | 0.051733 (--) | -- | -- | 0.051733 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Radius coverage | 0.4082 (--) | -- | -- | 0.4082 | -- |
| MGM | unspecified | 4b4c7a1b47aa | NN occupancy coverage | 0.41797 (--) | -- | -- | 0.41797 | -- |
| MGM | unspecified | 4b4c7a1b47aa | 1-NNA (ideal near 0.5) | 0.65869 (--) | -- | -- | 0.65869 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Recorded training seconds | 26.452 (--) | -- | -- | 26.452 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Biased kernel MMD2_V (primary) | 0.0017461 (--) | -- | -- | 0.0017461 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Symmetric NN distance | 0.084668 (--) | -- | -- | 0.084668 | -- |
| RWFlow | unspecified | 26499a4b1d83 | MinimumMatchingDistance (directed NN) | 0.050271 (--) | -- | -- | 0.050271 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Radius coverage | 0.44824 (--) | -- | -- | 0.44824 | -- |
| RWFlow | unspecified | 26499a4b1d83 | NN occupancy coverage | 0.44629 (--) | -- | -- | 0.44629 | -- |
| RWFlow | unspecified | 26499a4b1d83 | 1-NNA (ideal near 0.5) | 0.6123 (--) | -- | -- | 0.6123 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Recorded training seconds | 112.22 (--) | -- | -- | 112.22 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Biased kernel MMD2_V (primary) | 0.0067758 (--) | -- | -- | 0.0067758 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Symmetric NN distance | 0.10901 (--) | -- | -- | 0.10901 | -- |
| KGDM | unspecified | ce3fc1c8b63c | MinimumMatchingDistance (directed NN) | 0.053705 (--) | -- | -- | 0.053705 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Radius coverage | 0.42188 (--) | -- | -- | 0.42188 | -- |
| KGDM | unspecified | ce3fc1c8b63c | NN occupancy coverage | 0.43262 (--) | -- | -- | 0.43262 | -- |
| KGDM | unspecified | ce3fc1c8b63c | 1-NNA (ideal near 0.5) | 0.6416 (--) | -- | -- | 0.6416 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Recorded training seconds | 8.0906 (--) | -- | -- | 8.0906 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Biased kernel MMD2_V (primary) | 0.0096746 (--) | -- | -- | 0.0096746 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Symmetric NN distance | 0.13617 (--) | -- | -- | 0.13617 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | MinimumMatchingDistance (directed NN) | 0.076166 (--) | -- | -- | 0.076166 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Radius coverage | 0.3125 (--) | -- | -- | 0.3125 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | NN occupancy coverage | 0.31738 (--) | -- | -- | 0.31738 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | 1-NNA (ideal near 0.5) | 0.74902 (--) | -- | -- | 0.74902 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Recorded training seconds | 30.905 (--) | -- | -- | 30.905 | -- |

## Configuration Ledger

### KGDM / unspecified / calibration_v2 / ce3fc1c8b63c

```json
{
  "config_id": "ce3fc1c8b63c783904338f170d94b9a49e448b3892f8d2acfb8719d72ca621cb",
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

### MeanFlow / rmf_mt / calibration_v2 / 3354b88f1ac7

```json
{
  "config_id": "3354b88f1ac746267d08d5485a50745908764c89cd8a9dae8d9fc94ee62328f0",
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

### MGM / unspecified / mgm_calibration_v1 / 4b4c7a1b47aa

```json
{
  "config_id": "4b4c7a1b47aaaba1ccf55170996b96d0105d1faae428e9050aba6c04d42b649b",
  "config": {
    "batch_size": 256,
    "critic_steps": 5,
    "critic_warmup": 500,
    "depth": 3,
    "harmonics": 4,
    "latent_dim": 16,
    "lr": 0.0002,
    "lr_c": 0.0002,
    "noise": 0.7,
    "width": 128
  },
  "resolved_config": null
}
```

### RWFlow / unspecified / calibration_v2 / 26499a4b1d83

```json
{
  "config_id": "26499a4b1d8386ce1412182c47459b9a908e43e9a611bad03b55f568646c3bbd",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "epsilon": 0.3,
    "flow_step": 0.2,
    "latent_dim": 16,
    "lr": 0.0002,
    "sinkhorn_iters": 128,
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
    "source_batch_size": null,
    "weight_decay": 0.01,
    "width": 128
  }
}
```

