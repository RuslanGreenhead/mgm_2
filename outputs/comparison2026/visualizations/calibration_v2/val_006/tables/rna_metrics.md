# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

## RNA (7D)

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | unspecified | aafff7b0b635 | Biased kernel MMD2_V (primary) | 0.19841 (--) | -- | -- | 0.19841 | -- |
| MGM | unspecified | aafff7b0b635 | Symmetric NN distance | 0.97415 (--) | -- | -- | 0.97415 | -- |
| MGM | unspecified | aafff7b0b635 | MinimumMatchingDistance (directed NN) | 0.58787 (--) | -- | -- | 0.58787 | -- |
| MGM | unspecified | aafff7b0b635 | Radius coverage | 0 (--) | -- | -- | 0 | -- |
| MGM | unspecified | aafff7b0b635 | NN occupancy coverage | 0.32101 (--) | -- | -- | 0.32101 | -- |
| MGM | unspecified | aafff7b0b635 | 1-NNA (ideal near 0.5) | 0.88701 (--) | -- | -- | 0.88701 | -- |
| MGM | unspecified | aafff7b0b635 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MGM | unspecified | aafff7b0b635 | Recorded training seconds | 25.536 (--) | -- | -- | 25.536 | -- |
| RWFlow | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| RWFlow | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |
| KGDM | unspecified | ce3fc1c8b63c | Biased kernel MMD2_V (primary) | 0.19622 (--) | -- | -- | 0.19622 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Symmetric NN distance | 0.96234 (--) | -- | -- | 0.96234 | -- |
| KGDM | unspecified | ce3fc1c8b63c | MinimumMatchingDistance (directed NN) | 0.52078 (--) | -- | -- | 0.52078 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Radius coverage | 0.041183 (--) | -- | -- | 0.041183 | -- |
| KGDM | unspecified | ce3fc1c8b63c | NN occupancy coverage | 0.32629 (--) | -- | -- | 0.32629 | -- |
| KGDM | unspecified | ce3fc1c8b63c | 1-NNA (ideal near 0.5) | 0.8812 (--) | -- | -- | 0.8812 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Recorded training seconds | 8.4051 (--) | -- | -- | 8.4051 | -- |
| MeanFlow | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |

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

### MGM / unspecified / mgm_calibration_v1 / aafff7b0b635

```json
{
  "config_id": "aafff7b0b635a1d69fa0fc5cb4f7db683ae72f332a789f74386186fad7280ba9",
  "config": {
    "batch_size": 256,
    "critic_steps": 5,
    "critic_warmup": 500,
    "depth": 3,
    "harmonics": 4,
    "latent_dim": 16,
    "lr": 0.0002,
    "lr_c": 0.0002,
    "noise": 0.0,
    "width": 128
  },
  "resolved_config": null
}
```

