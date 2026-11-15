# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

All completed candidates are retained below; selected_for_comparison marks the density-preview choices. These are post-selection validation previews, not test results.

## RNA (7D)

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | unspecified | 4b4c7a1b47aa | Biased kernel MMD2_V (primary) | 0.20092 (--) | -- | -- | 0.20092 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Symmetric NN distance | 0.9884 (--) | -- | -- | 0.9884 | -- |
| MGM | unspecified | 4b4c7a1b47aa | MinimumMatchingDistance (directed NN) | 0.54337 (--) | -- | -- | 0.54337 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Radius coverage | 0.012672 (--) | -- | -- | 0.012672 | -- |
| MGM | unspecified | 4b4c7a1b47aa | NN occupancy coverage | 0.31468 (--) | -- | -- | 0.31468 | -- |
| MGM | unspecified | 4b4c7a1b47aa | 1-NNA (ideal near 0.5) | 0.8944 (--) | -- | -- | 0.8944 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MGM | unspecified | 4b4c7a1b47aa | Recorded training seconds | 26.949 (--) | -- | -- | 26.949 | -- |
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
| KGDM | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| KGDM | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| KGDM | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| KGDM | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| KGDM | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| KGDM | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| KGDM | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| KGDM | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Biased kernel MMD2_V (primary) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Symmetric NN distance | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | MinimumMatchingDistance (directed NN) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Radius coverage | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | NN occupancy coverage | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | 1-NNA (ideal near 0.5) | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Mean constraint residual | -- | -- | -- | -- | -- |
| MeanFlow | -- | -- | Recorded training seconds | -- | -- | -- | -- | -- |

## Configuration Ledger

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

