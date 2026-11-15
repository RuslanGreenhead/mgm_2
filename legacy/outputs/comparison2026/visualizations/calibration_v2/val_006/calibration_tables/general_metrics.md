# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

All completed candidates are retained below; selected_for_comparison marks the density-preview choices. These are post-selection validation previews, not test results.

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
| MGM | unspecified | aafff7b0b635 | Biased kernel MMD2_V (primary) | 0.020386 (--) | -- | -- | 0.020386 | -- |
| MGM | unspecified | aafff7b0b635 | Symmetric NN distance | 0.14306 (--) | -- | -- | 0.14306 | -- |
| MGM | unspecified | aafff7b0b635 | MinimumMatchingDistance (directed NN) | 0.062798 (--) | -- | -- | 0.062798 | -- |
| MGM | unspecified | aafff7b0b635 | Radius coverage | 0.31934 (--) | -- | -- | 0.31934 | -- |
| MGM | unspecified | aafff7b0b635 | NN occupancy coverage | 0.35059 (--) | -- | -- | 0.35059 | -- |
| MGM | unspecified | aafff7b0b635 | 1-NNA (ideal near 0.5) | 0.70459 (--) | -- | -- | 0.70459 | -- |
| MGM | unspecified | aafff7b0b635 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MGM | unspecified | aafff7b0b635 | Recorded training seconds | 6.9197 (--) | -- | -- | 6.9197 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Biased kernel MMD2_V (primary) | 0.0017461 (--) | -- | -- | 0.0017461 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Symmetric NN distance | 0.084668 (--) | -- | -- | 0.084668 | -- |
| RWFlow | unspecified | 26499a4b1d83 | MinimumMatchingDistance (directed NN) | 0.050271 (--) | -- | -- | 0.050271 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Radius coverage | 0.44824 (--) | -- | -- | 0.44824 | -- |
| RWFlow | unspecified | 26499a4b1d83 | NN occupancy coverage | 0.44629 (--) | -- | -- | 0.44629 | -- |
| RWFlow | unspecified | 26499a4b1d83 | 1-NNA (ideal near 0.5) | 0.6123 (--) | -- | -- | 0.6123 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Recorded training seconds | 112.22 (--) | -- | -- | 112.22 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Biased kernel MMD2_V (primary) | 0.0020674 (--) | -- | -- | 0.0020674 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Symmetric NN distance | 0.092911 (--) | -- | -- | 0.092911 | -- |
| RWFlow | unspecified | 82707b9f60f4 | MinimumMatchingDistance (directed NN) | 0.053584 (--) | -- | -- | 0.053584 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Radius coverage | 0.44043 (--) | -- | -- | 0.44043 | -- |
| RWFlow | unspecified | 82707b9f60f4 | NN occupancy coverage | 0.44727 (--) | -- | -- | 0.44727 | -- |
| RWFlow | unspecified | 82707b9f60f4 | 1-NNA (ideal near 0.5) | 0.63721 (--) | -- | -- | 0.63721 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Recorded training seconds | 112.1 (--) | -- | -- | 112.1 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Biased kernel MMD2_V (primary) | 0.0017465 (--) | -- | -- | 0.0017465 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Symmetric NN distance | 0.084669 (--) | -- | -- | 0.084669 | -- |
| RWFlow | unspecified | d74bc5e59dfc | MinimumMatchingDistance (directed NN) | 0.050269 (--) | -- | -- | 0.050269 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Radius coverage | 0.44531 (--) | -- | -- | 0.44531 | -- |
| RWFlow | unspecified | d74bc5e59dfc | NN occupancy coverage | 0.44629 (--) | -- | -- | 0.44629 | -- |
| RWFlow | unspecified | d74bc5e59dfc | 1-NNA (ideal near 0.5) | 0.61328 (--) | -- | -- | 0.61328 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Recorded training seconds | 128.27 (--) | -- | -- | 128.27 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Biased kernel MMD2_V (primary) | 0.002067 (--) | -- | -- | 0.002067 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Symmetric NN distance | 0.092904 (--) | -- | -- | 0.092904 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | MinimumMatchingDistance (directed NN) | 0.053586 (--) | -- | -- | 0.053586 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Radius coverage | 0.44043 (--) | -- | -- | 0.44043 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | NN occupancy coverage | 0.44727 (--) | -- | -- | 0.44727 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | 1-NNA (ideal near 0.5) | 0.6377 (--) | -- | -- | 0.6377 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Recorded training seconds | 112.2 (--) | -- | -- | 112.2 | -- |
| KGDM | unspecified | 5cb92833fb89 | Biased kernel MMD2_V (primary) | 0.011011 (--) | -- | -- | 0.011011 | -- |
| KGDM | unspecified | 5cb92833fb89 | Symmetric NN distance | 0.11816 (--) | -- | -- | 0.11816 | -- |
| KGDM | unspecified | 5cb92833fb89 | MinimumMatchingDistance (directed NN) | 0.051834 (--) | -- | -- | 0.051834 | -- |
| KGDM | unspecified | 5cb92833fb89 | Radius coverage | 0.42676 (--) | -- | -- | 0.42676 | -- |
| KGDM | unspecified | 5cb92833fb89 | NN occupancy coverage | 0.43945 (--) | -- | -- | 0.43945 | -- |
| KGDM | unspecified | 5cb92833fb89 | 1-NNA (ideal near 0.5) | 0.63525 (--) | -- | -- | 0.63525 | -- |
| KGDM | unspecified | 5cb92833fb89 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| KGDM | unspecified | 5cb92833fb89 | Recorded training seconds | 8.1417 (--) | -- | -- | 8.1417 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Biased kernel MMD2_V (primary) | 0.0067758 (--) | -- | -- | 0.0067758 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Symmetric NN distance | 0.10901 (--) | -- | -- | 0.10901 | -- |
| KGDM | unspecified | ce3fc1c8b63c | MinimumMatchingDistance (directed NN) | 0.053705 (--) | -- | -- | 0.053705 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Radius coverage | 0.42188 (--) | -- | -- | 0.42188 | -- |
| KGDM | unspecified | ce3fc1c8b63c | NN occupancy coverage | 0.43262 (--) | -- | -- | 0.43262 | -- |
| KGDM | unspecified | ce3fc1c8b63c | 1-NNA (ideal near 0.5) | 0.6416 (--) | -- | -- | 0.6416 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Recorded training seconds | 8.0906 (--) | -- | -- | 8.0906 | -- |
| KGDM | unspecified | defcfe078fb4 | Biased kernel MMD2_V (primary) | 0.011638 (--) | -- | -- | 0.011638 | -- |
| KGDM | unspecified | defcfe078fb4 | Symmetric NN distance | 0.11678 (--) | -- | -- | 0.11678 | -- |
| KGDM | unspecified | defcfe078fb4 | MinimumMatchingDistance (directed NN) | 0.04971 (--) | -- | -- | 0.04971 | -- |
| KGDM | unspecified | defcfe078fb4 | Radius coverage | 0.43652 (--) | -- | -- | 0.43652 | -- |
| KGDM | unspecified | defcfe078fb4 | NN occupancy coverage | 0.44434 (--) | -- | -- | 0.44434 | -- |
| KGDM | unspecified | defcfe078fb4 | 1-NNA (ideal near 0.5) | 0.61963 (--) | -- | -- | 0.61963 | -- |
| KGDM | unspecified | defcfe078fb4 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| KGDM | unspecified | defcfe078fb4 | Recorded training seconds | 8.2715 (--) | -- | -- | 8.2715 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Biased kernel MMD2_V (primary) | 0.0096746 (--) | -- | -- | 0.0096746 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Symmetric NN distance | 0.13617 (--) | -- | -- | 0.13617 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | MinimumMatchingDistance (directed NN) | 0.076166 (--) | -- | -- | 0.076166 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Radius coverage | 0.3125 (--) | -- | -- | 0.3125 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | NN occupancy coverage | 0.31738 (--) | -- | -- | 0.31738 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | 1-NNA (ideal near 0.5) | 0.74902 (--) | -- | -- | 0.74902 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Recorded training seconds | 30.905 (--) | -- | -- | 30.905 | -- |
| MeanFlow | rmf | f07b798671bd | Biased kernel MMD2_V (primary) | 0.010881 (--) | -- | -- | 0.010881 | -- |
| MeanFlow | rmf | f07b798671bd | Symmetric NN distance | 0.13655 (--) | -- | -- | 0.13655 | -- |
| MeanFlow | rmf | f07b798671bd | MinimumMatchingDistance (directed NN) | 0.081979 (--) | -- | -- | 0.081979 | -- |
| MeanFlow | rmf | f07b798671bd | Radius coverage | 0.33105 (--) | -- | -- | 0.33105 | -- |
| MeanFlow | rmf | f07b798671bd | NN occupancy coverage | 0.31055 (--) | -- | -- | 0.31055 | -- |
| MeanFlow | rmf | f07b798671bd | 1-NNA (ideal near 0.5) | 0.73877 (--) | -- | -- | 0.73877 | -- |
| MeanFlow | rmf | f07b798671bd | Mean constraint residual | 0 (--) | -- | -- | 0 | -- |
| MeanFlow | rmf | f07b798671bd | Recorded training seconds | 26.19 (--) | -- | -- | 26.19 | -- |

## Configuration Ledger

### KGDM / unspecified / calibration_v2 / 5cb92833fb89

```json
{
  "config_id": "5cb92833fb89dd1e2ddf89720c3f241c59af85c259b5fee316868f7c655d864d",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "kernel": "laplace",
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
    "kernel": "laplace",
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

### KGDM / unspecified / calibration_v2 / defcfe078fb4

```json
{
  "config_id": "defcfe078fb4df41f534a8b9a89051250579576c5e65ef099cb633ce937b43db",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "kernel": "laplace",
    "latent_dim": 16,
    "lr": 0.0002,
    "step_size": 0.5,
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
    "kernel": "laplace",
    "latent_dim": 16,
    "lr": 0.0002,
    "max_step": 1.0,
    "sample_ema": true,
    "step_size": 0.5,
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

### MeanFlow / rmf / calibration_v2 / f07b798671bd

```json
{
  "config_id": "f07b798671bddb420858bd72040f07942d87e3410cce36f7f3d581795de11c59",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "latent_dim": 16,
    "lr": 0.0002,
    "variant": "rmf",
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
    "variant": "rmf",
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

### RWFlow / unspecified / calibration_v2 / 82707b9f60f4

```json
{
  "config_id": "82707b9f60f47ef721a4376d958ec606d07183bd6aa401eb723ee9329a5379da",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "epsilon": 1.0,
    "flow_step": 0.05,
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
    "epsilon": 1.0,
    "flow_step": 0.05,
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

### RWFlow / unspecified / calibration_v2 / d74bc5e59dfc

```json
{
  "config_id": "d74bc5e59dfcab800180600672965a1fe32d7678a639ab10ba779f41bc88092b",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "epsilon": 0.3,
    "flow_step": 0.05,
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
    "flow_step": 0.05,
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

### RWFlow / unspecified / calibration_v2 / ed9b5373b9b2

```json
{
  "config_id": "ed9b5373b9b2ee10e9c34f4cefd4a6bc6a0dc7448095cc047ae49eea9ac68deb",
  "config": {
    "batch_size": 256,
    "depth": 3,
    "epsilon": 1.0,
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
    "epsilon": 1.0,
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

