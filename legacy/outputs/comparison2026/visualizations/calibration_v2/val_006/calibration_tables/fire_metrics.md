# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

All completed candidates are retained below; selected_for_comparison marks the density-preview choices. These are post-selection validation previews, not test results.

## Fire

| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGM | unspecified | aafff7b0b635 | Biased kernel MMD2_V (primary) | 0.002949 (--) | -- | -- | 0.002949 | -- |
| MGM | unspecified | aafff7b0b635 | Symmetric NN distance | 0.089439 (--) | -- | -- | 0.089439 | -- |
| MGM | unspecified | aafff7b0b635 | MinimumMatchingDistance (directed NN) | 0.050427 (--) | -- | -- | 0.050427 | -- |
| MGM | unspecified | aafff7b0b635 | Radius coverage | 0.036133 (--) | -- | -- | 0.036133 | -- |
| MGM | unspecified | aafff7b0b635 | NN occupancy coverage | 0.25391 (--) | -- | -- | 0.25391 | -- |
| MGM | unspecified | aafff7b0b635 | 1-NNA (ideal near 0.5) | 0.86865 (--) | -- | -- | 0.86865 | -- |
| MGM | unspecified | aafff7b0b635 | Mean constraint residual | 3.1178e-08 (--) | -- | -- | 3.1178e-08 | -- |
| MGM | unspecified | aafff7b0b635 | Recorded training seconds | 48.916 (--) | -- | -- | 48.916 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Biased kernel MMD2_V (primary) | 0.0013122 (--) | -- | -- | 0.0013122 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Symmetric NN distance | 0.080895 (--) | -- | -- | 0.080895 | -- |
| RWFlow | unspecified | 26499a4b1d83 | MinimumMatchingDistance (directed NN) | 0.035634 (--) | -- | -- | 0.035634 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Radius coverage | 0.020508 (--) | -- | -- | 0.020508 | -- |
| RWFlow | unspecified | 26499a4b1d83 | NN occupancy coverage | 0.27637 (--) | -- | -- | 0.27637 | -- |
| RWFlow | unspecified | 26499a4b1d83 | 1-NNA (ideal near 0.5) | 0.85205 (--) | -- | -- | 0.85205 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Mean constraint residual | 3.0388e-08 (--) | -- | -- | 3.0388e-08 | -- |
| RWFlow | unspecified | 26499a4b1d83 | Recorded training seconds | 335.06 (--) | -- | -- | 335.06 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Biased kernel MMD2_V (primary) | 0.0021233 (--) | -- | -- | 0.0021233 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Symmetric NN distance | 0.089036 (--) | -- | -- | 0.089036 | -- |
| RWFlow | unspecified | 82707b9f60f4 | MinimumMatchingDistance (directed NN) | 0.043507 (--) | -- | -- | 0.043507 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Radius coverage | 0.029297 (--) | -- | -- | 0.029297 | -- |
| RWFlow | unspecified | 82707b9f60f4 | NN occupancy coverage | 0.23828 (--) | -- | -- | 0.23828 | -- |
| RWFlow | unspecified | 82707b9f60f4 | 1-NNA (ideal near 0.5) | 0.87646 (--) | -- | -- | 0.87646 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Mean constraint residual | 3.1202e-08 (--) | -- | -- | 3.1202e-08 | -- |
| RWFlow | unspecified | 82707b9f60f4 | Recorded training seconds | 244.02 (--) | -- | -- | 244.02 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Biased kernel MMD2_V (primary) | 0.0013123 (--) | -- | -- | 0.0013123 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Symmetric NN distance | 0.080901 (--) | -- | -- | 0.080901 | -- |
| RWFlow | unspecified | d74bc5e59dfc | MinimumMatchingDistance (directed NN) | 0.03564 (--) | -- | -- | 0.03564 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Radius coverage | 0.020508 (--) | -- | -- | 0.020508 | -- |
| RWFlow | unspecified | d74bc5e59dfc | NN occupancy coverage | 0.27832 (--) | -- | -- | 0.27832 | -- |
| RWFlow | unspecified | d74bc5e59dfc | 1-NNA (ideal near 0.5) | 0.85303 (--) | -- | -- | 0.85303 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Mean constraint residual | 3.0605e-08 (--) | -- | -- | 3.0605e-08 | -- |
| RWFlow | unspecified | d74bc5e59dfc | Recorded training seconds | 116.95 (--) | -- | -- | 116.95 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Biased kernel MMD2_V (primary) | 0.0021227 (--) | -- | -- | 0.0021227 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Symmetric NN distance | 0.089038 (--) | -- | -- | 0.089038 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | MinimumMatchingDistance (directed NN) | 0.043515 (--) | -- | -- | 0.043515 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Radius coverage | 0.02832 (--) | -- | -- | 0.02832 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | NN occupancy coverage | 0.23828 (--) | -- | -- | 0.23828 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | 1-NNA (ideal near 0.5) | 0.87695 (--) | -- | -- | 0.87695 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Mean constraint residual | 3.3908e-08 (--) | -- | -- | 3.3908e-08 | -- |
| RWFlow | unspecified | ed9b5373b9b2 | Recorded training seconds | 118.47 (--) | -- | -- | 118.47 | -- |
| KGDM | unspecified | 5cb92833fb89 | Biased kernel MMD2_V (primary) | 0.0089455 (--) | -- | -- | 0.0089455 | -- |
| KGDM | unspecified | 5cb92833fb89 | Symmetric NN distance | 0.083793 (--) | -- | -- | 0.083793 | -- |
| KGDM | unspecified | 5cb92833fb89 | MinimumMatchingDistance (directed NN) | 0.034711 (--) | -- | -- | 0.034711 | -- |
| KGDM | unspecified | 5cb92833fb89 | Radius coverage | 0.081055 (--) | -- | -- | 0.081055 | -- |
| KGDM | unspecified | 5cb92833fb89 | NN occupancy coverage | 0.33691 (--) | -- | -- | 0.33691 | -- |
| KGDM | unspecified | 5cb92833fb89 | 1-NNA (ideal near 0.5) | 0.81055 (--) | -- | -- | 0.81055 | -- |
| KGDM | unspecified | 5cb92833fb89 | Mean constraint residual | 3.1063e-08 (--) | -- | -- | 3.1063e-08 | -- |
| KGDM | unspecified | 5cb92833fb89 | Recorded training seconds | 22.763 (--) | -- | -- | 22.763 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Biased kernel MMD2_V (primary) | 0.0078911 (--) | -- | -- | 0.0078911 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Symmetric NN distance | 0.073129 (--) | -- | -- | 0.073129 | -- |
| KGDM | unspecified | ce3fc1c8b63c | MinimumMatchingDistance (directed NN) | 0.030141 (--) | -- | -- | 0.030141 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Radius coverage | 0.043945 (--) | -- | -- | 0.043945 | -- |
| KGDM | unspecified | ce3fc1c8b63c | NN occupancy coverage | 0.29102 (--) | -- | -- | 0.29102 | -- |
| KGDM | unspecified | ce3fc1c8b63c | 1-NNA (ideal near 0.5) | 0.83789 (--) | -- | -- | 0.83789 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Mean constraint residual | 3.1008e-08 (--) | -- | -- | 3.1008e-08 | -- |
| KGDM | unspecified | ce3fc1c8b63c | Recorded training seconds | 21.214 (--) | -- | -- | 21.214 | -- |
| KGDM | unspecified | defcfe078fb4 | Biased kernel MMD2_V (primary) | 0.0086827 (--) | -- | -- | 0.0086827 | -- |
| KGDM | unspecified | defcfe078fb4 | Symmetric NN distance | 0.083095 (--) | -- | -- | 0.083095 | -- |
| KGDM | unspecified | defcfe078fb4 | MinimumMatchingDistance (directed NN) | 0.033225 (--) | -- | -- | 0.033225 | -- |
| KGDM | unspecified | defcfe078fb4 | Radius coverage | 0.086914 (--) | -- | -- | 0.086914 | -- |
| KGDM | unspecified | defcfe078fb4 | NN occupancy coverage | 0.34375 (--) | -- | -- | 0.34375 | -- |
| KGDM | unspecified | defcfe078fb4 | 1-NNA (ideal near 0.5) | 0.80908 (--) | -- | -- | 0.80908 | -- |
| KGDM | unspecified | defcfe078fb4 | Mean constraint residual | 3.0903e-08 (--) | -- | -- | 3.0903e-08 | -- |
| KGDM | unspecified | defcfe078fb4 | Recorded training seconds | 21.973 (--) | -- | -- | 21.973 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Biased kernel MMD2_V (primary) | 0.0091394 (--) | -- | -- | 0.0091394 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Symmetric NN distance | 0.092043 (--) | -- | -- | 0.092043 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | MinimumMatchingDistance (directed NN) | 0.048123 (--) | -- | -- | 0.048123 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Radius coverage | 0.022461 (--) | -- | -- | 0.022461 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | NN occupancy coverage | 0.24121 (--) | -- | -- | 0.24121 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | 1-NNA (ideal near 0.5) | 0.89307 (--) | -- | -- | 0.89307 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Mean constraint residual | 2.9428e-08 (--) | -- | -- | 2.9428e-08 | -- |
| MeanFlow | rmf_mt | 3354b88f1ac7 | Recorded training seconds | 19.681 (--) | -- | -- | 19.681 | -- |
| MeanFlow | rmf | f07b798671bd | Biased kernel MMD2_V (primary) | 0.0098744 (--) | -- | -- | 0.0098744 | -- |
| MeanFlow | rmf | f07b798671bd | Symmetric NN distance | 0.094183 (--) | -- | -- | 0.094183 | -- |
| MeanFlow | rmf | f07b798671bd | MinimumMatchingDistance (directed NN) | 0.052266 (--) | -- | -- | 0.052266 | -- |
| MeanFlow | rmf | f07b798671bd | Radius coverage | 0.015625 (--) | -- | -- | 0.015625 | -- |
| MeanFlow | rmf | f07b798671bd | NN occupancy coverage | 0.2334 (--) | -- | -- | 0.2334 | -- |
| MeanFlow | rmf | f07b798671bd | 1-NNA (ideal near 0.5) | 0.89209 (--) | -- | -- | 0.89209 | -- |
| MeanFlow | rmf | f07b798671bd | Mean constraint residual | 3.2562e-08 (--) | -- | -- | 3.2562e-08 | -- |
| MeanFlow | rmf | f07b798671bd | Recorded training seconds | 27.488 (--) | -- | -- | 27.488 | -- |

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

