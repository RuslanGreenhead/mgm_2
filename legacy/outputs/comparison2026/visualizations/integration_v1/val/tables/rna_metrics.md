# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. SD is sample SD across training seeds, not across draws.
Missing seeds remain missing; one seed has no estimable seed SD. No convergence or significance claim is made.

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

## RNA (7D)

| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Seed SD |
| --- | --- | --- | --- | --- | --- | --- |
| MGM | kernel_mmd2_v | -- | -- | -- | -- | -- |
| MGM | symmetric_nn | -- | -- | -- | -- | -- |
| MGM | minimum_matching_distance | -- | -- | -- | -- | -- |
| MGM | radius_coverage | -- | -- | -- | -- | -- |
| MGM | nn_occupancy_coverage | -- | -- | -- | -- | -- |
| MGM | one_nna | -- | -- | -- | -- | -- |
| MGM | constraint_mean | -- | -- | -- | -- | -- |
| MGM | train_seconds | -- | -- | -- | -- | -- |
| RWFlow | kernel_mmd2_v | 0.44895 | -- | -- | 0.44895 | -- |
| RWFlow | symmetric_nn | 2.1161 | -- | -- | 2.1161 | -- |
| RWFlow | minimum_matching_distance | 1.4118 | -- | -- | 1.4118 | -- |
| RWFlow | radius_coverage | 0 | -- | -- | 0 | -- |
| RWFlow | nn_occupancy_coverage | 0.19747 | -- | -- | 0.19747 | -- |
| RWFlow | one_nna | 0.95354 | -- | -- | 0.95354 | -- |
| RWFlow | constraint_mean | 0 | -- | -- | 0 | -- |
| RWFlow | train_seconds | 1.6809 | -- | -- | 1.6809 | -- |
| KGDM | kernel_mmd2_v | 0.45659 | -- | -- | 0.45659 | -- |
| KGDM | symmetric_nn | 2.3275 | -- | -- | 2.3275 | -- |
| KGDM | minimum_matching_distance | 1.6246 | -- | -- | 1.6246 | -- |
| KGDM | radius_coverage | 0 | -- | -- | 0 | -- |
| KGDM | nn_occupancy_coverage | 0.18057 | -- | -- | 0.18057 | -- |
| KGDM | one_nna | 0.96832 | -- | -- | 0.96832 | -- |
| KGDM | constraint_mean | 0 | -- | -- | 0 | -- |
| KGDM | train_seconds | 0.31373 | -- | -- | 0.31373 | -- |
| MeanFlow | kernel_mmd2_v | 0.43302 | -- | -- | 0.43302 | -- |
| MeanFlow | symmetric_nn | 1.8758 | -- | -- | 1.8758 | -- |
| MeanFlow | minimum_matching_distance | 1.1055 | -- | -- | 1.1055 | -- |
| MeanFlow | radius_coverage | 0 | -- | -- | 0 | -- |
| MeanFlow | nn_occupancy_coverage | 0.21542 | -- | -- | 0.21542 | -- |
| MeanFlow | one_nna | 0.93453 | -- | -- | 0.93453 | -- |
| MeanFlow | constraint_mean | 0 | -- | -- | 0 | -- |
| MeanFlow | train_seconds | 0.91892 | -- | -- | 0.91892 | -- |
