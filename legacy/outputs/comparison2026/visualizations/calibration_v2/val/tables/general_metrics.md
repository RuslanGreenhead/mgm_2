# VALIDATION ONLY: selected-candidate preview

Draws are averaged within each training seed first. SD is sample SD across training seeds, not across draws.
Missing seeds remain missing; one seed has no estimable seed SD. No convergence or significance claim is made.

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

## General

| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Seed SD |
| --- | --- | --- | --- | --- | --- | --- |
| MGM | kernel_mmd2_v | 0.010072 | -- | -- | 0.010072 | -- |
| MGM | symmetric_nn | 0.11144 | -- | -- | 0.11144 | -- |
| MGM | minimum_matching_distance | 0.051733 | -- | -- | 0.051733 | -- |
| MGM | radius_coverage | 0.4082 | -- | -- | 0.4082 | -- |
| MGM | nn_occupancy_coverage | 0.41797 | -- | -- | 0.41797 | -- |
| MGM | one_nna | 0.65869 | -- | -- | 0.65869 | -- |
| MGM | constraint_mean | 0 | -- | -- | 0 | -- |
| MGM | train_seconds | 26.452 | -- | -- | 26.452 | -- |
| RWFlow | kernel_mmd2_v | -- | -- | -- | -- | -- |
| RWFlow | symmetric_nn | -- | -- | -- | -- | -- |
| RWFlow | minimum_matching_distance | -- | -- | -- | -- | -- |
| RWFlow | radius_coverage | -- | -- | -- | -- | -- |
| RWFlow | nn_occupancy_coverage | -- | -- | -- | -- | -- |
| RWFlow | one_nna | -- | -- | -- | -- | -- |
| RWFlow | constraint_mean | -- | -- | -- | -- | -- |
| RWFlow | train_seconds | -- | -- | -- | -- | -- |
| KGDM | kernel_mmd2_v | -- | -- | -- | -- | -- |
| KGDM | symmetric_nn | -- | -- | -- | -- | -- |
| KGDM | minimum_matching_distance | -- | -- | -- | -- | -- |
| KGDM | radius_coverage | -- | -- | -- | -- | -- |
| KGDM | nn_occupancy_coverage | -- | -- | -- | -- | -- |
| KGDM | one_nna | -- | -- | -- | -- | -- |
| KGDM | constraint_mean | -- | -- | -- | -- | -- |
| KGDM | train_seconds | -- | -- | -- | -- | -- |
| MeanFlow | kernel_mmd2_v | -- | -- | -- | -- | -- |
| MeanFlow | symmetric_nn | -- | -- | -- | -- | -- |
| MeanFlow | minimum_matching_distance | -- | -- | -- | -- | -- |
| MeanFlow | radius_coverage | -- | -- | -- | -- | -- |
| MeanFlow | nn_occupancy_coverage | -- | -- | -- | -- | -- |
| MeanFlow | one_nna | -- | -- | -- | -- | -- |
| MeanFlow | constraint_mean | -- | -- | -- | -- | -- |
| MeanFlow | train_seconds | -- | -- | -- | -- | -- |
