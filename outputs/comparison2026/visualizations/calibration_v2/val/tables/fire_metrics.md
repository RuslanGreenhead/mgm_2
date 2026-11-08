# VALIDATION ONLY: selected-candidate preview

Draws are averaged within each training seed first. SD is sample SD across training seeds, not across draws.
Missing seeds remain missing; one seed has no estimable seed SD. No convergence or significance claim is made.

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

## Fire

| Method | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Seed SD |
| --- | --- | --- | --- | --- | --- | --- |
| MGM | kernel_mmd2_v | 0.002949 | -- | -- | 0.002949 | -- |
| MGM | symmetric_nn | 0.089439 | -- | -- | 0.089439 | -- |
| MGM | minimum_matching_distance | 0.050427 | -- | -- | 0.050427 | -- |
| MGM | radius_coverage | 0.036133 | -- | -- | 0.036133 | -- |
| MGM | nn_occupancy_coverage | 0.25391 | -- | -- | 0.25391 | -- |
| MGM | one_nna | 0.86865 | -- | -- | 0.86865 | -- |
| MGM | constraint_mean | 3.1178e-08 | -- | -- | 3.1178e-08 | -- |
| MGM | train_seconds | 48.916 | -- | -- | 48.916 | -- |
| RWFlow | kernel_mmd2_v | 0.0013123 | -- | -- | 0.0013123 | -- |
| RWFlow | symmetric_nn | 0.080901 | -- | -- | 0.080901 | -- |
| RWFlow | minimum_matching_distance | 0.03564 | -- | -- | 0.03564 | -- |
| RWFlow | radius_coverage | 0.020508 | -- | -- | 0.020508 | -- |
| RWFlow | nn_occupancy_coverage | 0.27832 | -- | -- | 0.27832 | -- |
| RWFlow | one_nna | 0.85303 | -- | -- | 0.85303 | -- |
| RWFlow | constraint_mean | 3.0605e-08 | -- | -- | 3.0605e-08 | -- |
| RWFlow | train_seconds | 116.95 | -- | -- | 116.95 | -- |
| KGDM | kernel_mmd2_v | 0.0078911 | -- | -- | 0.0078911 | -- |
| KGDM | symmetric_nn | 0.073129 | -- | -- | 0.073129 | -- |
| KGDM | minimum_matching_distance | 0.030141 | -- | -- | 0.030141 | -- |
| KGDM | radius_coverage | 0.043945 | -- | -- | 0.043945 | -- |
| KGDM | nn_occupancy_coverage | 0.29102 | -- | -- | 0.29102 | -- |
| KGDM | one_nna | 0.83789 | -- | -- | 0.83789 | -- |
| KGDM | constraint_mean | 3.1008e-08 | -- | -- | 3.1008e-08 | -- |
| KGDM | train_seconds | 21.214 | -- | -- | 21.214 | -- |
| MeanFlow | kernel_mmd2_v | 0.0091394 | -- | -- | 0.0091394 | -- |
| MeanFlow | symmetric_nn | 0.092043 | -- | -- | 0.092043 | -- |
| MeanFlow | minimum_matching_distance | 0.048123 | -- | -- | 0.048123 | -- |
| MeanFlow | radius_coverage | 0.022461 | -- | -- | 0.022461 | -- |
| MeanFlow | nn_occupancy_coverage | 0.24121 | -- | -- | 0.24121 | -- |
| MeanFlow | one_nna | 0.89307 | -- | -- | 0.89307 | -- |
| MeanFlow | constraint_mean | 2.9428e-08 | -- | -- | 2.9428e-08 | -- |
| MeanFlow | train_seconds | 19.681 | -- | -- | 19.681 | -- |
