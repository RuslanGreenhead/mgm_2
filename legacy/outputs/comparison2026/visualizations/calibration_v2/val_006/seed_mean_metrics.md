# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed. These are post-selection validation previews, not test results.

| Dataset | Method | Variant | Phase | Config ID | Seeds | Missing seeds | Biased kernel MMD2_V mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fire | MGM | unspecified | mgm_calibration_v1 | aafff7b0b635 | 0 | 1,2 | 0.002949 | -- |
| Fire | RWFlow | unspecified | calibration_v2 | 26499a4b1d83 | 0 | 1,2 | 0.0013122 | -- |
| Fire | KGDM | unspecified | calibration_v2 | ce3fc1c8b63c | 0 | 1,2 | 0.0078911 | -- |
| Fire | MeanFlow | rmf_mt | calibration_v2 | 3354b88f1ac7 | 0 | 1,2 | 0.0091394 | -- |
| General | MGM | unspecified | mgm_calibration_v1 | 4b4c7a1b47aa | 0 | 1,2 | 0.010072 | -- |
| General | RWFlow | unspecified | calibration_v2 | 26499a4b1d83 | 0 | 1,2 | 0.0017461 | -- |
| General | KGDM | unspecified | calibration_v2 | ce3fc1c8b63c | 0 | 1,2 | 0.0067758 | -- |
| General | MeanFlow | rmf_mt | calibration_v2 | 3354b88f1ac7 | 0 | 1,2 | 0.0096746 | -- |
| RNA (7D) | MGM | unspecified | mgm_calibration_v1 | aafff7b0b635 | 0 | 1,2 | 0.19841 | -- |
| RNA (7D) | RWFlow | -- | -- | -- | -- | 0,1,2 | -- | -- |
| RNA (7D) | KGDM | unspecified | calibration_v2 | ce3fc1c8b63c | 0 | 1,2 | 0.19622 | -- |
| RNA (7D) | MeanFlow | -- | -- | -- | -- | 0,1,2 | -- | -- |
