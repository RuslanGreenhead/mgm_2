# VALIDATION ONLY: non-final tuning preview

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

All completed candidates are retained below; selected_for_comparison marks the density-preview choices. These are post-selection validation previews, not test results.

| Dataset | Method | Variant | Phase | Config ID | Seeds | Missing seeds | Biased kernel MMD2_V mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fire | MGM | unspecified | mgm_calibration_v1 | aafff7b0b635 | 0 | 1,2 | 0.002949 | -- |
| Fire | RWFlow | unspecified | calibration_v2 | 26499a4b1d83 | 0 | 1,2 | 0.0013122 | -- |
| Fire | RWFlow | unspecified | calibration_v2 | 82707b9f60f4 | 0 | 1,2 | 0.0021233 | -- |
| Fire | RWFlow | unspecified | calibration_v2 | d74bc5e59dfc | 0 | 1,2 | 0.0013123 | -- |
| Fire | RWFlow | unspecified | calibration_v2 | ed9b5373b9b2 | 0 | 1,2 | 0.0021227 | -- |
| Fire | KGDM | unspecified | calibration_v2 | 5cb92833fb89 | 0 | 1,2 | 0.0089455 | -- |
| Fire | KGDM | unspecified | calibration_v2 | ce3fc1c8b63c | 0 | 1,2 | 0.0078911 | -- |
| Fire | KGDM | unspecified | calibration_v2 | defcfe078fb4 | 0 | 1,2 | 0.0086827 | -- |
| Fire | MeanFlow | rmf_mt | calibration_v2 | 3354b88f1ac7 | 0 | 1,2 | 0.0091394 | -- |
| Fire | MeanFlow | rmf | calibration_v2 | f07b798671bd | 0 | 1,2 | 0.0098744 | -- |
| General | MGM | unspecified | mgm_calibration_v1 | 4b4c7a1b47aa | 0 | 1,2 | 0.010072 | -- |
| General | MGM | unspecified | mgm_calibration_v1 | aafff7b0b635 | 0 | 1,2 | 0.020386 | -- |
| General | RWFlow | unspecified | calibration_v2 | 26499a4b1d83 | 0 | 1,2 | 0.0017461 | -- |
| General | RWFlow | unspecified | calibration_v2 | d74bc5e59dfc | 0 | 1,2 | 0.0017465 | -- |
| General | KGDM | unspecified | calibration_v2 | 5cb92833fb89 | 0 | 1,2 | 0.011011 | -- |
| General | KGDM | unspecified | calibration_v2 | ce3fc1c8b63c | 0 | 1,2 | 0.0067758 | -- |
| General | KGDM | unspecified | calibration_v2 | defcfe078fb4 | 0 | 1,2 | 0.011638 | -- |
| General | MeanFlow | rmf_mt | calibration_v2 | 3354b88f1ac7 | 0 | 1,2 | 0.0096746 | -- |
| General | MeanFlow | rmf | calibration_v2 | f07b798671bd | 0 | 1,2 | 0.010881 | -- |
| RNA (7D) | MGM | unspecified | mgm_calibration_v1 | 4b4c7a1b47aa | 0 | 1,2 | 0.20092 | -- |
| RNA (7D) | MGM | unspecified | mgm_calibration_v1 | aafff7b0b635 | 0 | 1,2 | 0.19841 | -- |
| RNA (7D) | RWFlow | -- | -- | -- | -- | 0,1,2 | -- | -- |
| RNA (7D) | KGDM | -- | -- | -- | -- | 0,1,2 | -- | -- |
| RNA (7D) | MeanFlow | -- | -- | -- | -- | 0,1,2 | -- | -- |
