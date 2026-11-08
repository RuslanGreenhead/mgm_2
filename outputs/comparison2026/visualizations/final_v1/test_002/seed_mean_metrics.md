# TEST: frozen configurations, validation-selected checkpoints

Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).
Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.
Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.
The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).

| Dataset | Method | Variant | Phase | Config ID | Seeds | Missing seeds | Biased kernel MMD2_V mean | Training-seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Volcano | MGM | unspecified | final_v1 | 461f24fd75e8 | 0,1,2 | none | 0.013135 | 0.0015554 |
| Volcano | RWFlow | unspecified | final_v1 | d58e75450893 | 0,1,2 | none | 0.012537 | 0.0002771 |
| Volcano | KGDM | unspecified | final_v1 | 46b998681a6e | 0,1,2 | none | 0.0139 | 0.0016943 |
| Volcano | MeanFlow | rmf_mt | final_v1 | 57e0297c5365 | 0,1,2 | none | 0.011957 | 0.00027519 |
| Earthquake | MGM | -- | -- | -- | -- | 0,1,2 | -- | -- |
| Earthquake | RWFlow | -- | -- | -- | -- | 0,1,2 | -- | -- |
| Earthquake | KGDM | unspecified | final_v1 | 46b998681a6e | 0,1,2 | none | 0.0021325 | 0.00011537 |
| Earthquake | MeanFlow | rmf_mt | final_v1 | 57e0297c5365 | 0,1,2 | none | 0.0028137 | 0.0006936 |
