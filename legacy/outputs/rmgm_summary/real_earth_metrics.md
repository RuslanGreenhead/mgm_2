# Real Earth/Sphere Benchmark

Selection: best evaluated checkpoint by MMD2.

| dataset | method | step | MMD2 | sym. NN | coverage | constraint |
|---|---:|---:|---:|---:|---:|---:|
| volcano | rmgm | 100 | 0.0104176 | 0.0933728 | 0.537 | 1.45e-09 |
| volcano | rfm | 450 | 0.01056 | 0.196757 | 0.000 | 0 |
| earthquake | rmgm | 150 | 0.00107294 | 0.0484681 | 0.345 | 4.48e-09 |
| earthquake | rfm | 300 | 0.00505757 | 0.111052 | 0.132 | 1.36e-09 |
| flood | rmgm | 450 | 0.00165707 | 0.0559093 | 0.318 | 1.84e-09 |
| flood | rfm | 600 | 0.00439334 | 0.106989 | 0.209 | 8.57e-10 |
| fire | rmgm | 300 | 0.000256658 | 0.0275002 | 0.344 | 2.79e-09 |
| fire | rfm | 450 | 0.00655013 | 0.091278 | 0.048 | 4.19e-10 |

Metrics are computed against the held-out 10% split. Lower MMD2 and
symmetric NN are better; higher coverage is better. R-MGM does not
produce an exact likelihood, so this table is sample-quality based.
