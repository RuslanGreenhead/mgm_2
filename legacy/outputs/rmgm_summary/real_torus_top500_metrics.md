# Real Torus Top500 Benchmark

Selection: best evaluated checkpoint by MMD2.

| dataset | method | step | MMD2 | sym. NN | coverage | constraint |
|---|---:|---:|---:|---:|---:|---:|
| general | rmgm | 200 | 0.000354528 | 0.0206095 | 0.738 | 0 |
| general | rfm | 600 | 0.0362966 | 0.0992389 | 0.480 | 0 |
| glycine | rmgm | 200 | 0.000199914 | 0.0528768 | 0.620 | 0 |
| glycine | rfm | 600 | 0.0212036 | 0.168261 | 0.181 | 0 |
| proline | rmgm | 200 | 5.67436e-05 | 0.0292187 | 0.640 | 0 |
| proline | rfm | 800 | 0.0657533 | 0.200614 | 0.246 | 0 |
| prepro | rmgm | 1 | 0.000840068 | 0.0512173 | 0.603 | 0 |
| prepro | rfm | 800 | 0.0206762 | 0.126203 | 0.415 | 0 |

Metrics are computed against the held-out 10% split. Lower MMD2 and
symmetric NN are better; higher coverage is better. R-MGM does not
produce an exact likelihood, so this table is sample-quality based.
