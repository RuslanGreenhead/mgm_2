# Final R-MGM / RFM Reproduction Status

This report aggregates completed local GPU runs only. Lower is better for `mmd2` and `symmetric_nn`; higher is better for `coverage_at_radius`.

Important caveat: official RFM author-budget training is audited but not completed here. Earth/Torus author configs require 1,000,000 steps; mesh/appendix official configs require 50,000-100,000 steps. The RFM rows below are the shared sample-quality baseline runs from `rmgm_train_real.py`, not the full Hydra likelihood runs from `train.py`.

## earth_real

| data | dim | method | steps | best mmd2 | final mmd2 | best symNN | final symNN | best coverage | constraint | run |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| earthquake | 3 | rfm | 600 | 0.00505757 | 0.00796962 | 0.100822 | 0.111013 | 0.184641 | 1.17e-09 | `seed0_20260801_001202` |
| earthquake | 3 | rmgm | 600 | 0.00107294 | 0.00298291 | 0.04451 | 0.0451453 | 0.344771 | 3.51e-09 | `seed0_20260801_001611` |
| fire | 3 | rfm | 600 | 0.00655013 | 0.00792766 | 0.091278 | 0.0934155 | 0.0640625 | 2.33e-10 | `seed0_20260801_001217` |
| fire | 3 | rmgm | 600 | 0.000256658 | 0.000898898 | 0.0257194 | 0.0257194 | 0.350781 | 1.77e-09 | `seed0_20260801_001621` |
| flood | 3 | rfm | 600 | 0.00439334 | 0.00439334 | 0.103538 | 0.106989 | 0.219713 | 8.57e-10 | `seed0_20260801_001209` |
| flood | 3 | rmgm | 600 | 0.00165707 | 0.00247413 | 0.0548751 | 0.0584343 | 0.36345 | 2.33e-09 | `seed0_20260801_001616` |
| volcano | 3 | rfm | 600 | 0.01056 | 0.0160875 | 0.196757 | 0.226768 | 0.0365854 | 0 | `seed0_20260801_001155` |
| volcano | 3 | rmgm | 600 | 0.00931352 | 0.043038 | 0.138423 | 0.150964 | 0.292683 | 2.91e-09 | `seed0_20260801_001604` |

## torus_real

| data | dim | method | steps | best mmd2 | final mmd2 | best symNN | final symNN | best coverage | constraint | run |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| general | 2 | rfm | 800 | 0.0362966 | 0.0479921 | 0.0979873 | 0.0979873 | 0.480469 | 0 | `seed0_20260801_002732` |
| general | 2 | rmgm | 800 | 0.000354528 | 0.0021286 | 0.0205642 | 0.0217387 | 0.74585 | 0 | `seed0_20260801_002726` |
| glycine | 2 | rfm | 800 | 0.0212036 | 0.0226241 | 0.163508 | 0.170856 | 0.234187 | 0 | `seed0_20260801_002739` |
| glycine | 2 | rmgm | 800 | 0.000199914 | 0.00131011 | 0.0528768 | 0.055145 | 0.633283 | 0 | `seed0_20260801_002735` |
| prepro | 2 | rfm | 800 | 0.0206762 | 0.0206762 | 0.126203 | 0.126203 | 0.41534 | 0 | `seed0_20260801_002753` |
| prepro | 2 | rmgm | 800 | 0.000840068 | 0.0028913 | 0.0447913 | 0.0452981 | 0.720695 | 0 | `seed0_20260801_002748` |
| proline | 2 | rfm | 800 | 0.0657533 | 0.0657533 | 0.200614 | 0.200614 | 0.246396 | 0 | `seed0_20260801_002745` |
| proline | 2 | rmgm | 800 | 5.67436e-05 | 0.00293124 | 0.0292187 | 0.0352857 | 0.639581 | 0 | `seed0_20260801_002741` |
| rna | 7 | rfm | 800 | 0.097817 | 0.097817 | 0.885173 | 0.885173 | 0.00527983 | 0 | `seed0_20260801_002845` |
| rna | 7 | rmgm | 800 | 0.00174743 | 0.00198722 | 0.405805 | 0.417441 | 0.312566 | 0 | `seed0_20260801_002840` |

## extra_pilot

| data | dim | method | steps | best mmd2 | final mmd2 | best symNN | final symNN | best coverage | constraint | run |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| hyperbolic | 2 | rfm | 2000 | 0.0114242 | 0.0150757 | 0.0243751 | 0.0268957 | 0.547 | 0 | `seed0_20260801_094407` |
| hyperbolic | 2 | rmgm | 2000 | 0.00584412 | 0.0107812 | 0.00891979 | 0.00891979 | 0.6535 | 0 | `seed0_20260801_094308` |
| maze3v2 | 3 | rfm | 500 | 0.121017 | 0.121017 | 0.147288 | 0.147288 | 0.0078125 | 0 | `seed0_20260801_094516` |
| maze3v2 | 3 | rmgm | 500 | 0.000899673 | 0.000899673 | 0.00477163 | 0.00516801 | 0.365234 | 0 | `seed0_20260801_094626` |
| maze4v2 | 3 | rfm | 500 | 0.0536802 | 0.0612944 | 0.126465 | 0.132143 | 0.00976562 | 0 | `seed0_20260801_094709` |
| maze4v2 | 3 | rmgm | 500 | 0.000617623 | 0.00500345 | 0.00706371 | 0.00780095 | 0.261719 | 0 | `seed0_20260801_094648` |
| simple_bunny | 3 | rfm | 500 | 0.509942 | 0.512111 | 0.309692 | 0.309692 | 0.00390625 | 9.7e-10 | `seed0_20260801_095649` |
| simple_bunny | 3 | rmgm | 500 | 0.00229335 | 0.00229335 | 0.00481769 | 0.00511378 | 0.458984 | 1.43e-09 | `seed0_20260801_095629` |
| wrapped_spd | 3 | rfm | 2000 | 0.00509578 | 0.00509578 | 0.112248 | 0.112248 | 0.556 | 0 | `seed0_20260801_094221` |
| wrapped_spd | 3 | rmgm | 2000 | 0.00453174 | 0.0095858 | 0.111952 | 0.114428 | 0.511 | 0 | `seed0_20260801_094103` |
| wrapped_torus | 5 | rfm | 2000 | 0.191247 | 0.191247 | 0.529927 | 0.529927 | 0.320801 | 0 | `seed0_20260801_094039` |
| wrapped_torus | 5 | rmgm | 2000 | 0.000696242 | 0.000696242 | 0.120988 | 0.121835 | 0.677246 | 0 | `seed0_20260801_094013` |

## scaling_pilot

| data | dim | method | steps | best mmd2 | final mmd2 | best symNN | final symNN | best coverage | constraint | run |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| wrapped_spd | 6 | rfm | 1000 | 0.0101262 | 0.0101262 | 0.388518 | 0.396239 | 0.512 | 0 | `seed0_20260801_094950` |
| wrapped_spd | 6 | rmgm | 1000 | 0.00870085 | 0.00870085 | 0.413695 | 0.41374 | 0.445 | 0 | `seed0_20260801_095105` |
| wrapped_spd | 10 | rfm | 1000 | 0.00674921 | 0.00674921 | 0.74143 | 0.741737 | 0.458 | 0 | `seed0_20260801_095323` |
| wrapped_spd | 10 | rmgm | 1000 | 0.0169346 | 0.0169346 | 0.80476 | 0.80476 | 0.389 | 0 | `seed0_20260801_095227` |
| wrapped_torus | 10 | rfm | 1000 | 0.404441 | 0.404441 | 1.66424 | 1.66424 | 0 | 0 | `seed0_20260801_094824` |
| wrapped_torus | 10 | rmgm | 1000 | 0.000884891 | 0.00104922 | 0.361661 | 0.363944 | 0.521484 | 0 | `seed0_20260801_094742` |
| wrapped_torus | 20 | rfm | 1000 | 0.497898 | 0.497898 | 3.62842 | 3.62842 | 0 | 0 | `seed0_20260801_094841` |
| wrapped_torus | 20 | rmgm | 1000 | 0.00086695 | 0.00095278 | 0.720346 | 0.7266 | 0.507812 | 0 | `seed0_20260801_094803` |

## Official RFM Audit

- Resolved author configs are in `outputs/rmgm_summary/original_rfm_config_audit.md`.
- Mesh dependency/data status is in `outputs/rmgm_summary/mesh_pipeline_status.md`.
- Extra-geometry command audit is in `outputs/rmgm_summary/extra_geometry_plan.md`.
- Boundary mesh arrays generated locally: `bunny_simple.npy`, `maze_3x3v2.npz`, `maze_4x4v2.npz`.
- Full official RFM mesh smoke through `train.py` reached validation but was stopped after 10 minutes because validation over the full 10k split dominates even at one training step.
