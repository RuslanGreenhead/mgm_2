# Real RFM-Style Experiments With R-MGM

This report summarizes the real-data pass run from the official
`facebookresearch/riemannian-fm` datasets/config map.  It uses the official
dataset loaders, the same deterministic 80/10/10 split pattern, and compares
one-step constrained R-MGM against the lightweight RFM sample baseline in this
workspace.

Important caveat: the RFM paper reports likelihoods for continuous normalizing
flows.  R-MGM here is an implicit one-step generator, so exact NLL is not a
like-for-like metric.  These tables therefore use sample-quality metrics on the
held-out test split: MMD2 in the natural embedding, symmetric geodesic nearest
neighbor distance, coverage at the empirical nearest-neighbor radius, and
constraint error.

## Earth/Sphere

Artifacts:
- `outputs/rmgm_summary/real_earth_metrics.md`
- `outputs/rmgm_summary/real_earth_metrics.json`
- `outputs/rmgm_summary/real_earth_contact_sheet.png`

Best evaluated checkpoint by MMD2:

| dataset | R-MGM MMD2 | RFM MMD2 | R-MGM symNN | RFM symNN | R-MGM coverage | RFM coverage |
|---|---:|---:|---:|---:|---:|---:|
| volcano | 0.010418 | 0.010560 | 0.093373 | 0.196757 | 0.537 | 0.000 |
| earthquake | 0.001073 | 0.005058 | 0.048468 | 0.111052 | 0.345 | 0.132 |
| flood | 0.001657 | 0.004393 | 0.055909 | 0.106989 | 0.318 | 0.209 |
| fire | 0.000257 | 0.006550 | 0.027500 | 0.091278 | 0.344 | 0.048 |

## Protein/RNA Torus

Artifacts:
- `outputs/rmgm_summary/real_torus_metrics.md`
- `outputs/rmgm_summary/real_torus_metrics.json`
- `outputs/rmgm_summary/real_torus_contact_sheet.png`

The Top500 and RNA torsion datasets were prepared from the official PDB lists:
- `data/top500/aggregated_angles.tsv`: 166,035 rows
- `data/rna/aggregated_angles.tsv`: 9,473 rows

Best evaluated checkpoint by MMD2:

| dataset | R-MGM MMD2 | RFM MMD2 | R-MGM symNN | RFM symNN | R-MGM coverage | RFM coverage |
|---|---:|---:|---:|---:|---:|---:|
| general | 0.000355 | 0.036297 | 0.020610 | 0.099239 | 0.738 | 0.480 |
| glycine | 0.000200 | 0.021204 | 0.052877 | 0.168261 | 0.620 | 0.181 |
| proline | 0.000057 | 0.065753 | 0.029219 | 0.200614 | 0.640 | 0.246 |
| prepro | 0.000840 | 0.020676 | 0.051217 | 0.126203 | 0.603 | 0.415 |
| rna | 0.001747 | 0.097817 | 0.409750 | 0.885173 | 0.247 | 0.005 |

## Remaining Paper Settings

Not fully run yet:
- mesh settings (`simple_bunny`, `bunny10/50/100`, `spot10/50/100`,
  `maze3v2`, `maze4v2`): require optional `igl` and generated `.npy/.npz`
  files from `data/synthesize_mesh_data.py`.
- appendix SPD/EEG settings: require extra EEG data not present in the
  downloaded RFM `data.zip`.
- high-dimensional wrapped torus/SPD: available as synthetic scalability
  experiments, not real-world datasets.
