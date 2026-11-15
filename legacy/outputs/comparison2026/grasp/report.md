# SE(3) Grasp Pilot: Completed Results

This is a **within-object held-out-pose pilot on two known public ACRONYM
objects**, Mug and Table. It is not a full ACRONYM unseen-object experiment.
All nine fixed 10000-update GPU jobs and 24 evaluations completed. Recorded
training time summed across jobs is 22.06 minutes. Independent audits passed.

## Test Results

Means are balanced over the two objects within each seed. SD is sample SD
across three training seeds, not across the six object/seed pairs. Each object
has 512 generated poses, with separate unfiltered collision checks on 100.

| Method | NFE | Unbiased pose MMD2 | Thin-gripper static proxy (%) | Full Franka collision (%) |
| --- | ---: | ---: | ---: | ---: |
| MGM | 1 | 0.004265 +/- 0.001591 | 24.67 +/- 1.76 | 71.00 +/- 5.22 |
| RMF-MT adaptation | 1 | 0.012096 +/- 0.001458 | 10.83 +/- 2.47 | 34.17 +/- 0.29 |
| RFM | 50 | 0.000530 +/- 0.000099 | 32.17 +/- 3.51 | 67.00 +/- 2.50 |
| RFM | 100 | 0.000515 +/- 0.000119 | 32.00 +/- 3.28 | 66.67 +/- 2.25 |

Lower MMD and full-mesh collision are better; higher static-proxy fraction is
better. These scores measure different properties and do not establish physical
grasp success. Reference held-out poses score 92%/100% on the static proxy for
Mug/Table and have 0% full-Franka collisions on the same checked poses.

The object-any-at-20 static-proxy scores are MGM 100%, RMF-MT 50%, RFM-50 100%
and RFM-100 100%. With only two known objects this is particularly coarse, and
must not be presented as a physical-success or large-benchmark comparison.

MGM improves distribution MMD and the thin-gripper proxy over the 1-NFE
RMF-MT adaptation, but multi-step RFM is more accurate. RMF-MT has fewer full
gripper collisions, partly illustrating why low collision alone is insufficient
when poses miss the object. **The substantial collision rates are unresolved;
this pilot does not demonstrate a deployable grasp generator.**

## Protocol and Fidelity

- Actual public ACRONYM HDF5 poses and matched meshes, no synthetic targets.
- Positive historical FleX labels select source poses. Per-object source-index
  partitions are disjoint 80/10/10; object geometry is known in all splits.
- SO(3) x R3 product geometry represents SE(3). Rotation is constrained by the
  rotation map; translations use mesh-center/radius normalization.
- Three seeds, fixed final checkpoints at 10000 updates, no test-based tuning
  or checkpoint selection. MGM additionally uses 100 warmup and 50000 critic
  updates per seed. Optimizer work is not matched across methods.
- RMF-MT executes pinned source loss/PCGrad algebra with declared architecture,
  path/time and representation adaptations, not the original SDF experiment.
- Distribution MMD uses fixed chordal-product kernel bandwidths. It is not
  numerically comparable to the separate sphere/torus primary MMD scores.
- Static feasibility checks mesh intersections and between-finger overlap; no
  closing, friction, lifting, dynamics, scene collision or reachability is tested.
  Physical-success fields remain null.

## Visualizations and Evidence

The publication comparisons use the first 12 unfiltered poses, shared cameras
and bounds including every displayed gripper. They are not all-512-tail views;
the full-range mesh figures remain available separately. No successful-pose
selection or substitution was used. PNGs are 5280px, with individual zooms,
vector PDFs and provenance JSON for all three seeds.

- [Mug, seed 0](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/publication_figures/seed0/mug_comparison.png>)
- [Table, seed 0](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/publication_figures/seed0/table_comparison.png>)
- [All publication figures](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/publication_figures/final_manifest.json>)
- [Machine-readable final results](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/final_summary.json>)
- [Per-object results](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/test_per_object_summary.json>)
- [Audit](<E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/comparison2026/grasp/final_audit.json>)
