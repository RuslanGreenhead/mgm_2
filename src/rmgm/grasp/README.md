# SE(3) pilot code

This is the promoted conditional grasp pilot from
`legacy/paper/benchmarks2026/grasp/`. It uses SO(3) × R³ geometry and its own
data, model, training, and evaluation contract. Its historical results remain
in `legacy/outputs/comparison2026/grasp/`; new runs default to
`outputs/grasp/`. Optional HDF5, mesh, and collision dependencies are needed
for full integration tests. The prior detailed protocol is documented in
`legacy/paper/benchmarks2026/grasp/README.md`.
