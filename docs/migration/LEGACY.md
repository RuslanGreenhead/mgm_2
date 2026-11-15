# Legacy and provenance rules

`legacy/` contains the earlier paper, its experiments, historical outputs,
PDFs, article history, environment observations, and package manifests. The
older general-geometry R-MGM code remains in the modified external RFM tree
because its import and licensing context is distinct. Its broader geometry
paths still need equivalence tests before promotion.

New code belongs in `src/rmgm/`, `experiments/`, and `tests/`; new run files go
to `outputs/`. The promoted SE(3) pilot is in `src/rmgm/grasp/` while the old
protocol remains in `legacy/paper/benchmarks2026/grasp/`. Do not use old
checkpoints with a new implementation without state and output equivalence
tests. Recorded paths inside historical manifests have not been rewritten.
