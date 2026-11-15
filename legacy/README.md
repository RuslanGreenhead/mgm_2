# Legacy code index

No code has been classified as disposable in this migration. The older
general-geometry R-MGM modules remain at
`external/riemannian-fm/manifm/rmgm/`, where historical imports and
third-party attribution are intact. The older controlled suite remains at
`legacy/paper/experiments/`. The original SE(3) pilot remains at
`legacy/paper/benchmarks2026/grasp/`; its maintained copy is `src/rmgm/grasp/`.

The old manuscript, generated figures, paper code, run records, PDFs, article
history, environment observations, and archive manifests are all here. These
recorded protocols should be rerun from the original tagged layout; some old
path-dependent entry points are intentionally not new-project commands.

New work should use `src/rmgm/`. Move a legacy implementation here only after
documenting its callers, result provenance, license, and migration test. Do
not use this directory as a substitute for deciding which implementation is
canonical.
