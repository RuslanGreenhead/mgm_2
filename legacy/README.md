# Legacy code index

No code has been classified as disposable in this migration. The older
general-geometry R-MGM modules remain at
`external/riemannian-fm/manifm/rmgm/`, where historical imports and
third-party attribution are intact. The older controlled suite remains at
`paper/experiments/`. SE(3) pilot methods remain at
`paper/benchmarks2026/grasp/` with their distinct protocol.

New work should use `src/rmgm/`. Move a legacy implementation here only after
documenting its callers, result provenance, license, and migration test. Do
not use this directory as a substitute for deciding which implementation is
canonical.
