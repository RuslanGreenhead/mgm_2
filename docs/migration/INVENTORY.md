# Migration inventory

The baseline is commit `73f2b89`, tagged `archive/pre-restructure-2026-11-15`.
Classification covers the major trees. No source or historical result was
deleted during this migration.

| Tree | Classification | Decision |
| --- | --- | --- |
| `paper/benchmarks2026/` | 2026 protocol and reference implementation | Retain at original path for frozen source snapshots; promote method code into `src/rmgm/`. |
| `paper/experiments/` | Earlier controlled protocol | Retain as historical runnable suite. |
| `paper/benchmarks2026/grasp/` | SE(3) pilot | Retain with its separate geometry/data contract. |
| `external/riemannian-fm/` | Modified upstream RFM plus historical R-MGM | Retain, with provenance and license. |
| Other `external/` trees | Pinned author material | Retain unmodified. |
| `experiments/` | Newer local runnable protocols | Retain; Fire RMF-MT extension now imports maintained package. |
| `configs/` | Local experiment settings | Retain and extend per new protocol. |
| `outputs/`, `output/`, `article_history/` | Archival evidence and manuscript releases | Retain at exact paths. |
| `artifacts/` | Local generated runs | Keep ignored; never use as canonical input. |
| `docs/`, `environment/`, `tools/` | Documentation, environment, archive utilities | Retain. |
| `.venv/`, caches, generated binary assets | Local environment or generated files | Keep ignored; no deletion required. |
