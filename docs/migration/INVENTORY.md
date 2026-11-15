# Migration inventory

The baseline is commit `73f2b89`, tagged `archive/pre-restructure-2026-11-15`.
Classification covers the major trees. No source or historical result was
deleted during this migration.

| Tree | Classification | Decision |
| --- | --- | --- |
| `legacy/paper/benchmarks2026/` | 2026 protocol and reference implementation | Kept for original source snapshots; methods promoted into `src/rmgm/`. |
| `legacy/paper/experiments/` | Earlier controlled protocol | Kept as historical source; use the baseline tag for exact reruns. |
| `legacy/paper/benchmarks2026/grasp/` | SE(3) pilot | Promoted to `src/rmgm/grasp/`; old protocol retained. |
| `external/riemannian-fm/` | Modified upstream RFM plus historical R-MGM | Retain, with provenance and license. |
| Other `external/` trees | Pinned author material | Retain unmodified. |
| `experiments/` | Newer local runnable protocols | Retain; Fire RMF-MT extension now imports maintained package. |
| `configs/` | Local experiment settings | Retain and extend per new protocol. |
| `legacy/outputs/`, `legacy/output/`, `legacy/article_history/` | Archival evidence and manuscript releases | Relocated intact; internal recorded paths are historical. |
| `outputs/` | New local generated runs | Ignore contents and use a unique run directory. |
| `docs/`, `legacy/environment/`, `tools/` | Guidance, historical environment, utilities | Keep current guidance at root. |
| `.venv/`, caches, generated binary assets | Local environment or generated files | Keep ignored; no deletion required. |
