# Active code map

This archive contains several generations of code and many frozen output
snapshots. Treat the following as the working map for new development.

| Area | Primary files | Status |
| --- | --- | --- |
| Method specification | `paper/sections/method.tex` | Mathematical reference |
| Compact current benchmark R-MGM | `paper/benchmarks2026/mgm.py`, `paper/benchmarks2026/common.py` | Preferred starting point for new development |
| Benchmark orchestration | `paper/benchmarks2026/run_jobs.py`, `paper/benchmarks2026/worker.py` | CUDA-oriented paper sweep; do not run unchanged on a laptop |
| Table 2 controlled suite | `paper/experiments/controlled_suite.py` | Historical reproduction target; currently CUDA-bound |
| Original broad R-MGM implementation | `external/riemannian-fm/manifm/rmgm/` | Reference implementation; preserve until behaviour is covered by tests |
| Frozen paper results | `outputs/` and `paper/**/outputs/` | Evidence/provenance, not live implementation |

For the first local validation, use a synthetic sphere smoke test. The Table 2
Fire experiment additionally requires its absent raw dataset and a portable
runner; it is not yet reproducible from this archive alone.
