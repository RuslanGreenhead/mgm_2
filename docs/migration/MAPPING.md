# Source mapping

| Earlier source | Maintained location | Reason |
| --- | --- | --- |
| `paper/benchmarks2026/common.py` | `src/rmgm/geometry.py` | Shared geometry and generator primitives. |
| `paper/benchmarks2026/mgm.py` | `src/rmgm/methods/rmgm/model.py` | Primary R-MGM objective and training state. |
| `external/riemannian-fm/manifm/rmgm/generators.py` | `src/rmgm/methods/rmgm/generators.py` (new local GMM variant only) | Keep an optional GMM in the maintained method; historical version stays for exact reference. |
| `paper/benchmarks2026/rwflow.py` | `src/rmgm/methods/baselines/rwflow.py` | Maintained RW-Flow-C reimplementation. |
| `paper/benchmarks2026/kgdm.py` | `src/rmgm/methods/baselines/kgdm.py` | Maintained adapter; pinned field stays in `external/kgdm-author/`. |
| `paper/benchmarks2026/meanflow.py` | `src/rmgm/methods/baselines/rmf_mt.py` | Maintained RMF-MT implementation with clearer name. |
| `external/riemannian-fm*/` | `src/rmgm/methods/baselines/rfm.py` (source locator only) | RFM retains its own historical protocol and upstream source. |
| `paper/benchmarks2026/grasp/` | unchanged | SE(3) pilot has different conditional models and data contract. |
| `paper/experiments/mmd_control.py` | unchanged | Direct-MMD is an experiment control. |
| `outputs/` | unchanged | Paths and hashes are part of historical provenance. |

The earlier paths remain in the working tree. The maintained package is used
by new development; existing 2026 benchmark snapshots continue to describe
the exact earlier files. There is no claim that the new GMM is identical to
train-fitted historical mixtures.
