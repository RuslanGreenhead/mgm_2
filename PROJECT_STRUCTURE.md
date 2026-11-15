# Project structure

This is the working R-MGM project. The earlier paper and its evidence are
under `legacy/`; the baseline before either migration is tagged
`archive/pre-restructure-2026-11-15`.

| Folder | What belongs here |
| --- | --- |
| `src/rmgm/geometry.py` | Maintained sphere/torus geometry and shared neural primitives. |
| `src/rmgm/methods/rmgm/` | Our R-MGM method. The default generator is MLP; `generator="gmm"` enables a local Gaussian mixture under the same objective. |
| `src/rmgm/methods/baselines/` | RW-Flow-C, KGDM, and RMF-MT implementations, plus an RFM upstream-source locator. |
| `src/rmgm/grasp/` | The promoted SE(3) pilot: conditional models, geometry, data, evaluation, training, and commands. Its optional data/mesh dependencies are separate from the sphere/torus package. |
| `src/rmgm/data/`, `training/`, `evaluation/` | Dataset manifest checks, method selection, and shared sphere/torus metrics. Extend these for new protocols. |
| `experiments/` | Runnable study-specific entry points, including smoke, reproduction, and comparison work. |
| `configs/` | Settings for new runs; record dataset and source identity in each protocol. |
| `data/` | Dataset guidance and future manifests. Raw and processed assets are ignored by Git. |
| `tests/` | Maintained-package regression checks. |
| `external/` | Pinned upstream implementations and historical RFM extensions; retain licenses and provenance. |
| `outputs/` | **The one destination for new generated run files:** configs, metrics, logs, checkpoints, figures, and reports. Its contents are ignored by Git. Use `outputs/<experiment>/<run-id>/` and avoid overwriting runs. |
| `legacy/` | Old paper source, historical experiments and outputs, prior PDF releases, article history, environment records, and archive manifests. |
| `docs/`, `tools/` | Current guidance and supporting tools; some old archive utilities remain for historical verification. |

The old `paper/benchmarks2026/` and `paper/experiments/` code is now at
`legacy/paper/`. The 2026 benchmark modules are retained there to interpret
their original source snapshots. The original historical GMM has train-fitted
variants and broader geometry coverage; the new optional GMM is a local
sphere/torus generator and is not an exact replacement for those runs.

Old run records are at `legacy/outputs/`. Paths and hashes inside them remain
historical evidence. New runs must use `outputs/`. The original archive's
`MANIFEST.json`, `EXCLUSIONS.json`, and environment references now live under
`legacy/`. They describe the old package, not this working tree.

For new work, import from `rmgm`, write thin experiment runners, and record
the code commit, dataset hash, config, seed, and selected checkpoint inside
each output directory. The prior 2026 queue is a legacy protocol and is not
the entry point for new experiments.
