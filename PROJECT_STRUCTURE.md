# Project structure

Start new method development in `src/rmgm/`. The root [README.md](README.md)
describes the packaged research archive and its 2026 results; this document
describes the maintained development layout. The pre-reorganization tree is
recoverable at Git tag `archive/pre-restructure-2026-11-15`.

| Location | Purpose and rule |
| --- | --- |
| `src/rmgm/geometry.py` | Shared sphere and torus operations, base sampling, and common neural building blocks. |
| `src/rmgm/methods/rmgm/` | Our maintained R-MGM model, critic, objective, and generators. `generator="mlp"` is the default; `generator="gmm"` selects a local chart Gaussian mixture. The GMM is an R-MGM generator choice, not a separate baseline. |
| `src/rmgm/methods/baselines/` | Our maintained RW-Flow-C, KGDM, and RMF-MT benchmark implementations, plus the RFM source locator. `rmf_mt.py` corresponds to the older benchmark module called `meanflow.py`. |
| `experiments/` and `configs/` | Runnable protocols and their settings. Keep experiment-specific budgets, datasets, and reporting here; reusable method logic belongs in `src/rmgm/`. |
| `tests/` | Tests of the maintained package, including parity with the old default R-MGM and GMM geometry/training checks. |
| `external/` | Pinned upstream code and the historically extended RFM tree. Preserve author licenses and provenance. The RFM tree also holds the historical broader-geometry R-MGM implementation. |
| `paper/` | Manuscript, figures, and the frozen 2026 experiment implementations. New method changes go in `src/rmgm/`. |
| `outputs/` | Historical metrics, manifests, and source snapshots. Treat these as evidence; do not write new experiments into this tree. |
| `artifacts/` | Local experiment runs. Its contents are Git-ignored. |
| `data/` | Dataset documentation and tracked manifests. Raw data and processed arrays belong in ignored storage; see [data/README.md](data/README.md). |
| `legacy/` | Index of older runnable code and the rules for retiring it. See [legacy/README.md](legacy/README.md). |
| `archive/` | Index of immutable historical evidence. Existing `outputs/` and old release paths have not been moved because recorded paths and checksums matter. |
| `docs/` and `environment/` | Research protocol, provenance, setup, migration mapping, and dependency references. |

## What is canonical now?

The current R-MGM sphere/torus implementation lives in
`src/rmgm/methods/rmgm/model.py`. Its default MLP configuration was checked
against `paper/benchmarks2026/mgm.py`. The optional local GMM generator is
available through the same R-MGM factory. Historical GMM with train-fitted
centers, and the SPD/hyperbolic/mesh branches, remain in
`external/riemannian-fm/manifm/rmgm/` until their behavior is covered by
equivalence tests. A current random-start GMM run must not be called an exact
reproduction of a historical train-fitted GMM run.

The nine-dataset 2026 benchmark uses R-MGM, RW-Flow-C, KGDM, and RMF-MT.
RFM belongs to the historical controlled suite, Fire reproduction, and SE(3)
pilot. Direct-MMD MLP and empirical resampling are controls, not named model
baselines. The SE(3) pilot has its own implementations and protocol under
`paper/benchmarks2026/grasp/`; those are not yet part of the canonical
sphere/torus package.

## Running and extending

Install the local package with `python -m pip install -e . --no-deps` in an
environment with the project's runtime dependencies, or set `PYTHONPATH=src`.
Run maintained-package tests with
`PYTHONPATH=src:. python -m unittest discover -s tests -v`.

The old `paper/benchmarks2026/` queue and worker remain at their original
paths so historical source snapshots, plans, and records can be interpreted.
They do not automatically execute the new package. Before launching a new
research comparison, create a new named protocol in `experiments/`, record
the exact package commit and data hashes, and write to `artifacts/`.

`MANIFEST.json`, `EXCLUSIONS.json`, and `tools/verify_bundle.py` describe
the earlier packaged archive. They are not a manifest of the current working
tree. To verify that archive, use the preserved baseline tag in its original
packaging context; do not rewrite historical hashes to make new code appear
part of past experiments.
