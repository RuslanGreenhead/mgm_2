# MGM project handover

## Purpose

This is a research archive for **Riemannian Midpoint Generative Models (R-MGM)**: a direct, one-neural-call generator for data on constrained manifolds, trained with a hidden-flip midpoint/tangent-critic objective. It also contains paper sources, historical R-MGM experiments, and comparisons with RFM, RW-Flow-C, KGDM, and RMF-MT. It is not a ready-to-run release: weights, full datasets, and installed environments are intentionally absent.

## Main components

- `paper/`: current manuscript, figures, experiment protocols, and current benchmark code.
- `paper/benchmarks2026/`: current MGM-vs-recent-method benchmark, shared geometry, worker, plans, tests, and reporting.
- `paper/experiments/`: earlier controlled MGM/RFM real-data suite and diagnostics; it reuses historical R-MGM code.
- `external/riemannian-fm/`: historical RFM-derived repository with the general-geometry R-MGM layer.
- `external/kgdm-author/`, `external/riemannian-meanflow-author/`: preserved baseline source material.
- `outputs/`: historical metrics, plans, logs, and frozen source snapshots; not primary editable implementations.
- `docs/`: experiment map, source index, asset/data guidance, and notices.

## Proposed model implementation

### Current general benchmark implementation

- `paper/benchmarks2026/mgm.py`
  - `Critic`: time-conditioned tangent critic; uses periodic spatial features on a torus.
  - `MGM`: owns the generator/critic optimizers, hidden-flip observation creation, alternating updates, sampling, and checkpoint state.
  - `create_method(ops, cfg, device)`: factory used by the worker.
- `paper/benchmarks2026/common.py`
  - `ManifoldOps`: sphere/torus projection, tangent projection, metric inner product, exponential/log maps, paths and velocities.
  - `PointGenerator`: direct generator; maps latent noise to a manifold base point, predicts a residual with an MLP, then applies `exp`.
  - `MLP`, `TimeFeatures`, RNG helpers.
- Training entry point: `paper/benchmarks2026/run_jobs.py` launches the isolated training worker in `paper/benchmarks2026/worker.py`. The worker dynamically imports `paper.benchmarks2026.mgm`, calls `create_method`, and repeatedly calls `MGM.update`.

### Related implementations, not the same current benchmark module

- Historical/general-geometry R-MGM: `external/riemannian-fm/manifm/rmgm/generators.py`, `geometry.py`, `critics.py`, and `losses.py`; historical trainers include `external/riemannian-fm/rmgm_train.py` and `rmgm_train_real.py`.
- Earlier controlled suite: `paper/experiments/controlled_suite.py`; imports the historical `manifm.rmgm` implementation.
- SE(3) pilot: `paper/benchmarks2026/grasp/models.py` (`Generator`, `Field`, `mgm_objective`) and `grasp/runner.py`; a local conditional SO(3) x R3 adaptation.
- Baselines: `paper/benchmarks2026/rwflow.py`, `kgdm.py`, `meanflow.py`; historical RFM baseline is `external/riemannian-fm/rfm_baseline_train.py`.

## Current MGM: interface and computation

- Inputs: latent Gaussian vector `z`; a real-data batch `y`; RNG; manifold operations/configuration. The critic receives folded time `t` and a manifold observation `x`.
- Output: `PointGenerator(z)`, a valid sphere or torus point in the current benchmark. At inference, only this generator is used (one neural forward call).
- Per outer generator iteration:
  1. Generate `x0 = G(z)` and pair it with real endpoint `y`.
  2. Sample `t` uniformly from `[0, 1/2]` and hidden bit `b`; set path time `s = t + b(1 - 2t)`.
  3. Form a geometry-aware path observation `x = gamma(s)` and its actual path-time tangent velocity `a = d gamma(s)/ds`.
  4. Update the critic to regress `a` from `(t, x)`, with tangent/metric geometry.
  5. Freeze the critic and update the generator using `2 <f,a>_g - <f,f>_g`, differentiating through the observation, velocity, and geometry.
  6. At inference, sample `z` and return `G(z)`; no critic, ODE, or flow integration is used.

The implementation is MLP-based, not the DDPM++/U-Net-like generator and critic architecture used in the original MGM context described by the user. Current benchmark defaults are small MLPs (generator width 128/depth 3/latent 16; often critic width 256); see `outputs/comparison2026/selected_config_v1.json`. Historical R-MGM is also MLP plus manifold-specific output heads.

## Established facts and uncertainties

Established:

- The manuscript identifies R-MGM as a hidden-flip, tangent-velocity, variational critic construction on reversible manifold paths (`paper/sections/method.tex`).
- The current benchmark’s implementation is `paper/benchmarks2026/mgm.py` plus `common.py`; its queue/worker is the current training route.
- Sphere and torus are the geometries implemented by the current `ManifoldOps`; historical code covers further geometries.
- The archive excludes learned weights and full raw datasets, so archived outputs do not prove current reproducibility.
- The repository contains a partial modern comparison, not a completed universal benchmark; the paper cautions against superiority claims.

Uncertain / needs source or author confirmation:

- There is no implementation of the original DDPM++ MGM architecture in this archive, so architectural parity with original MGM cannot be assessed here.
- The current “proposed model” has multiple experiment-specific implementations (current benchmark, historical code, SE(3) pilot); diagrams should state which branch they represent.
- Historical frozen snapshots under `outputs/` may record executed versions, but are archival copies rather than canonical edit targets.
- Exact data acquisition/preprocessing cannot be reproduced from this archive alone because raw arrays/assets are omitted.

## Files to attach for an accurate architecture diagram

1. `paper/benchmarks2026/mgm.py`
2. `paper/benchmarks2026/common.py`
3. `paper/sections/method.tex`
4. `paper/benchmarks2026/worker.py`
5. `outputs/comparison2026/selected_config_v1.json`

For the historical/general-geometry diagram also attach:

- `external/riemannian-fm/manifm/rmgm/generators.py`
- `external/riemannian-fm/manifm/rmgm/critics.py`
- `external/riemannian-fm/manifm/rmgm/geometry.py`
- `external/riemannian-fm/manifm/rmgm/losses.py`

## Data and experiment setups

- Current benchmark: nine real tasks—sphere data (Volcano, Earthquake, Flood, Fire), torus data (General, Glycine, Proline, Pre-Pro), and seven-angle RNA—documented in `paper/benchmarks2026/README.md` and `docs/EXPERIMENTS.md`. Final planned runs use three seeds, 50k generator/field updates, validation checkpoint selection, and one-call sampling; completion is partial.
- Earlier controlled suite: same broad real sphere/torus/RNA family with fixed small-network MGM/RFM controls; primary archived authority is described as `outputs/paper_icml2026/controlled_independent/`, but not all required data/checkpoints are present. See `paper/experiments/README.md`.
- Historical branch: toy and pilot sphere, torus, SPD, hyperbolic, mesh, and selected real-data experiments; see `external/riemannian-fm/README_RMGM.md` and `docs/EXPERIMENTS.md`.
- SE(3) pilot: conditional pose generation for two ACRONYM objects (Mug/Table), using SO(3) x R3 geometry; code/protocol are under `paper/benchmarks2026/grasp/`.

## Structure assessment

Positive:

- Clear separation of current benchmark, earlier controlled study, historical code, baselines, paper, and archived outputs.
- Strong provenance practices in the current benchmark: source snapshots/hashes, isolated workers, separate RNG streams, resumable checkpoints, and validation-before-test protocol.
- The model is compact and readable: geometry, model, objective, and worker orchestration are separated.
- Documentation candidly identifies incomplete experiments, limitations, and non-equivalent baselines.

Negative:

- The same method exists in several branches with different architectures/geometries; it is easy to confuse the current benchmark with historical R-MGM or the SE(3) pilot.
- The project is research-archive oriented rather than turnkey: dependencies, raw data, model weights, and some artifacts are missing.
- `outputs/` contains many frozen snapshots and historical records, increasing navigation noise and the risk of treating archival copies as live source.
- The benchmark is incomplete, and its small MLP architecture is not an implementation of the original MGM DDPM++/U-Net setup.
