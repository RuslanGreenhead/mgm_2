# KGDM Adapter

## Provenance

- Paper: https://arxiv.org/html/2605.10727 (v1, 2026-05-11), equations 10-12.
- The paper's code link is https://anonymous.4open.science/r/kernel-grad-drift-B4D5 .
  On 2026-10-24 its public options endpoint,
  https://anonymous.4open.science/api/repo/kernel-grad-drift-B4D5/options ,
  redirects to https://github.com/jcarrascopollo/kernel-grad-drift . That GitHub
  repository/API returned 404. The endpoint reports source date 2026-05-18.
- Coauthor Jorge Carrasco Pollo's public release:
  https://github.com/jcarrascopollo/kgd_thesis . Its README identifies this as
  the KGDM submission code. This is verified author code, not a third-party
  reimplementation, but is NOT claimed to be the original anonymous snapshot.
- Clone: `external/kgdm-author/`, unmodified, MIT license retained.
- Pinned commit: `5860d212a1e53b53a499e251f2d087fcf0a5fc00`.
- Executed source: `earth/drift_core.py`, loaded directly with importlib.
  LF-normalized SHA256:
  `39008dd140a304851dc200216d035c714fe3264c9319ee28bb534caefbf0a5d0`.
  The adapter verifies this hash before loading. CRLF checkouts are supported.
- Permanent source:
  https://github.com/jcarrascopollo/kgd_thesis/blob/5860d212a1e53b53a499e251f2d087fcf0a5fc00/earth/drift_core.py
- Training wrapper reference:
  https://github.com/jcarrascopollo/kgd_thesis/blob/5860d212a1e53b53a499e251f2d087fcf0a5fc00/earth/models/earth_drift.py

The released README's results differ from arXiv v1. Do not describe runs with
this adapter as reproducing the original reported numbers.

## Fidelity

**Sphere: `author_field_shared_backbone`.** Executes the released
`compute_V_sphere_multi(..., grad_form=True)` unchanged. Preserves both native
geodesic Laplace and Gaussian kernels. Uses the mandated common PointGenerator
and common log/exp geometry for target regression, rather than the author's
EarthModel and clamped-acos regression geometry. Teacher-free, single forward
sampling, no generation teacher, no scalar MMD SGD.

**Torus: `torus_adaptation_of_author_field`.** No torus implementation was found
in this author release. Extends the same released finite-batch algebra using
native wrapped angular logarithms and flat-product metric. Generator features
are common.py's periodic cos/sin features. This is a benchmark adaptation, not
an author-reported torus result. Neither extension asserts characteristic-kernel
or convergence guarantees for the chosen geodesic radial kernels.

Spectral Matérn is a distinct released option and is NOT silently approximated
here. This adapter exposes only the requested native radial Gaussian/Laplace
variants. `kernel='gaussian'` is an alias for the native spelling `'gauss'`.

## Released Finite-Batch Algorithm

Let `x` be generated points, `y` real points, and negatives be the same `x`.
For each temperature T:

1. Compute raw geodesic distances to positives and negatives. If `dist_norm`,
   divide both distance matrices by their single concatenated batch mean,
   clamped below at 1e-6. The mean includes the self diagonal BEFORE masking.
2. Laplace logits are `-d_scaled/T`; Gaussian logits are `-d_scaled^2/T`.
   Add 1e6 to the negative diagonal distance/exponent before dividing by T,
   precisely as the source does. This removes same-index generated self pairs,
   not every coincident pair. Batches must contain at least two generated points.
3. Concatenate positive and negative logits. With row and column softmax,
   `A = sqrt(softmax_row(logits) * softmax_col(logits))`.
4. `W_pos = A_pos * sum_row(A_neg)` and
   `W_neg = A_neg * sum_row(A_pos)`. Their row masses balance.
5. Directions are `Log_x(y)/d_raw` for Laplace and `Log_x(y)` for Gaussian.
   Attractive contribution minus generated-negative contribution gives the
   field. Project to the tangent space on the sphere.
6. With native `drop_T=True`, omit the derivative's per-T constant. If False,
   multiply the field by 1/T (Laplace) or 2/T (Gaussian). Sum temperatures,
   NOT their average. Scale the summed field by `step_size`, then cap its
   per-sample metric norm by `max_step`.
7. Target is `stop_gradient(Exp_x(step))`. Minimize
   `mean(inner_x(Log_x(target), Log_x(target)))` with the generator's optimizer.

All fields, distance scaling, affinities, normalizers, and targets are computed
under `torch.no_grad()`. Only regression through the current generator output
is differentiated.

**Important distinction:** the released radial finite-batch double-softmax field
is not the plain row-normalized population KDE score-ratio estimator. Also,
with `dist_norm=True`, the released direction retains raw distances and omits
the batch-scale chain factor even when `drop_T=False`. We preserve this rather
than silently changing the official field. Compared with differentiating the
normalized-distance kernel with frozen batch scale `s`, the source drops a
factor 1/s for Laplace or 1/s^2 for Gaussian, in addition to optional T factors.
The autodiff reference tests explicitly compensate these documented constants.

## Defaults and Adaptations

Author release defaults are in:
https://github.com/jcarrascopollo/kgd_thesis/blob/5860d212a1e53b53a499e251f2d087fcf0a5fc00/earth/configs/config.yaml

| Setting | Author Earth default | Adapter default |
| --- | --- | --- |
| Generator | noise 3, width 128, input + 4 hidden blocks + output | common uniform-prior residual generator, latent 16, width 128, depth 3 |
| Batch | 128 | 256, actual batch passed by parent |
| Learning rate | 5e-4 YAML (TrainConfig fallback differs) | 2e-4 |
| Optimizer | AdamW, betas (0.9, 0.95), decay .01 | preserved |
| Warmup | 2500 steps, first update LR zero | 0 for short controlled budgets; configurable |
| EMA | .999, evaluate EMA | preserved; `sample_ema=True` |
| Gradient clip | 1 | preserved |
| Kernel | Laplace | preserved; Gaussian selectable |
| Temperatures | (.02, .05, .2) | preserved |
| Distance normalization | True | preserved |
| Drop T constant | True | preserved |
| Maximum step | 1 radian | preserved |
| Global step multiplier | implicit 1 | `step_size=1` |
| Bandwidth multiplier | implicit 1 | `bandwidth_scale=1` |
| Training length | up to 20 epochs | parent owns budget |

`max_step=None` disables capping, as in the author wrapper. It is not advised
without a separate injectivity-radius safeguard. The default 1-radian cap is
below the sphere and angular-torus injectivity radii. Do not enable 2500-step
warmup for a 2000-step run and interpret it as a converged model.

The author's `earth/configs/winners.yaml` explicitly says its per-dataset
choices were selected using seed-mean **test MMD**. These are NOT imported,
and no data-specific winner flags or spectral ladders are used here. Selection
must use the parent's saved validation split and fixed probes only.

## Contract and Continuation

`create_method(ops, cfg, device)` returns `KGDM` with `.model`, `.update`,
`.sample`, `.checkpoint`, and `.load_checkpoint`. Sample always uses one
generator evaluation and rejects `nfe != 1`. `.model` is the trainable common
generator; `.ema_model` is the sampling copy unless `sample_ema=False`.

Each update returns detached device scalar tensors, with no `.item()` or Python
float conversion of device results. GPU finite checks are asynchronous; a failed
device assertion requires worker restart. No CUDA branch was exercised here.
There is no `generator_updated` flag because every call updates the generator
(apart from the optional author's zero-LR first warmup step).

Checkpoints are frozen CPU snapshots containing raw model, optimizer including
Adam moments, EMA, step counter, resolved config, fidelity, and source metadata.
`load_checkpoint` restores all these training states and rejects mismatched
config/fidelity. The warmup LR is reconstructed exactly from the step counter.
The parent owns and separately saves/restores passed generator RNG state and
global RNGs. Neither update nor sample reseeds or draws from global RNG state.
An EMA copy is not an extra sampling NFE or teacher.

## Suggested Validation Grid and Budget

Suggestion only; not executed or tuned:

- Keep the shared width/depth/latent/batch and LR fixed for the primary comparison.
- `kernel`: `['laplace', 'gauss']`.
- `bandwidth_scale`: `[0.5, 1.0, 2.0]`, multiplying the complete native ladder.
- `step_size`: `[0.25, 0.5, 1.0]`; keep `max_step=1`, `dist_norm=True`,
  `drop_T=True`, EMA .999 and no warmup. Eighteen configurations in total.
- Screen at the same 2000 generator updates as other shared-backbone methods,
  with fixed validation probes every 500 or 1000 updates. Freeze the grid,
  selection rule, and budget before the first queued job.
- Pick only by validation. Then run the selected recipe with the parent's
  predeclared seeds, preferably three, and evaluate test only after selection.
- An optional separately declared longer-budget comparison can give EVERY method
  10000 generator updates. Report that separately rather than extending KGDM
  alone. These budgets are controlled adaptations, not the author epoch recipe.

No training recommendation here authorizes a GPU job. Parent owns the queue.

## CPU Verification

Command: `.conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_kgdm -v`

Result on 2026-10-24, supplied torch 2.5.1 environment: **20 tests passed**.

Tests cover analytic pair gradients versus intrinsic autodiff and centered
manifold finite differences; complete fields versus frozen-normalization
autodiff across both geometries/kernels and both normalization/T flags; exact
released spherical-field equality; temperature summation; self mask and balanced
cross masses; attraction sign; permutation equivariance; coincident samples;
tangency, cap and manifold constraints; detached targets; exact shared generator;
finite CPU updates; single-call EMA sampling; isolated RNGs; frozen checkpoint
snapshots; exact optimizer/EMA continuation; optional author warmup; input checks.

No data files or train/validation/test metrics were accessed by these tests.
All updates were small synthetic CPU unit tests. No CUDA work or training run
was started. GPU numerical behavior remains unverified until parent releases it.
