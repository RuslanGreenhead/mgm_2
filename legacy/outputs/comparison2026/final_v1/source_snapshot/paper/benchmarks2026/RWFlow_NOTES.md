# RW-Flow-C Adapter

## Provenance And Fidelity

Classification: **equation-faithful paper reimplementation with a shared-generator adaptation**, not author code and not a generation-teacher method. Implements the chordal-cost RW-Flow-C variant only.

Primary source: [RW-Flow, arXiv v2](https://arxiv.org/html/2609.39271v2), dated October 1, 2026. Mathematical references: equations (5), (7), (72), Algorithms 1-2, and the chordal row of Table 1. Algorithm 1 uses an independent generated self bank. Table 1 defines the chordal cost without a factor of one half.

Public-code search checked October 24, 2026: the paper, [Seungwoo Yoo's page](https://dvelopery0115.github.io/), [Prin Phunyaphibarn's page](https://prinphunya.github.io/), [Minhyuk Sung's page](https://mhsung.github.io/), and the [lab publication list](https://visualai.kaist.ac.kr/publications/). Also searched GitHub-indexed results and the public repository listing at [GitHub API](https://api.github.com/users/dvelopery0115/repos?per_page=100). No verified RW-Flow author repository was found. This is a search result, not proof that none exists. Author source URL and commit are **null**; no `external/rw-flow-author/` directory was created.

## Implemented Algorithm

Let `x = G(z)` and `s = stopgrad(G(z_independent))`; the real minibatch is `y`. All field inputs and both Sinkhorn solves are detached. For each pair of banks:

```text
C_ij = cost_scale * ||embed(x_i) - embed(bank_j)||^2
OT_epsilon = min_pi <C,pi> + epsilon * KL(pi || a tensor b)
a_i = 1/N, b_j = 1/M
P_ij = pi_ij / a_i                         # conditional, NOT joint mass

attraction_i = -sum_j P_target_ij * grad_x C(x_i,y_j)
repulsion_i  = +sum_l P_self_il   * grad_x C(x_i,s_l)
V_i = attraction_i + repulsion_i
target_i = stopgrad(exp_x_i(flow_step * V_i))
loss = mean_i distance2(x_i, target_i)
```

Both terms have coefficient one. The half-factor on the symmetric self-OT energy is canceled by its two source-distribution arguments. The real-real energy term is constant with respect to the generator and does not enter its field. There is no differentiated OT objective, differentiable Sinkhorn unrolling, target-gradient path, velocity norm regression, EMA teacher, or trajectory teacher.

Sphere coordinates are XYZ on the unit sphere, with cost gradient `2 * cost_scale * tangent_x(x-y)`. Torus coordinates are angles with the product-circle embedding `[cos(x),sin(x)]`; its intrinsic gradient is `2 * cost_scale * sin(x-y)`. This cost is periodic, smooth, and dimension-summed, not raw squared angular differences and not mean distance per coordinate. The regression loss remains intrinsic squared geodesic distance, not embedding MSE.

Every update draws independent latent tensors sequentially from the supplied RNG. The self bank is fresh, not reused from `x` and not a stale cache. **No diagonal is masked.** An equal row/column index across independent banks has no identity meaning; accidental duplicate coordinates retain their mass. The mathematical tests additionally use a same-bank solve to check the symmetric energy identity, but training never does so.

The default Sinkhorn solver uses fixed-count, log-domain alternating updates, target potential first and source potential last. The last update enforces the row conditional. An explicit positive-scaling alternative is documented below. `epsilon` is in full squared-chordal cost units, not a Gaussian standard deviation or a GeomLoss `blur`. With cost multiplied by `k`, multiplying epsilon by `k` preserves couplings; dividing `flow_step` by `k` also preserves targets. There is no epsilon scheduling or data-derived bandwidth.

## Defaults

The paper's Table 8 reports widths 1024 (sphere) and 512 (torus), four hidden linear/SiLU stages, raw coordinates, AdamW with learning rate 3e-4 cosine-decaying to 3e-6, weight decay 0.01, betas (0.9,0.999), optimizer epsilon 1e-8, batch 1024, clipping 1, EMA 0.999, float32/TF32, three seeds, and equal wall time. Numeric entropy epsilon, flow step, Sinkhorn count, and sphere/torus wall-time duration were not located in the inspected text. They must not be invented as author defaults.

Adapter defaults:

| Config | Value | Status |
| --- | --- | --- |
| `width`, `depth`, `latent_dim` | 128, 3, 16 | Shared-suite architecture |
| `lr`, `batch_size` | 2e-4, 256 | Shared-suite adaptation |
| `cost`, `cost_scale` | `chordal`, 1.0 | Full squared embedding cost |
| `epsilon`, `flow_step` | 0.1, 0.1 | Explicit numerical adaptation |
| `sinkhorn_iters` | 256 | Explicit finite-solver adaptation |
| `sinkhorn_solver`, `sinkhorn_tol` | `log`, `None` | Fixed count; no early stopping |
| `source_batch_size` | `None` | Uses actual real minibatch size |
| `self_batch_size` | `None` | Uses generated source count |
| `weight_decay`, `grad_clip` | 0.01, 1.0 | AdamW and global-norm clipping |

The actual generator is parent's `common.PointGenerator`: a data-independent uniform-prior residual map, periodic torus features, zero-initialized last layer. It is the same G used by MGM/KGDM. No alternate architecture, learned initialization, or data fit is substituted. No EMA or LR scheduler is implemented; `.model` is the trained sampling model. Parent-controlled TF32 and selection policies remain authoritative. Consequently this adapter is **not an exact experimental reproduction** of the paper's large-network setup.

## Parent Integration

```python
from paper.benchmarks2026.rwflow import create_method

method = create_method(ops, cfg, device)
logs = method.update(real_batch, train_rng)  # one optimizer step
samples = method.sample(n, sample_rng, nfe=1)
state = method.checkpoint()
method.load_checkpoint(state)
```

`update` returns detached tensors on the model device, including loss, gradient norm, field component norms, step, and both solver marginal residuals. It performs no `.item()`/float conversion. Every successful update changes the generator; no `generator_updated` flag is necessary. Sampling consumes only its supplied RNG, performs one G evaluation, and restores the training/eval mode. CPU RNGs are supported even with another model device by explicit transfer; device-local RNGs avoid that transfer.

Checkpoints are deep snapshots of model, optimizer, resolved config, step, and provenance. Restoring requires the same resolved config. There is no unrecorded EMA state. Parent owns and must separately save all RNG states, wall-time accounting, probe selection, and data indices. This module reads no datasets, writes no output, launches no jobs, and touches no CUDA until explicitly constructed with a CUDA device by parent.

## CPU Verification

```powershell
.\.conda\rmgm\python.exe -B -m unittest paper.benchmarks2026.test_rwflow -v
```

All 22 tests pass. Both test classes set PyTorch's CPU thread count to one and restore it afterward. The suite covers sphere/T2/T7 cost-gradient directional finite differences, full debiased Sinkhorn-energy finite differences including the self half-factor, an independent-bank field derivative, a closed-form 2x2 entropy/diagonal check, rectangular marginals, analytic singleton signs, cancellation, periodicity, cost/epsilon/step scaling, and complete detach behavior. Integration uses the actual shared `ManifoldOps` and `PointGenerator` for updates, independent-bank draws, sampling constraints, global RNG isolation, immutable checkpoints, and bit-exact CPU continuation. Additional tests compare batched/log/scaling/scaling64 outputs with the original scalar log reference at every tested L, check underflow fallback and scalar-read counts, and verify TorchScript compatibility and failure-before-optimizer behavior.

No GPU tests, training run, hyperparameter sweep, or test-set inspection was performed. CPU-only timing and training-bank numerical diagnostics follow.

## Solver Optimization And Safety

Pinned source: [arXiv 2609.39271v2, Algorithm 2](https://arxiv.org/html/2609.39271v2#A5). Its loop takes a symbolic count L; **there is no published numeric maxiter or tolerance default in the inspected version**. The implementation pins `sinkhorn_iters=256`, `sinkhorn_tol=None`, and `sinkhorn_solver='log'` as adapter choices. Non-null tolerance is rejected, not silently ignored. A lower L in a parent job is an explicit adaptation, never an automatic optimization.

The original log-domain equations are retained in `_sinkhorn_log` and independently copied into the tests as `original_log_conditional`. When the real and independent self banks have equal sizes, two independent problems are stacked on a leading batch axis. Their masses never mix. This halves the number of sequential logsumexp calls (4L to 2L) without changing L. Different-sized banks use separate solves.

To opt into the alternate solver, set **`sinkhorn_solver='scaling'` explicitly**. It constructs a row-shifted kernel `K_ij=exp(-(C_ij-rowmin_i)/epsilon)` and uses positive matrix-vector scaling. Crucially, its first target potential is computed from the **unshifted** log kernel, preserving the reference's exact initialization even for L=1. Simply starting both positive scalings at one after a row shift would not be finite-L equivalent. Subsequent iterations use:

```text
v = (N/M) / (transpose(K) @ u)
v = v / max(v)                   # gauge only; no coupling change
u = 1 / (K @ v)
P = diag(u) @ K @ diag(v)
```

Before this loop a single scalar range check requires the maximum row-shifted exponent range R to satisfy `2*R + log(M) + 2 <= -log(finfo.tiny)`. Thus max-normalized v is at least exp(-R), u is between 1/M and exp(R), and even the smallest final mass is at least exp(-2R)/M, above normal underflow with a margin. At M=256 the conservative cutoff is about 39.9 for float32 and 350.4 for float64. This is stricter than mere kernel representability. Unsafe ranges (including many sharp epsilon=0.1 or 0.3 T7 cases) **fall back to the unchanged log solver**, with `target_scaling_used`/`self_scaling_used` logs set to zero. If a batched pair contains one unsafe problem, both use the reference. No Gibbs-kernel floor is introduced. Denominator clamps at `finfo.tiny` are defensive and inactive under the range bound; they are not a new regularizer.

CPU tests verify finite-L coupling agreement in float32 and float64, including nonuniform large row offsets and rectangular matrices, and field agreement in float64 on sphere/T2/T7 for both equal and unequal bank sizes. Positive-scaling tests include exponent ranges up to 250 in float64 and exact fallback agreement for unsafe float32 inputs. Equivalence means up to floating-point roundoff, not cross-solver bit equality. GPU numerical equivalence is still untested here and should be checked by parent before freezing a scaling-based run.

### Optional FP64 Scaling

Set **`sinkhorn_solver='scaling64'`** to promote the already-computed cost matrix to float64 before the guarded positive solve. Conditional weights are cast back to the input cost's working dtype **before** field contraction. With float32 models, the generator, cost construction, cost gradients, regression, optimizer, field, returned weights, and logs remain float32. Only the detached coupling solve uses float64. There is no epsilon change, target change, iteration reduction, or teacher. This is an explicit solver-precision adaptation, not an author default. The existing `log` and `scaling` paths and all default settings are unchanged.

The float64 cutoff is about R=350.4 at M=256. T7's full product-circle cost is at most 28, so epsilon=0.3 gives an exponent range no greater than about 93.3, safely inside that bound for valid inputs. Even epsilon=0.1 is inside the FP64 bound on T7. Unsafe FP64 ranges still fall back to a **float64 log-domain** solve at the same L, followed by the same output cast. Tiny individual weights may underflow during the final float32 cast; diagnostics are computed from those returned weights, not the uncast internal mass. No kernel floor is introduced.

Added tests compare finite counts L=1/2/32/64/256 against `original_log_conditional(cost.double(), epsilon, L).to(cost.dtype)`, check float32-log tolerances, prove acceptance beyond the float32 safety range and retention of FP64 fallback, check float32 sphere/T2/T7 field contractions for both bank layouts, and verify exact `scaling64` checkpoint continuation. Internal FP64 arithmetic is numerically equivalent up to roundoff; it is not expected to match the float32 log path bit-for-bit.

Suggested parent benchmark candidate: `epsilon=0.3, sinkhorn_iters=128, sinkhorn_solver='scaling64', sinkhorn_tol=None`, compared with `log` at the **same epsilon and L**. A separate L64 candidate is an explicit convergence-budget change and must pass its own residual checks. No claim about RTX4090 FP64 throughput is made without the parent's authorized GPU measurement. No CUDA work or other-process management was performed here; contaminated concurrent-GPU timings must not determine final throughput budgets, and the final queue remains subject to the parent's GPU-policy decision.

Five data-dependent synchronizations in the original update are consolidated into **one final pre-optimizer guard** (finite real batch, loss, gradient norm). Nonfinite gradients still prevent optimizer mutation. There are no convergence `.item()`/boolean reads inside either solver loop. The scaling range check adds one scalar synchronization per batched pair, so a standard equal-bank update uses one guard for log or two for scaling. AdamW's ordinary CPU step-counter reads are not CUDA synchronizations. The restore path explicitly places non-capturable/non-fused Adam counters on CPU because the parent worker's `map_location=device` can otherwise put them on CUDA and introduce extra host transfers after resume.

One-thread CPU timing, two warmups and median of five measured solves, two 256x256 float32 T7 costs, epsilon=1 (solver only, not model training):

| L | Separate Log | Batched Log | Batched Scaling | Scripted Batched Log |
| --- | --- | --- | --- | --- |
| 64 | 13.75 ms | 12.65 ms | 4.96 ms | 11.28 ms |
| 256 | 53.01 ms | 46.84 ms | 16.74 ms | 45.20 ms |

These CPU results do **not** establish GPU throughput. `_sinkhorn_log` is scriptable and tested, but scripting alone left the same dominant logsumexp work and offered little CPU benefit; no scripted runtime or `torch.compile` dependency is enabled. Scaling removes all but the initialization logsumexp and uses matrix-vector products. CUDA launch/runtime overhead and WDDM behavior must be measured by parent. A changed flow step does not alter the fixed-L operation count.

One fixed training-only initialization probe per fire/general/RNA, N=M=256, epsilon=1, seed 2026, float64, compared with L=1024 log-domain reference:

| Data | L32 Maximum Relative Column Error | L32 Relative Field RMS Error | L64 Maximum Relative Column Error |
| --- | --- | --- | --- |
| fire | 4.2e-15 | 4.6e-15 | 4.4e-16 |
| general | 4.1e-9 | 4.0e-9 | 5.6e-16 |
| rna | 2.4e-8 | 3.8e-9 | 2.2e-15 |

L64 relative field RMS errors were below 7e-16. This supports explicitly testing epsilon=1, L=64 for speed, not silently reducing the default or assuming later trained distributions converge as fast. Continue logging marginal residuals; parent can preregister a numerical adequacy target such as relative column residual <=1e-3 and field discrepancy <=1e-3 against a longer solve on fixed training/validation probes. These thresholds are proposed numerical checks, not paper defaults or in-loop early-stopping criteria.

## Opt-In CUDA Graph FP64 Solve

`sinkhorn_solver='graph64'` is an additional explicit option. The default
remains `log`; the old log/scaling/scaling64 implementations and their default
settings are unchanged. Graph64 captures **only the detached FP64 OT solve**.
The model, latent/target RNGs, field contraction, targets, loss, optimizer,
checkpoint state, and sampler remain outside capture. CPU graph64 executes the
same eager capture-safe algebra and never initializes CUDA.

The cache key is `(CUDA device, complete cost shape, epsilon, finite L)`. Each
entry owns a static FP64 cost buffer, captured output, graph, and private
stream. Capture uses three eager warmups on that side stream, then
`torch.cuda.CUDAGraph` with `torch.cuda.graph(..., stream=...)`. Every solve
copies the new cost into the static buffer and replays; row scaling and both
potentials are recomputed from scratch, never warm-started from the previous
bank. The first target update still uses the **unshifted** log kernel, and all
subsequent positive scaling/gauge steps are identical to scaling64, including
finite L=1. Batched target/self banks remain independent along the leading
axis. The always-FP64 internal dtype need not be another cache-key dimension.

Every replay returns an **owned clone** before casting back to the cost's
working dtype. Earlier couplings cannot change on a subsequent call, including
when the caller keeps a float64 result. Host locks serialize cache creation
and reuse; stream waits/recording protect input and owned-output lifetimes.
`clear_graph64_cache()` releases graph/private-buffer resources between probes;
it is not called per training update or serialized into checkpoints.

The original conservative exponent-range/finite check remains outside capture,
with one host decision per solve, not one per iteration. Unsafe ranges use the
unchanged FP64 log solver at the same L, then cast back; no numerical floor,
entropy change, early stopping, or tolerance is introduced. Valid unit-sphere
chordal costs are bounded by 4 and product-circle T-d costs by `4*d` (times
`cost_scale`). For unit scale, T7, epsilon >=0.3, N=M=256, the largest possible
range is 93.34, below the FP64 safety cutoff about 350.4. Runtime guarding is
retained even though these intended inputs have an analytic safe bound.

API evidence was read directly from the installed PyTorch 2.5 source:
`.conda/rmgm/Lib/site-packages/torch/cuda/graphs.py`, especially `CUDAGraph`,
`graph.__enter__/__exit__`, and the side-stream warmup in
`make_graphed_callables`. No compiler, custom CUDA extension, or new dependency
is used. Capture is performed lazily only after explicitly calling graph64 on
CUDA, never at import, construction of a CPU method, or ordinary test discovery.

`test_rwflow_graph.py` CPU verification passed for L=1/2/128/256/512/1024/2048:
bit-exact agreement with unchanged scaling64, agreement with the FP64 log
reference, first-iterate row-offset behavior, independent two-bank batching,
owned results under changing costs, guarded fallback, no global RNG changes,
no-grad results, sphere/T2/T7 uniform/concentrated fields, and exact CPU adapter
continuation. The original RW tests and worker regressions also passed (44
tests discovered, 42 passed, two explicitly gated CUDA tests skipped; CUDA
remained uninitialized). Full benchmark test discovery then passed: 111
discovered, 109 passed, only those two CUDA tests skipped, 5.862 seconds, Torch
threads/inter-op threads 1/1 and CUDA still uninitialized. The owned selection
fixture includes the parent's `iteration_choices[group][str(float(epsilon))]`
schema, synthetic RW epsilon=0.3, and unchanged candidate markers.

Additional CPU-only synthetic T7, epsilon=0.3, two independent 256x256 banks:

| Fixed L | Uniform Max Relative Column Error | Concentrated Max Error |
| --- | ---: | ---: |
| 128 | 0.00923866 | 2.38419e-7 |
| 256 | 0.00363153 | 2.38419e-7 |
| 512 | 0.00138694 | 2.38419e-7 |
| 1024 | 0.000285566 | 2.38419e-7 |
| 2048 | 0.0000131726 | 2.38419e-7 |

Graph64's eager CPU body was bit-exact with unchanged scaling64 at every
entry in this table. These are synthetic numerical checks, not real-training
evidence or GPU timings; the parent's real training-only probe remains needed.

After the parent's explicit exclusive GPU release, both
`paper.benchmarks2026.test_rwflow_graph.Graph64CUDATests` passed on
2026-10-26, PyTorch 2.5.1 / CUDA 12.1 / NVIDIA RTX 4090. Tests completed in
3.756 seconds (process wall time 5.096 seconds) under a 110-second watchdog;
the process exited zero and returned the GPU immediately. No model training,
experimental data, or data-quality scoring was performed in this GPU window.

Actual replay was compared with eager scaling64 and log fields for uniform
and concentrated sphere/T2/T7 point banks at the same L=128/512/1024. All
maximum field discrepancies were below 1e-5. Consecutive changed-cost results
matched scaling64; previously returned FP64 couplings remained unchanged and
had distinct storage. Captured cache entries were exercised, then explicitly
cleared at test teardown. GPU tests require `RWFLOW_GRAPH_CUDA_TESTS=1` and
are skipped during normal CPU discovery; enabling this flag requires a new
parent release if the GPU is reserved again.

Warm solver timings below include promotion, the finite/range guard, static
input copy, replay, owned result clone/cast, and residual diagnostics. Each
entry is the median of three synchronized measurements after one untimed call;
capture/three internal warmups are excluded. Batch shape is `(2,256,256)`,
synthetic uniform T7, epsilon=0.3:

| L | Eager Scaling64 | Graph64 | Same-L Speedup | Max Relative Column Error |
| --- | ---: | ---: | ---: | ---: |
| 128 | 8.887 ms | 2.478 ms | 3.59x | 0.00791216 |
| 512 | 34.373 ms | 8.553 ms | 4.02x | 0.00113827 |
| 1024 | 67.564 ms | 16.406 ms | 4.12x | 0.000160396 |

This measures the OT solve, not whole-update or whole-experiment throughput.
It demonstrates the expected launch-overhead reduction but does not remove
FP64 arithmetic cost. L512 still misses 1e-3 on this particular synthetic GPU
bank; L1024 meets it. Parent owns the separate real-training-only residual and
throughput probe before choosing the frozen L/solver. No count or config was
changed automatically by these tests.

The parent's prospective L-selection set is {128,256,512,1024,2048}, using
uniform and training-resampled independent banks and maximum relative column
error <=1e-3 before final test-configuration freezing. T7 epsilon=0.3 at L128
had roughly 1.45% column error in calibration, so simply accelerating that
under-converged count would preserve a solver handicap. Increasing fixed L to
meet the predeclared train-only numerical target is an explicit **numerical
refinement**, not entropy/model tuning or an author default. Graph equivalence
is always checked at the same L. There is no hidden automatic L change inside
the adapter; the parent freezes the selected count in the job config.

## Parent Worker Audit (Historical)

No parent-owned file was edited. Current shared `PointGenerator`, geometry, RNG interface, one-forward sampling, and state contract pass CPU integration.

- `worker.py:47` loads all tensors onto CUDA, including Adam's step counters. RW-Flow's `load_checkpoint` now corrects its own counters; other non-capturable Adam adapters may need the same treatment.
- `worker.py:45-50` automatically resumes only `last.pt`, whereas its exception path saves `interrupted.pt`. A rerun can therefore discard the freshest saved optimizer/RNG state, or all progress before the first scheduled validation. Parent should explicitly select/validate the intended resume checkpoint.
- `worker.py:95` already adds the current training interval before validation; if validation/checkpoint writing then raises, `worker.py:120` adds the same interval again plus validation time. Failed-run `train_seconds` can be inflated. Parent should reset the phase timer or distinguish training from validation in its exception accounting.
- The worker records the adapter source hash but not `common.py`; a source snapshot/hash covering the shared generator remains necessary for reproducible frozen runs. It should also validate saved job identity when resuming an existing output directory.

## Validation And Budget Suggestion

For parent consideration only, not an executed sweep: a six-point grid `epsilon in {0.1,0.3,1.0}` crossed with `flow_step in {0.05,0.1}`, holding the shared architecture, optimizer, batch size, cost scale, and solver count fixed. Choose with the existing fixed validation probes; never tune to the test split. Start with equal 20k-update validation-only screening, then allow 50k-100k updates for the frozen choice and all three seeds, with matched wall-time results alongside update counts. These are proposed adaptation budgets, not author defaults or demonstrated convergence requirements. A 2k-update run is only a smoke test. Parent may reduce the grid consistently if the allocated budget cannot support it.

For practical queue sizing, first measure 100-200 updates after warmup on the intended GPU without simultaneous CPU diagnostics. At 27 runs x 50k updates, total training time is `0.375 * milliseconds_per_update` hours, excluding validation/startup: 10/30/75 ms imply about 3.75/11.25/28.13 hours. A 100-step integration job checks the contract and speed, not convergence. Solver equivalence must be checked at the **same** L; choosing L64 instead of L256 is a separate documented numerical-budget decision. Freeze the solver/count/config after validation-only calibration; reserve test scoring for the selected final run.

Finite-iteration convergence must be checked separately from model quality. Logs report `max|sum_j P_ij-1|` and `max|M*mean_i P_ij-1|`. On a CPU-only uniform T7 check at N=M=256, epsilon 0.1, the relative column error was approximately 0.031/0.0082/0.0040/0.0019 at 100/256/512/1000 iterations. These values are solver diagnostics, not data performance. Sharp real distributions can be harder. Before freezing the queue, parent should inspect these residuals and, where needed, uniformly increase L (e.g. 512 or 1024); do not interpret finite-L fields as exact balanced OT. No early stopping or automatic regularization change is hidden in the adapter.

Other limitations: only chordal costs are implemented, not spectral or nonsquared-geodesic variants; minibatch fields are noisy approximations to population fields; the finite-step target can wrap past an injectivity radius for large flow steps; and shared-generator capacity, optimizer scheduling, and omitted EMA can change reported performance. Small flow steps mitigate wrapping but are not convergence guarantees.
