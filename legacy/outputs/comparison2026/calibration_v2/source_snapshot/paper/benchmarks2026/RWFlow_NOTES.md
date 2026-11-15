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

Sinkhorn uses fixed-count, log-domain alternating updates, target potential first and source potential last. The last update enforces the row conditional. `epsilon` is in full squared-chordal cost units, not a Gaussian standard deviation or a GeomLoss `blur`. With cost multiplied by `k`, multiplying epsilon by `k` preserves couplings; dividing `flow_step` by `k` also preserves targets. There is no epsilon scheduling or data-derived bandwidth.

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

The suite covers sphere/T2/T7 cost-gradient directional finite differences, full debiased Sinkhorn-energy finite differences including the self half-factor, an independent-bank field derivative, a closed-form 2x2 entropy/diagonal check, rectangular marginals, analytic singleton signs, cancellation, periodicity, cost/epsilon/step scaling, and complete detach behavior. Integration uses the actual shared `ManifoldOps` and `PointGenerator` for updates, independent-bank draws, sampling constraints, global RNG isolation, immutable checkpoints, and bit-exact CPU continuation.

No GPU tests, training run, dataset sweep, test-set inspection, or performance claim was performed.

## Validation And Budget Suggestion

For parent consideration only, not an executed sweep: a six-point grid `epsilon in {0.1,0.3,1.0}` crossed with `flow_step in {0.05,0.1}`, holding the shared architecture, optimizer, batch size, cost scale, and solver count fixed. Choose with the existing fixed validation probes; never tune to the test split. Start with equal 20k-update validation-only screening, then allow 50k-100k updates for the frozen choice and all three seeds, with matched wall-time results alongside update counts. These are proposed adaptation budgets, not author defaults or demonstrated convergence requirements. A 2k-update run is only a smoke test. Parent may reduce the grid consistently if the allocated budget cannot support it.

Finite-iteration convergence must be checked separately from model quality. Logs report `max|sum_j P_ij-1|` and `max|M*mean_i P_ij-1|`. On a CPU-only uniform T7 check at N=M=256, epsilon 0.1, the relative column error was approximately 0.031/0.0082/0.0040/0.0019 at 100/256/512/1000 iterations. These values are solver diagnostics, not data performance. Sharp real distributions can be harder. Before freezing the queue, parent should inspect these residuals and, where needed, uniformly increase L (e.g. 512 or 1024); do not interpret finite-L fields as exact balanced OT. No early stopping or automatic regularization change is hidden in the adapter.

Other limitations: only chordal costs are implemented, not spectral or nonsquared-geodesic variants; minibatch fields are noisy approximations to population fields; the finite-step target can wrap past an injectivity radius for large flow steps; and shared-generator capacity, optimizer scheduling, and omitted EMA can change reported performance. Small flow steps mitigate wrapping but are not convergence guarantees.
