# Riemannian MeanFlow Adapter

## Provenance And Classification

- Paper: Zhong, Sun, Zhao, Gong, Yin, [Riemannian MeanFlow for One-Step Generation on Manifolds, v3](https://arxiv.org/abs/2603.10718v3), ICML 2026.
- [Official repository](https://github.com/haolsun/Riemannian_MeanFlow), cloned without edits into `external/riemannian-meanflow-author/`.
- Pinned commit: `45d96bcd09a271ea3a770b2014865dabbac9eeeb`.
- Main classification: **source-derived native RMF loss and native TMLP architecture, with an explicitly reduced-budget benchmark adaptation**. Not an independent flow-matching approximation, teacher-distillation method, or full reproduction of reported paper scores.
- Optional `variant="rmf_mt"`: native decomposed objectives with two-task PCGrad. Core `variant="rmf"` is the default.

Author source anchors at the pinned commit:

- [MeanFlowModule: core loss, decomposed losses, time law, and sampling](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/models/meanflow_module.py).
- [TMLP and optional periodic Fourier encoder](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/models/arch/rfm/arch.py).
- [Two-time affine layers and time-dependent Swish](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/models/arch/rfm/diffeq_layers/basic.py).
- [Tangent projection and training dispatch](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/models/generative_module.py), [PCGrad](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/models/pcgrad.py).
- [Released experiment YAMLs](https://github.com/haolsun/Riemannian_MeanFlow/tree/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/configs/experiment/meanflow), [base amino model](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/configs/model/meanflow_amino.yaml).

The clone's imports/config targets still refer to `gfm`, while its package directory is `rmf`. This adapter avoids importing the Lightning, plotting, geometry, and logging stack. Production depends only on PyTorch plus the parent-provided geometry. Tests execute selected **unchanged original AST definitions** rather than maintaining a second handwritten reference loss. The upstream architecture carries Meta copyright/all-rights-reserved notices, preserved by attribution in the port. No LICENSE file was found in this clone; do not infer a new redistribution license.

## Parent Interface

```python
from paper.benchmarks2026.meanflow import create_method

method = create_method(ops, {"width": 128, "depth": 3, "lr": 2e-4}, device)
logs = method.update(real_batch, rng)
samples = method.sample(n, rng, nfe=1)
state = method.checkpoint()
method.load_checkpoint(state)
```

Use `{"variant": "rmf_mt"}` for optional RMF-MT, or `{"torus_fourier": 0}` for the exact raw-angle architecture setting used in the released torus YAMLs. The generator must be on the same device as the method. The parent owns initialization seeds, data/saved splits, batch sampling, RNG checkpointing, budget/schedule, validation-only selection, and final evaluation. This module reads no datasets and writes no outputs.

`checkpoint()` contains independent copies of model and AdamW state, resolved config, update counter, and provenance metadata. `load_checkpoint()` requires the same resolved config, then restores optimizer moments and counters as well as weights. There is no EMA, scheduler, hidden teacher, or internal RNG to restore. The parent must restore its supplied generator state for exact continuation. Sampling restores the previous training/evaluation mode and does not touch global training RNG.

All scalar loss logs are detached tensors on the computation device. Static counts/counters are Python integers. Normal GPU updates do not call `float`, `.item`, or perform scalar tensor-to-host conversions: finite checks use `torch._assert_async` on CUDA. That CUDA guard has not been exercised here because the GPU remains reserved by the parent. CPU nonfinite loss/gradient checks raise before stepping the optimizer.

## Native Mathematics Retained

The author's actual convention is **data at time 0 and uniform prior at time 1**:

```text
x_t, v = JVP(s -> proj(Exp_real(s * Log_real(noise))), t, 1)
u, du  = JVP(model, (x_t, r, t), (v, 0, 1))
target = stop_gradient(v - (t-r) * du)
L_RMF  = mean(inner(x_t, u-target, u-target))
sample_step = proj(Exp_z(-(t-r) * model(z, r, t)))
```

- Uniform times are `t ~ U[0,1]`, `r = U[0,1] * t`, not two independent times sorted afterward. The first `floor(0.75 * batch_size)` rows have `r=t`, matching the dataset configs. This is independent of data since the parent samples IID batches.
- Optional `time_distribution="lognormal"` reproduces the author's unusual `sigmoid(LogNormal(-0.4, 1))` law, not sigmoid-normal. It is not selected by the released dataset YAMLs.
- Interpolation, path velocity, random endpoints/times, and the entire regression target are outside reverse-mode AD. The primal model output retains parameter gradients. Lower time `r` is held fixed in the directional derivative. No learned `u(x,t,t)` is substituted for `v`: that belongs to the separate upstream `imf` variant, which is not implemented here.
- **Sphere caveat:** the released code differentiates the tangent-projected network in ambient coordinates and uses that full derivative in its target. It does not project `du` again or explicitly implement parallel transport. A derivative of a moving tangent field need not itself be tangent. This port deliberately retains and tests that native behavior; it should not be described as a new covariant-derivative implementation.
- RMF-MT uses `L1=mean(||u-v||^2)` and `L2=mean(inner(u, 2*(t-r)*stop_gradient(du)))`. Both use exactly the same geodesic and JVP. The stopped-target core gradient equals the decomposed sum gradient before conflict projection, although scalar loss values differ by the detached constant.
- For two fully shared tasks, the native random PCGrad ordering has no effect. Default `pcgrad_reduction="mean"` matches the released code. The paper's Eq. 16 instead sums the projected gradients; optional `"sum"` implements that stated rule. The upstream truthiness branch mistakenly treats the string `"sum"` as mean; this bug is not copied into the explicit alternative. There is no automatic second optimizer step from Lightning.
- Sampling uses exactly `nfe` network evaluations with descending equally spaced times, including the native one-step exponential-map update. Training uses JVPs and is not compute-equivalent to a one-forward-pass point-generator update.

## Architecture And Adaptation

The native network is **not** `common.PointGenerator`. It is a two-time vector field: every affine layer has independent zero-initialized additive `r`/`t` weights; every hidden activation uses the author gate `x * sigmoid(x * (beta(t)-gamma(r)))`, where beta and gamma are learned Softplus MLPs. Its final layer is randomly initialized as in upstream, not zeroed. `latent_dim` is accepted for shared configuration bookkeeping but unused. Prior samples come directly from `ops.base`.

The benchmark defaults are width 128, **3 main affine layers**, batch 256, AdamW lr `2e-4`, weight decay `0.01`, equal-time fraction `0.75`, no clipping, no scheduler. Width/depth/lr/batch are the requested common benchmark scale, not the author's released budgets. The time-dependent activation MLPs add trainable parameters, so matching width/depth does not mean matching parameter count.

| Representation | Default RMF parameters | Common point G parameters |
| --- | ---: | ---: |
| Sphere XYZ | 51,721 | 17,411 |
| Torus 2 angles, periodic encoder | 51,718 | 17,410 |
| Torus 7 angles, periodic encoder | 53,653 | 19,335 |

There are no extra trainable components in RMF-MT. The method exposes `parameter_count` and includes it in metadata/checkpoints. Parent job logging should still count `method.model.parameters()` directly.

For periodic tori, default `torus_fourier=1` enables the author's existing optional Fourier encoder: `[sin(x), cos(x)]`, then the same TMLP. This gives a smooth periodic field and periodic JVP across coordinate seams. It is an **architecture-setting adaptation**, since the released dataset configs omit Fourier features and feed raw canonical angles. `torus_fourier=0` is provided for source-setting equivalence, but that raw-angle network is not a globally periodic field. Both settings retain wrapped shortest-angle Log/Exp geometry and never use Euclidean chords between angle values.

Geometry is delegated to the common sphere/torus wrapper. It uses stable sphere formulas rather than geoopt's clamped-acos/small-step branches. Native numerical equivalence is checked in float64 on nonsingular points; bitwise agreement near coincident or antipodal points is not claimed. Exact antipodes and torus half-period cut loci have nonunique geodesics and are not an analytic equivalence domain. This adapter does not replace or change the common geometry.

The author NaN-dropping path is not retained: it reduces over two dimensions even for `(batch, coordinates)` inputs. Nonfinite runs fail instead of silently discarding arbitrary data. Upstream source files remain unmodified.

## Released Configurations And Budgets

The following table follows the **pinned repository YAMLs**, not the conflicting paper table. All use AdamW, uniform times, equal-time fraction 0.75, `my_method="pcgrad"`, raw torus angles, and a configured per-epoch cosine schedule to `1e-6`. The class's bare `prop_equal_times` default is 0; the experiment YAMLs override it to 0.75. `author_config(dataset, source="repo")` exposes these references without automatically imposing their epochs/scheduler.

| Dataset | Width/layers | Batch | LR | WD | Min/max epochs | Repeat factor | Native parameters |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| volcano | 2048/12 | 8192 | 1e-4 | .01 | 100/2000 | 1550 | 44,954,377 |
| earthquake | 2048/10 | 8192 | 1e-4 | .001 | 100/700 | 210 | 36,020,489 |
| fire | 2048/10 | 8192 | 1e-4 | .01 | 100/600 | 100 | 36,020,489 |
| flood | 2048/7 | 8192 | 1e-4 | .01 | 100/900 | 260 | 22,619,657 |
| general | 512/4 | 2048 | 1e-4 | .001 | 8000/10000 | 1 | 731,398 |
| glycine | 512/10 | 4096 | 1e-4 | .001 | 8000/10000 | 10 | 2,714,374 |
| proline | 512/4 | 4096 | 1e-4 | .001 | 8000/10000 | 18 | 731,398 |
| prepro | 512/4 | 4096 | 1e-4 | .001 | 8000/10000 | 20 | 731,398 |
| rna | 512/6 | 4096 | 1e-4 | .01 | 1000/1450 | 14 | 1,397,525 |

[Earth dataloader](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/data/earth_datamodule.py) and [torus dataloader](https://github.com/haolsun/Riemannian_MeanFlow/blob/45d96bcd09a271ea3a770b2014865dabbac9eeeb/rmf/rmf/data/amino_datamodule.py) split 80/10/10 with seed 42, then repeat the training subset using the factors above. They retain the partial last batch. At one device, author steps/epoch are `ceil(N_train * repeat_factor / batch_size)`, not `ceil(N_train / batch_size)`. Using the paper's dataset sizes gives approximate maximum update budgets, in table order: **252000, 88200, 75600, 111600, 540000, 260000, 270000, 270000, 37700**. These are config-derived estimates, not observed convergence times or timings. The parent uses its own saved split and must not recreate the author split for this comparison.

[Paper Appendix I, Table 19](https://arxiv.org/html/2603.10718v3) instead reports lr `5e-4` and WD `.01` for every dataset; earthquake batch 4096; general batch 4096; glycine/proline/prepro/RNA batches 2048; flood 700 epochs; four protein subsets 5000 epochs; RNA 2000 epochs. Layer counts/widths agree with the repository. `author_config(..., source="paper")` makes this distinction explicit. Its repeat-factor field is labeled repository-only, since the paper table does not specify that factor. Do not silently combine these sources into an alleged exact author run.

The repository's default callbacks select the best validation `val/mmd-001` checkpoint and early-stop with patience 100 validation checks. Validation MMD runs every second epoch. Reported scores average five runs. The adapter intentionally leaves all selection to the parent's fixed-probe **validation-only** protocol. No tests or hyperparameters here were chosen using test data.

Convergence guidance: first check finite losses, correct sampling direction, valid manifold outputs, and periodicity; then compare validation curves at predeclared compute/update budgets. A 2k-update small-model run is a bounded adaptation, not evidence that the original 22M-45M-parameter sphere models or 38k-540k-update author schedules converged. The 75% diagonal samples supervise instantaneous velocities, while only 25% carry unequal-time average-velocity correction. Core RMF versus RMF-MT, raw versus periodic inputs, and different budgets must be labeled separately; do not select these alternatives using test scores. No empirical convergence or performance claim is made before the parent's authorized GPU queue runs.

## CPU Verification

```powershell
.conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_meanflow -v
```

13 test methods pass using Python/PyTorch 2.5.1 on CPU. Coverage includes:

- Unchanged author TMLP output, full directional JVP, and reverse parameter gradients; raw and periodic features, including seven angles.
- Unchanged author core loss, both native MT loss terms, all parameter gradients, and 1-/4-step sample updates against the real common wrapper. Float64 comparison tolerance is `atol=rtol=2e-10` away from cut loci.
- Closed-form sphere geodesic/path velocity, wrapped torus seam direction, directional JVP finite differences, analytic stopped-target parameter gradients and frozen-target finite differences.
- Retention of the native ambient sphere derivative's nonzero normal component; fixed lower-time derivative and complete target stop-gradient.
- Unchanged author PCGrad default algebra, aligned/conflicting/zero gradients; only its `torch.compile` decorator is disabled in this CPU reference harness to avoid compiler processes.
- Exact supplied time laws, periodic field/JVP invariance, descending sample sign, exact NFE count, mode restoration, global RNG isolation, nonfinite fail-before-step, static parameter counts.
- Core/MT updates and bitwise-identical checkpoint continuation on sphere, torus-2, torus-7. Subsequent updates exercise restored AdamW moments, not just restored weights.

The native sampler's `linspace` uses the global default dtype, so the float64 reference test temporarily sets that default and restores it. No upstream code is rewritten. Tests parse selected source definitions, so an upstream unrelated invalid-escape deprecation warning may appear. No training datasets, validation targets, or test targets are loaded by this suite. No CUDA computation, training jobs, descendant agents, or compiler subprocesses have been started.
