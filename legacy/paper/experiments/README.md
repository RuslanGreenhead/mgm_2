# Controlled Empirical Supplement

The controlled training/evaluation sidecar lives in this directory. Generated
experiment artifacts go to `outputs/paper_icml2026/`. Parent manuscript-rendering
scripts and the additional `noisy_torus_study.py` diagnostic live one level above.
The original core trainers are reused rather than overwritten by this sidecar.

Final primary table authority is `outputs/paper_icml2026/controlled_independent/`.
The earlier `controlled/` bundle is preserved as a contaminated historical
ablation. `controlled_independent_attempt1/` is explicitly excluded: its wrapper
briefly changed a constant factor in the Torus RFM loss; it was stopped and
preserved after 13 completed jobs. The clean rerun checks exact RFM expression
equivalence and midpoint loss/gradient equivalence before training.

## Locked Protocol

- Real sphere/torus data: the existing local files, author-style 80/10/10 row
  split, data seed 0. Model seeds 0, 1, 2 share the same split. Protein/RNA
  splitting is by row, not by molecular source. Historical test exposure from
  legacy experiments is disclosed; these are not pristine unseen benchmarks.
- Width 128, three layers, SiLU, critic/RFM time Fourier features 8, batch 256,
  AdamW 0.0002, no weight decay, float32, TF32 disabled, one critic update and
  one generator update per MGM iteration. RFM uses one update per iteration.
  On torus, the shared TangentCritic/RFM field concatenates RAW angles with
  time features (critics.py:94); it has no circular spatial features or enforced
  periodicity. The natural metric/MMD embedding is periodic but this network
  architecture is not. Seam limitations are shared by MGM critics and RFM;
  no architectural correction is silently introduced in this supplement.
- MGM final step 2000; RFM fixed checkpoints 2000 and 10000. The 2000-step
  comparison matches batch and iterations, NOT update counts, FLOPs or wall
  time. The 10000-step RFM is explicitly a longer-budget reference, not an
  author-budget reproduction. Measured training and setup costs are saved.
- Train-initialized GMM at step 0 and 2000, plus random-initialized MLP MGM on
  fire/general/RNA. A parameterization-matched MMD-objective MLP is trained
  on these three representative datasets for 2000 steps on fresh batches of
  256, with the exact same fixed training-only kernel used for evaluation.
  This is not a reproduction of KGDM, RW-Flow, R-WGF or Riemannian MeanFlow.
  GMM component logits receive no pathwise gradients through
  searchsorted; train-fitted mixture weights therefore stay fixed. Center and
  scale changes are saved in completed.json. No exact likelihood is claimed.
- No checkpoint, model, or hyperparameter is selected using validation/test.
  Final validation and test metrics are both saved. Learning curves contain
  training objectives; no test-based early stopping occurs.
- Three evaluation draws per checkpoint are Monte Carlo replicates, NOT three
  extra training seeds. First average draws within seed; then mean and sample
  SD across the three independent training seeds. Seed-level values are saved.
- Targets are selected without replacement, fixed per dataset and split, with
  min(1024, split count) points. Generated batch size matches target batch size.
  The same target and draw seeds are used for every method/training seed.
- MMD2 is the biased V-statistic, including diagonals, using k(x,y) =
  exp(-||embed(x)-embed(y)||^2 / bandwidth2). Bandwidth2 is median positive
  squared distance among a fixed training-only reference of at most 1024
  points. Torus embedding concatenates cosine/sine; sphere uses ambient XYZ.
  Coverage threshold is median leave-one-out NN distance on that training
  reference. Neither kernel nor threshold depends on generated/test samples.
- Sample NN uses great-circle distance on sphere and wrapped Euclidean angle
  distance on torus. Mesh NN is explicitly ambient Euclidean, not intrinsic.
- RFM uses the legacy expmap/projected Euler sampler with midpoint-in-time
  evaluation. It calls the field ONCE per step: 128 steps = 128 actual NFE,
  verified by a forward hook. This is not a spatial midpoint/RK2 integrator.
- Timings use perf_counter; CUDA is synchronized before/after GPU sampling and
  training. Two GPU sample warmups and one CPU warmup are excluded. CPU times
  use one PyTorch thread, three repeats, and the same sample count. Sampling
  timings include RNG/control overhead and exclude metric computation.
- Preliminary checkpoints contain model/critic state, config and final step,
  without complete optimizer/RNG state. The 54 final independent reruns also
  save optimizer and all training RNG streams; see Independent RNG Correction.
  The copied 18 earlier MLP jobs retain their original continuation limitation.

## Commands

From the workspace root, in PowerShell:

```powershell
$env:PATH = "$PWD\.conda\rmgm\Library\bin;$env:PATH"
& '.conda/rmgm/python.exe' -B 'paper/experiments/test_controlled_suite.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/controlled_suite.py' benchmark
& '.conda/rmgm/python.exe' -B 'paper/experiments/controlled_suite.py' train --datasets fire general rna volcano earthquake flood glycine proline prepro --mgm-steps 2000 --rfm-steps 10000 --max-seconds 1050
& '.conda/rmgm/python.exe' -B 'paper/experiments/mmd_control.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/controlled_suite.py' mesh --draws 3
& '.conda/rmgm/python.exe' -B 'paper/experiments/controlled_suite.py' summarize
& '.conda/rmgm/python.exe' -B 'paper/experiments/audit_legacy.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/calibration.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/reevaluate.py' --checkpoint 'outputs/paper_icml2026/controlled/fire/mgm_gmm/seed0/step2000.pt' --output-dir 'outputs/paper_icml2026/reevaluation/fire_gmm_seed0' --device cpu --seed-base 200000
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_artifacts.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/stabilization.py' --config 'paper/experiments/stabilization_general.json'
```

The run deadline is checked between complete jobs; progress.json reports actual
started/finished counts. No partial job is silently counted as completed.
An initial execution completed fire/GMM seed 0, then hit a CPU timing-wrapper
attribute error after training fire/MLP seed 0. The timing helper was corrected;
the incomplete MLP job was rerun. No metric/protocol or training change was made.
The MMD/RNA-MLP extension is recorded separately in mmd_protocol.json and does
not replace the primary all-nine protocol manifest.
One primary-protocol RNG caveat is retained, not concealed: the original CPU
latency helper called torch.manual_seed, which also reset CUDA's global RNG
at fixed evaluation boundaries. Initialization, target-index and latent seeds
still vary across training seeds, but part of the hidden-time/base-noise
schedule after GMM step 0 / RFM step 2000 is shared across runs. These are three
distinct model-training seeds, not three fully independent RNG-stream bundles;
the across-seed SD is conditional on that shared schedule. No evaluation
metric controls this reset or the checkpoint choices. The helper is corrected
for extension/future runs; the exact original primary source is hash-verified
and archived as source_snapshot/primary_executed_runner.py. MLP MGM/MMD have
no intermediate evaluation before the final checkpoint, so their training
streams are unaffected by this CPU-timing caveat.

## Legacy Mesh Reevaluation

All six datasets, all three existing generator heads, final last.pt at step
800, seed 0/effective data_seed 0, fixed held-out target of exactly 1024 points,
three generated draws. No best.pt selection and no new mesh training. Geometry
buffers are restored directly from checkpoints to avoid recomputing eigenvectors.
The radial map is the trained soft-radius-plus-projection map, not a guaranteed
bijection and not the separate ray-intersection smoke utility. Repeated mesh
draws measure MC variation only: there is still just one legacy training seed.

## Outputs

protocol.json locks jobs and code hashes; benchmark.json contains 100-step
timings; progress.json and per-run completed.json record completion and compute.
datasets/*/manifest.json records split/data hashes and metric references.
controlled/*/*/seed*/ contains step0/final checkpoints, all generated draw NPZs,
training loss curves and raw validation/test evaluation JSON. results.json is
the flat evaluation table; summary.json aggregates MC draws within seeds before
computing across-seed SD. legacy_mesh_results.json and legacy_mesh/*/ hold mesh
samples, fixed target indices and tangency probes. Do not pool legacy and new
metrics: their kernels, thresholds, batch sizes, and selection rules differ.
calibration_results.json and calibration_summary.json contain CPU-only empirical
train-resampling reference draws on the same held-out targets. Their SD is
across MC draws and n_training_seeds is zero. This is a memorization and
finite-sample reference, not a universal MMD floor or optimized generative model.

## Validation-Only Stabilization Study

The unstable all-nine results remain unchanged. The separate General seed-0
study is exploratory and was requested after those failures were observed;
it is not a fresh preregistered benchmark and does not evaluate test again.
The six trial configs are fixed in stabilization_general.json before launch.
All MGM trials use batch 512, 500 frozen-generator critic warmup updates,
generator LR 0.00001, critic LR 0.0002, final generator step 2000. Four use five
critic updates per generator step; a fifth uses ten. The raw-coordinate MLP
control isolates the effect of periodic features from slower two-timescale
optimization. Random GMM centers are uniform-random, not data-fitted; their
mixture logits remain fixed. The train-kmeans GMM is a separate retention
control and is excluded from random-initialization candidate selection.

PeriodicTangentCritic uses sin/cos spatial harmonics 1..4, time Fourier features
1..8, width 128 and three layers for BOTH MGM critic and RFM field. No core
module is changed. Its seam-periodicity is tested. RFM fixed checkpoints 12500
and 22500 match the total optimizer-update counts of the 5- and 10-critic MGM
trials (including warmup), not FLOPs or identical update costs. Both use
128-call Euler sampling. Synchronized measured training times are recorded.

Training and validation sample probes occur at fixed initial/warmup/joint/final
checkpoints, three MC draws each, using the same training-only reference kernel
and fixed validation points as the main General protocol. Test is not scored.
The RNG helper is fully isolated in this study. Final validation MMD alone
selects a random-initialized periodic MGM trial; intermediate checkpoints are
diagnostics, not candidates. A selected seed-0 trial still needs locked
independent-seed confirmation before any robust efficacy claim.

Outputs are in outputs/paper_icml2026/stabilization_general/:

- protocol.json: timestamped fixed trial/config/source hashes and test policy.
- <trial>/config.json: architecture, seed, reference and validation hashes.
- <trial>/<phase>_step<N>_metrics.json: array of train/val rows with trial,
  phase, step, generator_step/field_updates, split, draw, training_seed,
  sampler_nfe and the common sample metrics. There are no test rows.
- Matching samples.npz: generated_draw_0/1/2; checkpoints also save model,
  critic, optimizer and RNG states for continuation on the same runtime.
- <trial>/learning_curve.json and completed.json: loss probes, true update
  counts, parameter changes, synchronized training time and initial/final
  validation summaries. results.json contains all six completed trial summaries.
- selection.json: timestamped fixed-final-step validation selection, selected
  candidate and explicit zero new test evaluations.

The capped second phase is stabilization_refinement.py: exactly two additional
periodic/random-MLP trials with G LR 0.0001 and 0.0002, all other settings fixed.
Its prospective protocol, full results and separate selection.json are under
outputs/paper_icml2026/stabilization_general_refinement/. It selects among all
five random-initialized periodic MGM candidates from both phases, using only
fixed-final-step validation means. Neither phase evaluates test. Earlier trial
outputs and the initial selection are preserved. Independent-seed confirmation
remains future work, not an assumed success.

## Independent RNG Correction

`independent_suite.py --prepare` creates a new immutable source tree. The worker
is executed FROM that snapshot, not live project files. It reruns all 54
GMM/RFM jobs with unchanged hyperparameters/objectives and copies the 18
unaffected MLP-MGM/MMD jobs byte-for-byte, with per-file provenance hashes.
The primary all-nine protocol and earlier source_snapshot are not overwritten.

Final table paths retain the original row/NPZ schemas:
`outputs/paper_icml2026/controlled_independent/controlled/<data>/<method>/seed<S>/`.
The new root also has datasets/, protocol.json, progress.json, results.json,
summary.json, provenance.json, objective_equivalence.json, frozen_sources.json,
executed_source_hashes.json, run_manifest.json and final verification.json.

Training generators are separate and derived from training seed S: target
50000+S, latent 60000+S, midpoint flip/RFM time 70000+S, RFM base 80000+S.
Initialization retains S. All four stream states plus global CPU/CUDA state
and optimizer states are saved in every rerun checkpoint. Evaluation must leave
ALL these states exactly unchanged; the worker asserts this at every checkpoint.
Private RFM bases have the same sphere-normalized-Gaussian / uniform-torus laws.
The original RFM `manifold.inner(...).mean()/dim` scaling is preserved exactly,
including its additional coordinate mean on FlatTorus. This is an RNG repair,
not a normalization or hyperparameter retune. GPU reductions are not promised
bitwise deterministic across hardware; saved state supports continuation.

All executed project modules must originate in the frozen source tree; source
hashes are checked before/after each run and module hashes recorded at finish.
The CPU verifier checks all 54 reruns, all copied files, 108 saved RNG checkpoint
states, evaluation invariance, distinct states across the three training seeds,
fixed split/kernel identity, actual NFE, tensor/NPZ finiteness and 1080 rows.

```powershell
& '.conda/rmgm/python.exe' -B 'paper/experiments/independent_suite.py' --prepare
& '.conda/rmgm/python.exe' -B 'outputs/paper_icml2026/controlled_independent/source_snapshot/paper/experiments/independent_suite.py' --worker --root "$PWD"
& '.conda/rmgm/python.exe' -B 'paper/experiments/correct_sphere_metrics.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_independent.py'
```

Do not prepare over an existing bundle. Verification is read-only except for
writing verification/package metadata in the new output root. Two separately
recorded General validation-only 10000-G-step follow-ons are queued after all
54 reruns: random MLP and random GMM, periodic critic, batch512, G/C LR2e-4,
five critic steps and 500-step critic warmup. They use stabilization_long.py
from the same frozen tree and write stabilization_general_long/. They do not
run the optional stochastic-torus-path sweep or evaluate new test metrics.

## Sphere Metric V2

Reviewer finding 3 exposed unnormalized float32 sphere dot products producing
spurious nonzero duplicate angles. The analysis helper now normalizes float64
vectors and computes 2*atan2(norm(xhat-yhat), norm(xhat+yhat)); reference radius
explicitly excludes identical normalized directions. Saved-float32 duplicate,
antipodal, synthetic duplicate-radius and actual Volcano-radius tests cover it.
The corrected Volcano training reference radius is 0.010018575232836861 radians,
not 0.00497482; its 663 reference rows contain 239 unique coordinate vectors.

Training is unaffected and the frozen training source is preserved. After all
54 reruns, correct_sphere_metrics.py rescores every Sphere NN/coverage/radius
from existing NPZ files, leaving MMD and ALL arrays unchanged. Raw score JSONs
and manifests are preserved in sphere_metric_v1_raw.zip, with versioned raw/
corrected hashes in metric_provenance.json. Calibration is similarly corrected
and its original JSON retained in sphere_metric_v1_calibration.zip. The new
metric source and tests are copied separately into metric_source_snapshot/;
neither training source_snapshot is modified.

Final table generation must wait for controlled_independent/FINAL_TABLES_READY.json
and passing verification.json with sphere_normalized_float64_atan2_v2. Interim
Sphere evaluation JSON emitted by the frozen trainer is not final-table-ready.

```powershell
& '.conda/rmgm/python.exe' -B 'paper/experiments/correct_sphere_metrics.py' --calibration-only
& '.conda/rmgm/python.exe' -B 'paper/experiments/correct_sphere_metrics.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_independent.py'
```

Unrelated Python GPU processes were observed during part of the independent
rerun. resource_contention_observation.json and resource_contention.jsonl
record that observation without controlling other jobs. Synchronized timings
remain actual wall times, not isolated hardware-cost measurements; no speedup
claim should be inferred from contended runs.

## Fixed Queue Closeout

All 54 independent reruns completed at 2026-10-23T20:03:25 UTC. Primary
verification passes with 74 frozen source files, 21 executed project modules,
108 evaluation-invariance checks and 216 RNG groups distinct across three
training seeds. Sphere-v2 correction covers 468 primary rows and 24 calibration
rows. `controlled_independent/FINAL_TABLES_READY.json` is the table-readiness
gate; old v1 scores remain archived and sample arrays/MMD are unchanged.

The two long validation-only trials are invoked from the frozen source tree:

```powershell
& '.conda/rmgm/python.exe' -B 'outputs/paper_icml2026/controlled_independent/source_snapshot/paper/experiments/stabilization_long.py' --root "$PWD" --output-root "$PWD/outputs/paper_icml2026/stabilization_general_long"
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_empirical_closeout.py'
```

The closeout check requires both trials at 10000 generator updates and 50500
critic updates, verifies all 16 saved checkpoints and 96 train/validation
metric rows, reruns the 10 metric/protocol regression tests, and writes
`outputs/paper_icml2026/GPU_RELEASED.json` only on success. No optional noisy
path experiment is owned by this sidecar. The inherited base configuration's
selection description mentions 2000 steps; the long wrapper and its actual
selection use fixed-final 10000 steps, not an intermediate minimum. The two
long trials are exploratory seed-0 evidence, not independent-seed confirmation.

Closeout completed: both long trials exited successfully at 2026-10-23T20:10:10
UTC; GPU ownership was explicitly released at 20:10:47 UTC. The closeout checks
pass, including all 16 long-trial checkpoints, 96 train/validation rows and all
10 CPU tests. Final validation MMD2 is 0.034906396 (MLP) and 0.016395038 (random
GMM), both above the earlier periodic-RFM value 0.001909495. No new test was
evaluated. Exact commands, output schema and paths are recorded in
`outputs/paper_icml2026/CLOSEOUT.md`; the release marker is `GPU_RELEASED.json`.
