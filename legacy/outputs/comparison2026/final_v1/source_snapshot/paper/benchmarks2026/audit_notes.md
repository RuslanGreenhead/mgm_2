# Bounded Independent CPU Protocol Audit

Date: 2026-10-26. Reviewed the requested working-tree benchmark modules and
method notes, plus the prospective MGM refinement plan. No GPU work,
descendant agents, actual dataset arrays, historical outputs, grasp artifacts,
or plot artifacts were accessed or changed. Only this file and
`test_worker_protocol.py` were added. Temporary fixtures contain synthetic
arrays and are deleted by the tests. The running frozen calibration queue was
not inspected, stopped, or modified. Findings concern the reviewed source, not
an assertion that an existing result has already been corrupted.

## Current Recheck: PASS Before Prospective Freeze

The parent's four demonstrated fixes now pass as **ordinary regression tests**;
all four `expectedFailure` decorators were removed. The parent also added the
`pickle.UnpicklingError` load fallback and atomic JSON replacement. No new
blocking defect was observed in the revised protocols checked below. This is
a CPU protocol pass for prospective source freezing, not a GPU performance or
final-artifact certification, and does not retrospectively change old phases.

Full discovery on 2026-10-26: **103 tests passed, zero errors, zero failures,
zero expected failures, zero skips**, 4.858 seconds of test execution:

- 65 existing common-geometry, metric, MGM/KGDM/RW/RMF checks.
- 24 parent-owned synthetic plot/reporting contract checks, unchanged here.
- 14 tests in `test_worker_protocol.py`, all normal regressions.

The command used `.conda/rmgm/python.exe -B` and unittest discovery with
`start_dir='paper/benchmarks2026', pattern='test_*.py', top_level_dir='.'`,
verbosity 2. Before importing NumPy/Torch it set `CUDA_VISIBLE_DEVICES=''` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`; both Torch thread
counts were set to one. End-of-run verification reported
`CPU_ONLY True THREADS 1 1 EXPECTED_FAILURES 0`.

Specific revised contracts verified:

- Calibration never materializes the guarded synthetic test archive member.
- An unfinished checkpoint with a different requested seed is rejected.
- A deliberately truncated **old-style `last.pt`** produces
  `pickle.UnpicklingError` when loaded directly, while the worker skips it and
  recovers the intact training-stage `interrupted.pt`.
- A failed step-2 **`last.tmp.pt`** write leaves the prior step-1 `last.pt`
  byte-for-byte unchanged. Its complete step-2 interruption snapshot is then
  selected; the corrupt temporary file is not a recovery candidate.
- Each of `last.pt`, `interrupted.pt`, and `best.pt` is correctly selected when
  it holds the highest training step among otherwise intact candidates.
- Higher-step interruption files marked `validation` or `test` are ignored.
- A failure during validation does not publish a training-stage interruption;
  the intact initial best checkpoint can restart the synthetic job.
- A failure after test sampling does not save selected sampling weights as
  final training state. Retry restores step-2 training first, then step-0 best
  sampling weights, without repeating a generator update or modifying last.
- Selection ignores an unplanned completed folder, and rejects a planned
  `job.json` whose seed differs from the frozen job metadata.
- Failed JSON replacement preserves an existing completion flag's exact bytes,
  or leaves an initially absent completion flag absent. A sibling temporary
  JSON cannot masquerade as the committed completion record.

Worker stages were reviewed at both transitions: validation sets
`training_active=False, stage='validation'` before evaluation; successful
validation returns to `stage='training'` before checkpoint publication. Final
test loading/scoring switches to `stage='test'` first. The exception handler
saves only training-stage interruption states and does not add validation/test
time to training time. The stage regressions exercise actual failures in both
evaluation paths, not just fabricated stage labels.

Reviewed recheck source SHA256 values:

- `worker.py`:
  `b7a779dd5d482fc2738e8a3b5d54b036a00a041456a0c77378402c7eba03a3d9`
- `select_configs.py`:
  `dda29b87adf1a8486b274c34e12ea701b7223bc07730f8f51c54c646d0a3b09f`
- `data_metrics.py`:
  `4a78fde200b02515cda4a3a48d1b5fdbbd62e24661658c85bcc82361b9bd6eae`

Common geometry and all four method-module hashes remain equal to the initial
audit. No algorithm/factory implementation was edited by this audit. Broader
hardening suggestions in the historical record below are not claims of
additional checks implemented by this recheck; the four originally reproduced
failures and the requested atomic/stage regressions are resolved.

## Initial Findings (Historical)

The following records the initial defects and recommendations, including their
old source line numbers. The current status above supersedes the initial
failure results; these are not outstanding findings against the rechecked code.

### P1: Calibration materializes the test target before selection

- Location: `worker.py:34`; the test flag is not checked until `worker.py:104`.
- `arrays["test_target"]` is evaluated even for `score_test=False`. This violates
  the strict no-test-target-access boundary. It does NOT establish that test
  metrics influenced tuning: the scoring branch remains disabled.
- Proof: `test_calibration_never_materializes_test_target` supplies a lazy
  synthetic archive whose test member raises on access. The exception occurs
  at line 34 before calibration can start.
- Parent fix: materialize the test member only inside the final test-scoring
  branch, after recipe/checkpoint selection. Preserve old frozen sources;
  record their eager materialization honestly instead of claiming no access.

### P2: Resume silently accepts a different job identity

- Locations: `worker.py:43`, `worker.py:47`, `worker.py:51`.
- The checkpoint contains `job`, but restore never compares it with the
  requested job. Changing seed 0 to seed 1 in an unfinished directory restores
  seed-0 weights/RNGs and then rewrites `job.json` and completion as seed 1.
  Config checks in some adapters do not protect seed, data, or evaluation policy.
  Checkpoints also do not bind the data manifest or executed-source hashes.
- Proof: `test_resume_rejects_changed_training_seed` interrupts after a valid
  `last.pt`, changes only the job seed, and expects rejection. None occurs.
- Parent fix: validate immutable job identity and source/data fingerprints
  before loading or rewriting metadata. Define permitted budget extensions
  explicitly. Also verify completed-job identity before the unconditional skip
  at `run_jobs.py:35`.
- The frozen queue's plan equality already limits normal plan changes; the
  demonstrated failure is reuse of an incompatible unfinished output folder,
  not a claim that the unchanged queue currently has duplicate seeds.

### P2: Torn checkpoint writes cannot recover an intact interruption snapshot

- Locations: `worker.py:43`, `worker.py:47`, `worker.py:100`, `worker.py:123`.
- Checkpoints overwrite their destinations directly. If writing `last.pt`
  truncates it and then raises, the handler can save a complete
  `interrupted.pt`, but startup unconditionally loads the corrupt `last.pt`.
  Restart fails with `UnpicklingError` despite recoverable state.
- Proof: `test_resume_recovers_complete_snapshot_after_torn_last_write`
  injects this write failure, successfully loads `interrupted.pt` at the
  completed synthetic training step, and then observes restart fail on last.
- Parent fix: atomic same-directory temporary writes and replacement for
  checkpoints; preserve a valid previous checkpoint. Document/validate any
  recovery fallback and its selected-best lineage. Ordinary rollback to the
  last intact validation checkpoint is otherwise consistent with the notes.

### P2: Config selection is not restricted to the frozen candidate lists

- Locations: `select_configs.py:13`, `select_configs.py:18`,
  `select_configs.py:21`.
- Selection walks every completed child directory, rather than the jobs in
  each phase's frozen protocol. A stale/unplanned validation-only folder with
  a lower score is eligible and wins. Recording the phase protocol hash does
  not bind this extra candidate to that protocol.
- Proof: `test_selection_excludes_completed_jobs_outside_frozen_plan` creates
  all three required synthetic phases, then inserts an unplanned MGM folder.
  The frozen sphere MGM config comes from that folder, not a declared job.
- Parent fix: enumerate protocol jobs, verify job/completion identity and
  provenance, and reject extra candidate folders or exclude them explicitly.
  Apply the same whitelist principle when inheriting the old MGM noise choice.
  No real phase directories were opened to check whether extra folders exist.

## CPU Verification

Used `.conda/rmgm/python.exe -B`, with `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, and both Torch thread
counts set to one. Worker integration uses the real MGM on CPU and replaces
only device/CUDA services and file inputs with synthetic fixtures. Unexpected
CUDA initialization raises. The test module performs no subprocess launch.

Initial run: **73 tests, 69 ordinary passes, 4 expected failures**, 2.774 seconds
of test execution. The 65 existing tests passed. Initial added checks cover worker
resume, all four reproductions above, independent NumPy MMD on sphere/T2/T7,
draw means followed by training-seed sample SD, duplicate-seed rejection, and
the six-job refinement plan. Running the initial failing bodies without their
decorators confirmed the expected source locations and failures, rather than
harness/import errors. The recheck removed those decorators and passed the
expanded suite as documented above.

No additional bug was found in the tested explicit RNG isolation, restored
KGDM EMA sampling/one-NFE counting, native RMF/core-MT continuation, manifold
constraints, or same-L RW log/scaling/scaling64 numerical equivalence on CPU.
The independent synthetic MMD recomputation agrees to 12 decimal places.
Aggregation computes sample SD across training-seed draw means, not across
pooled draws. Existing method tests cover optimizer/EMA continuation; the new
worker no-extra-update resume check preserves model and explicit RNG states.

Source freezing statically resolves the worker/method/common imports under the
snapshot and copies the executed KGDM field plus license. No executed-snapshot
or final-artifact audit was run: that would require reading actual outputs and
test arrays. `audit.py` recomputes MMD and checks hashes/counts, but is not a
substitute for validating checkpoint identity before resume. No GPU numerical
or latency guarantee follows from these CPU tests.

## Prospective MGM Refinement And Fairness

The parent added `make_mgm_refinement_plan.py` before final test scoring, and
`select_configs.py:8` now requires completed `mgm_refinement_v1` in addition to
the two prior phases. Existing frozen phases are unchanged. The six new jobs
are two settings on fire/general/RNA, seed 0, 4,000 generator updates each,
validation only. They inherit the old validation-selected noise setting:

- Critic width 256, 5 critic updates/G update, critic LR 8e-4.
- Critic width 256, 10 critic updates/G update, critic LR 4e-4.

The common generator remains width 128, depth 3. Exact parameter counts from
CPU construction with the current depth/harmonics settings:

| Geometry | Common G | Old 128-wide C | Refined 256-wide C | Native small RMF |
| --- | ---: | ---: | ---: | ---: |
| Sphere XYZ | 17,411 | 19,459 | 71,683 | 51,721 |
| Torus 2 | 17,410 | 20,994 | 74,754 | 51,718 |
| Torus 7 | 19,335 | 26,759 | 86,279 | 53,653 |

Assuming the inherited 500-update critic warmup, each 5C refinement job performs
20,500 critic updates and each 10C job 40,500. Across the six jobs this adds
**24,000 G updates and 184,500 C updates**, with larger critics, before final
training. These are counts, not measured FLOPs or GPU timings. If selected for
the final 50,000-G-update job, 5C means 250,500 C updates and 10C means 500,500.
The worker records critic parameter/update counts; final reporting must retain
both and the extra validation-search work.

This is a validation-informed, prospectively declared pre-test expansion of
MGM tuning. It is not identical hyperparameter-search budget or equal compute
across methods. With old candidates retained, MGM has 3 sphere candidates and
4 candidates per torus geometry, versus KGDM 3, native RMF/core-MT 2, RW 4.
Inherited noise selection and the additional critic capacity must be disclosed.
The README's unconditional description of five critic updates should become
conditional on the selected recipe before final reporting.

The final cap remains 50,000 G updates, seeds 0/1/2, nine datasets, four methods
(108 final jobs); no budget extension is proposed here. RMF is the small
source-derived native time network, not the common point generator, and has a
different parameter count. KGDM's torus field and periodic RMF features remain
documented adaptations. Equal G-update caps do not imply equal trainable
capacity, optimization effort, convergence, or full author reproduction.

## Initial Source Fingerprints

The reproduced worker was SHA256
`f8578bc082407165789e59e87b7dd2f400f49fd5619eedf83c22a9f2abf4d7da`.
`run_jobs.py` was
`5528b31161276783e2c3e7f82ebe6f20dbcbf427b1e37aabc79d8aa011c73510`.
Selection was re-read after the parent's addition of the third required phase;
the stale-candidate reproduction was updated to include that phase. Parent
edits to reporting code during this audit were preserved and the aggregation
check was rerun against its updated pure functions.
