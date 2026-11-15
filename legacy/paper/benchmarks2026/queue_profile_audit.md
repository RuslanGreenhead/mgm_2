# Queue and Isolated Profile Regression Audit

## Result

2026-10-26, CPU-only, `.conda/rmgm/python.exe`, one Torch/OMP/MKL/OpenBLAS
thread, `CUDA_VISIBLE_DEVICES=''`:

- After the parent's runtime-cleanup fix: expanded suite **32/32 PASS**
  (2.271 seconds), including all 14 existing worker-protocol tests and the
  post-spawn runtime-publication failure regression.
- Five forced failure completion orders: **5/5 PASS**, 0 errors (1.090 seconds).
  Orders: delayed consumer, failure first, cancellations first, reverse-order
  cancellations first, and immediate consumer. Original failure remains job 0.
- Full benchmark CPU discovery: **127 PASS, 2 CUDA-only SKIP**, 0 failures/errors
  (129 discovered tests, 7.238 seconds). Discovery used `test_*.py` in
  `paper/benchmarks2026` with the repository as top level. The separately owned
  SE3/real-data `grasp/tests.py` suite was not included.
- Interruptible guard waiting and already-set stop events both PASS. The queue's
  guard mock already accepts `stop_event=`; no compatibility workaround was needed.
- Neither queue regression is skipped or marked expected. Both prior findings
  are resolved in the fingerprinted source; no remaining finding from this recheck.
- All nine source/test SHA-256 values below were identical before and after the
  expanded suite and five-order run. All 30 top-level benchmark Python sources
  were additionally stable before/after full CPU discovery. The nine reported
  hashes match across both verification runs.
- Only new `test_queue_profile.py` and this audit were authored. No parent
  implementation, report, README, plan, dataset, or experiment output was edited.

## Resolved Runtime Cleanup

The previous P2 runtime-publication finding is fixed. The cleanup scope now
starts before `Popen`, tracks whether a child was actually spawned, aborts on a
post-spawn exception, waits for the owned child before exiting, and removes its
active runtime PID.

Passing regression:
`QueueTests.test_runtime_publication_failure_after_spawn_still_waits_for_child_exit`.

Fake `Popen` returns PID 42000 and the first nonempty runtime publication raises
`OSError`. The queue remains blocked until the fake child is released, preserves
the original `OSError`, clears every active fake process and runtime PID, and
does not publish phase completion.

The owned test's handshake now waits for its injected I/O error rather than
`queue_failure.json`: correctly repaired code waits for the child before that
future completes, so the earlier handshake would deadlock the fixture. The
failure injection and cleanup assertions were retained, not weakened. No new
test scope was added in this final recheck.

## Resolved Race

The original P1 queue race is fixed at the fingerprinted revision. Worker-side
abort publication plus the shared abort/spawn lock prevent queued job 2 from
launching after job 0 fails. All five forced completion orders preserve job 0's
failure attribution, wait for active job 1, clear runtime worker PIDs, and never
publish phase completion. Cancelled tasks cannot replace the original failure.

## Passing Coverage

- Actual `ThreadPoolExecutor` threads with gated fake `Popen`/`wait`: maximum
  two simultaneously active processes; one-worker frozen plans remain serial.
- Completed jobs are skipped on partial resume; a fully completed queue launches
  no additional workers and preserves its frozen protocol.
- Runtime records carry the queue PID, worker PID, original job, start time,
  concurrency bound, and frozen worker path; active records are removed on exit.
- Snapshot layout preserves repository-relative paths, notes, and KGDM author
  source/license. Commands use the snapshot worker, frozen job JSON, and root.
  Snapshot tampering, plan drift, and unvalidated concurrency are rejected.
- Phase timing explicitly labels per-job wall time as potentially contended and
  not an isolated efficiency comparison.
- `profile_all` rejects missing/incomplete phase completion, other GPU Python
  workers including this project, source tampering, stale checkpoint hashes,
  and nonisolated prior profiles. Child failure does not publish profile completion.
- Driver profiling is serial, uses frozen worker/job paths and `--latency-only`,
  and resumes only existing isolated profiles matching the selected checkpoint.
- The latency worker requires completed training and no other GPU Python worker;
  selected checkpoint identity mismatch is rejected before sampling.
- A real tiny CPU MGM performs 35 synthetic sampling draws: five warmups and 30
  timed draws, 1,024 samples per draw, exactly one model forward per draw.
  Generator, critic, nonempty optimizer states, and iteration remain identical.
  All preexisting checkpoint and training-artifact bytes remain identical.
- Latency-only `np.load`, training updates, metric evaluation, and `torch.save`
  are fail-fast traps and never called. Only `best.pt` is loaded. Sampling
  restores CPU RNG and removes hooks.
- The GPU guard is tested with mocked `nvidia-smi` output: project workers are
  included for isolation, self is excluded, and waiting never modifies workers.
  `stop_event.wait(60)` is used instead of uninterruptible sleep; cancellation
  raises `CancelledError`, and an already-set event avoids even the GPU query.

## Source Fingerprints

SHA-256, paths relative to `paper/benchmarks2026`, unchanged across final tests:

| File | SHA-256 |
| --- | --- |
| `run_jobs.py` | `4cb7c764362f79dca9fcedd82a1db117aadc4403d43821b4bb970d463e11cf95` |
| `gpu_guard.py` | `7bc721c90cdd4c62f03f11f6de5d3c55a7daf4d570ca8823864132b357a2b549` |
| `worker.py` | `603bd567db5767008e175396493ea016077c3fbed1b3c58ceeae6e61e15719a1` |
| `profile_all.py` | `9559c6ce880f8fef17659a3f1173431799ddf8fac836f6c47d15b9a330938648` |
| `common.py` | `8b77d30bc3f9ff87997e9f6485a92729ceee96573ad34c4407cc34ee598ed8ee` |
| `mgm.py` | `a9505f5bb662fb4027611e4ba5ce30df7f7348b26e04d92e32403d71525daabb` |
| `data_metrics.py` | `4a78fde200b02515cda4a3a48d1b5fdbbd62e24661658c85bcc82361b9bd6eae` |
| `test_worker_protocol.py` | `2d2e3ec245376091b5375da4b686ddd83f5f62378f608916f812c267db491b82` |
| `test_queue_profile.py` | `4644a5b6c500bb3d0537e194b8b0b3b14928d2459cc0f6d37d43c42aa6382f34` |

## Scope and Caveats

All inputs live in temporary synthetic fixtures. CUDA initialization and real
`subprocess.Popen`/`run` are fail-fast traps. The fake CUDA facade redirects the
worker device and RNG context to CPU; no GPU or real held-out data was used.
No descendants were created. These tests do not measure GPU speed or reproduce
the parent's throughput experiment.

`worker.py:61` still hashes the complete `data.npz` byte stream before the
latency-only return. The passing guarantee is **no materialization or access of
train/validation/test arrays**, not zero archive I/O. If zero archive reads are
required too, move the archive hash check into the training branch while keeping
manifest/checkpoint provenance checks for profiling.

The parent's reported 2.12x serial-versus-two-process result with bitwise-identical
500-step MGM generator/critic states is resource scheduling evidence, not a loss
change. Training wall times under contention must retain their label. Publish
inference speed only from isolated serial profiles after all training has exited;
this audit does not turn the throughput probe into an inference-speed claim.

## Reproduce

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:RWFLOW_GRAPH_CUDA_TESTS='0'
& ./.conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_queue_profile -v
& ./.conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_queue_profile paper.benchmarks2026.test_worker_protocol -v
& ./.conda/rmgm/python.exe -c "import sys,unittest; loader=unittest.defaultTestLoader; suite=unittest.TestSuite(); name='paper.benchmarks2026.test_queue_profile.QueueTests.test_failure_cancels_queued_and_waits_active_exit_without_phase_completion'; orders=('delayed','failure_first','cancel_first','reverse_cancel_first','immediate'); [(setattr(case,'failure_order',order),suite.addTest(case)) for order in orders for case in loader.loadTestsFromName(name)]; result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"
& ./.conda/rmgm/python.exe -B -c "import sys,subprocess,unittest,torch; from contextlib import ExitStack; from unittest.mock import patch; torch.set_num_threads(1); stack=ExitStack(); stack.enter_context(patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden'))); stack.enter_context(patch.object(torch.cuda,'synchronize',side_effect=AssertionError('CUDA forbidden'))); stack.enter_context(patch.object(subprocess,'Popen',side_effect=AssertionError('Real process forbidden'))); stack.enter_context(patch.object(subprocess,'run',side_effect=AssertionError('Real process forbidden'))); suite=unittest.defaultTestLoader.discover('paper/benchmarks2026',pattern='test_*.py',top_level_dir='.'); result=unittest.TextTestRunner(verbosity=2).run(suite); stack.close(); sys.exit(not result.wasSuccessful())"
```

Bounded recheck complete: all requested CPU regressions are green, including the
bookkeeping-failure regression and five completion orders. This is CPU protocol
verification, not a GPU performance claim or authorization to access held-out data.
