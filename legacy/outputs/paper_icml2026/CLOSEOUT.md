# Empirical Sidecar Closeout

GPU ownership released at **2026-10-23 20:10:47 UTC**. Both training processes
exited with code 0. No additional GPU job or plotting task is queued here.
The optional true-velocity noisy-torus diagnostics belong to the parent.

## Completed Queue

- Independent rerun: 54/54 jobs completed, 18 unaffected MLP jobs copied with
  explicit unchanged provenance. All-nine fixed-budget protocol preserved.
- Primary training ran 19:46:24 to 20:03:25 UTC; summed synchronized training
  time 906.91947 seconds. Contention was observed, so these are not isolated
  hardware timings and do not support speedup claims.
- Sphere correction: 468 primary evaluation rows in 54 JSON files, plus 24
  calibration rows. All NPZ arrays and MMD values unchanged; v1 JSONs archived.
- Verification: 74 frozen files, 21 executed modules, 108 saved-RNG checkpoints
  and evaluation-invariance checks, 216 RNG groups distinct across three seeds.
- Long follow-ons: 2/2 completed at 10000 G / 50500 C updates each, seed 0,
  periodic critic, batch 512, G/C LR 2e-4, 500 frozen-G critic warmup steps.
  Verified 16 checkpoints, 96 train/validation rows and zero test evaluations.
- All 10 CPU metric/protocol regression tests pass.

## Long Validation Results

| Random initialization | Initial MMD2 | Fixed-final MMD2 | SD over 3 MC draws | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.750335144 | 0.034906396 | 0.001594483 | 164.287386 |
| GMM | 0.203861373 | 0.016395038 | 0.001088358 | 171.038559 |

These are single-training-seed exploratory trials, not seed-level error bars.
Both improve random initialization, but neither beats the earlier periodic
RFM validation MMD2 0.001909495 at 22500 updates. Budgets differ; this is not
an equal-compute superiority comparison. All trials and intermediate probes
remain available. The long-trial selection rule uses the fixed-final step
10000, despite stale inherited base-config prose mentioning 2000.

## Authoritative Outputs And Interface

All paths below are relative to this file's folder.

- `GPU_RELEASED.json`: ownership release and verification hashes.
- `controlled_independent/FINAL_TABLES_READY.json`, `verification.json`:
  primary readiness and passing checks.
- `controlled_independent/results.json`: flat list, 1080 evaluation rows.
- `controlled_independent/summary.json`: grouped list. Each metric contains
  `mean`, `sd_across_training_seeds`, `per_seed_mc_means`; MC draws are averaged
  within each seed first, then sample SD is computed across the 3 training seeds.
- `controlled_independent/controlled/<data>/<method>/seed<S>/evaluation_step<T>.json`:
  per-case row list with `data`, `method`, `training_seed`, `step`, `split`,
  `draw`, `sampler_nfe`, `vector_field_nfe`, `generated_n`, `target_n`,
  `mmd2_biased`, `symmetric_nn`, `coverage` and timing fields. Sphere rows also
  contain `metric_version` and `coverage_radius`.
- Same case folder: `samples_step<T>_nfe<N>.npz` with
  `generated_draw_0`, `generated_draw_1`, `generated_draw_2`; `step<T>.pt` saves
  model, optimizer and six RNG states for rerun jobs. Final steps are MGM 2000,
  RFM 10000; GMM step0 is retained. RFM128 uses 128 actual field calls, not 256.
- `controlled_independent/datasets/<data>/split_indices_and_targets.npz`:
  fixed split indices, targets and train-only kernel reference; matching
  `manifest.json` stores split counts, data seed and kernel specification.
- `controlled_independent/{provenance,frozen_sources,executed_source_hashes}.json`:
  copied/rerun mapping and source hashes. The original controlled bundle and
  both frozen source snapshots remain preserved.
- `controlled_independent/metric_provenance.json`, `metric_source_snapshot/`:
  version **sphere_normalized_float64_atan2_v2**, corrected JSON/NPZ hashes,
  radius changes and metric source hashes. Normalize float64 vectors, use
  `2*atan2(norm(xhat-yhat), norm(xhat+yhat))`; exclude identical reference
  directions explicitly. Volcano radius is 0.010018575232836861 radians.
- `controlled_independent/sphere_metric_v1_raw.zip`: pre-correction primary JSON.
  `sphere_metric_v1_calibration.zip`: original calibration JSON.
- `controlled_independent/calibration_{results,summary,metric_provenance}.json`:
  corrected empirical-resampling reference. Its draws are not training seeds.
- `stabilization_general_long/{results,selection,verification}.json` and trial
  subfolders: all long-trial metrics, learning curves, NPZs and checkpoints.
- `stabilization_general/periodic_rfm/completed.json`: stronger periodic RFM
  validation baseline. `legacy_audit.json` and `../../paper/experiments/AUDIT.md`
  retain the exploratory legacy/data/mesh limitations.

## Exact Commands

Working directory: `E:\Codex Projects\Riemmanian Midpoint Generative Models`.
Python environment is `.conda/rmgm/python.exe`; add its `Library/bin` to PATH.
The commands below describe the completed run, not a request to rerun training.
Do not call `--prepare` over this existing output bundle.

```powershell
$env:PATH="E:\Codex Projects\Riemmanian Midpoint Generative Models\.conda\rmgm\Library\bin;$env:PATH"
& '.conda/rmgm/python.exe' -B 'outputs/paper_icml2026/controlled_independent/source_snapshot/paper/experiments/independent_suite.py' --worker --root 'E:\Codex Projects\Riemmanian Midpoint Generative Models' 2>&1 | Tee-Object -FilePath 'outputs/paper_icml2026/controlled_independent/train.log'
& '.conda/rmgm/python.exe' -B 'outputs/paper_icml2026/controlled_independent/source_snapshot/paper/experiments/stabilization_long.py' --root 'E:\Codex Projects\Riemmanian Midpoint Generative Models' --output-root 'E:\Codex Projects\Riemmanian Midpoint Generative Models\outputs\paper_icml2026\stabilization_general_long' 2>&1 | Tee-Object -FilePath 'outputs/paper_icml2026/stabilization_long.log'
```

CPU-only correction/verification, safe to run separately from training:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:PATH="E:\Codex Projects\Riemmanian Midpoint Generative Models\.conda\rmgm\Library\bin;$env:PATH"
& '.conda/rmgm/python.exe' -B 'paper/experiments/correct_sphere_metrics.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_independent.py'
& '.conda/rmgm/python.exe' -B 'paper/experiments/verify_empirical_closeout.py'
```

The correction is idempotent. The Windows ZIP-member path normalization was
repaired as metadata only and recorded in `metric_provenance.json`. Existing
test exposure and exploratory legacy selection are disclosed in the audit;
the RNG rerun changes measurement integrity, not fixed hyperparameters in
response to test results. No method-superiority or official RFM-budget claim.
