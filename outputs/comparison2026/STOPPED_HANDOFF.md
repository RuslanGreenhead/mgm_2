# Explicit User Pause

STATUS UPDATE 2026-10-26: the user explicitly resumed this work ("finish it;
GPU is free"). The pause below is historical. The preserved calibration queue
has been restarted, with foreign GPU jobs still left untouched.

The user explicitly requested stopping our work and continuing later on
2026-10-24. Do not restart training, evaluation, agents, or GPU monitoring
without a new request from the user. No MGM automation was created.

Our GPU queue and all four subagents have been stopped/closed. A process
check after closing the last agents found no Python process from this project.
Unrelated ParetoSR jobs were not modified. The user's GPU policy before the
pause was to wait for ParetoSR to finish, not run alongside or stop it.

## Preserved Experimental State

- `mgm_calibration_v1`: all five validation-only jobs complete, 4000 generator
  updates each, plus 500 critic warmup and five critic updates per G step.
- `integration_v1`: all nine 100-update GPU contract checks complete for
  KGDM, RW-Flow and Riemannian MeanFlow on fire/general/RNA.
- `calibration_v2`: seven of 27 validation-only jobs complete. The eighth,
  `fire_rwflow_c2_s0`, has `last.pt` at 1000 updates. Its queue was stopped
  while waiting for the other project's GPU training. Restarting the same
  queue command resumes its frozen code, optimizer and independent RNGs.
- Some calibration training times are contaminated by ParetoSR contention
  after approximately 11:41 Moscow. Do not use those times as clean method
  speed comparisons. Numerical scores remain validation, not final test.
- No final configuration was selected and no final test comparison was run.
- The intended, not-yet-executed final protocol is nine real datasets, four
  methods, seeds 0/1/2, 50000 generator updates, separate MGM critic counts,
  validation checkpoint selection and final-only test scoring.

## Implementation and Remaining Checks

New code lives in `paper/benchmarks2026/`. The main paper and previous
controlled results were not overwritten.

KGDM uses the pinned author's spherical field; its torus field is an adaptation.
RW-Flow-C is an equation-checked reimplementation, not verified author code.
Riemannian MeanFlow uses a source-derived native time network/loss and optional
PCGrad, with explicit periodic-input/small-network adaptations. MGM/KGDM/RWFlow
share an unfitted neural generator. RMF has more parameters; report this and
training cost rather than claiming compute matching or author reproduction.

CPU tests run before the pause: common 4, KGDM 20, MeanFlow 13, RWFlow 22,
metric checks 4. All passed. Additional agent plot/grasp checks may have run,
but inspect their logs before claiming them. The optional new RWFlow scaling
and scaling64 solvers passed CPU finite-L equivalence tests; they still need
the prepared GPU `solver_probe` after the GPU is free. Live sources changed
after old phases were frozen, intentionally; never alter frozen snapshots.

`plots.py` and `grasp/` were being finished when their agents were closed.
Review current files/tests before considering these final. Several preview
PNGs already exist under `visualizations/calibration_v2/`; they are incomplete
validation-only previews, not final three-seed test results.

SE(3): two real, public, licensed ACRONYM H5/mesh pairs were fetched. User has
no local full ACRONYM/ShapeNet data. Full unseen-object benchmarking remains
unavailable. A within-object held-out-pose pilot for the two public objects
was being added; verify split disjointness and revised method labels/tests.
Author-style static collision/overlap is not physical grasp success. No
substantial GPU grasp pilot has been run; old 30-step CPU smoke is integration
evidence only. Inspect `grasp/cli.py` and its notes for current commands.

## Resume Only After User Asks

Use `.conda/rmgm/python.exe` in the project root. First inspect current processes
and run `-m paper.benchmarks2026.gpu_guard`; leave foreign jobs untouched.
Review/test final live plotting and grasp additions, then resume the preserved
calibration with:

```powershell
.\.conda\rmgm\python.exe -u -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/calibration_v2.json
```

After calibration and a free GPU: `solver_probe`, `select_configs`,
`make_comparison_plan final_v1`, `run_jobs` on that plan, `audit`, then `plots`.
Never silently overwrite `selected_config_v1.json` if one exists on resume.
Inspect grasp's current pilot command before starting it. Supply real final
metrics/figures only after the actual jobs finish; otherwise report the exact
partial status. No external-project processes should be terminated.
