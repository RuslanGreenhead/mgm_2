# MGM and Recent One-Step Manifold Baselines

This directory is a separate controlled experiment. It does not replace the
previous paper's controlled suite or turn pilot experiments into reproductions.

## Scope and Fidelity

- Nine real datasets: volcano, earthquake, flood, fire, general, glycine,
  proline, prepro, and seven-angle RNA. The historically exposed row splits
  and train-only metric references are copied with checksums, not regenerated.
- MGM uses the geometric hidden-flip minimax objective, full observation and
  velocity differentiation, a uniform-prior neural generator, validation-selected
  five or ten critic updates per generator update, and a separately counted
  critic warmup. Larger-critic refinement keeps the point generator unchanged.
- RW-Flow-C is an equation-checked reimplementation using chordal costs and
  independent generated self banks. No verified public author code was found.
- KGDM executes a checksum-pinned author's spherical field. The wrapped-log
  torus version is an explicitly documented adaptation. Its EMA sampling
  model is restored with its selected checkpoint and counts as one NFE.
- Riemannian MeanFlow is the Zhong et al. method. Its native loss, time
  network, stopped JVP target, and optional PCGrad are checked against the
  unmodified release. Periodic torus features and small-network budgets are
  benchmark adaptations. Its parameter count differs from the common point
  generator used by the other three methods.
- Mesh pilots are not silently included: their projection-loss surrogate is
  not the same closed-form path objective. The SE(3) public-data pilot has a
  separate protocol and is not a full ACRONYM unseen-object benchmark.

See `RWFlow_NOTES.md`, `KGDM_NOTES.md`, `MEANFLOW_NOTES.md` and `grasp/` for
precise source revisions, equations, defaults, tests and limitations.

## Locked Workflow

Run from the project root with `.conda/rmgm/python.exe`:

```powershell
python -B -m unittest discover -s paper/benchmarks2026 -t . -p 'test_*.py'
python -m paper.benchmarks2026.data_metrics
python -m paper.benchmarks2026.make_mgm_plan
python -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/mgm_calibration_v1.json
python -m paper.benchmarks2026.make_comparison_plan calibration_v2
python -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/calibration_v2.json
python -m paper.benchmarks2026.make_mgm_refinement_plan
python -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/mgm_refinement_v1.json
python -m paper.benchmarks2026.solver_probe
python -m paper.benchmarks2026.throughput_probe
python -m paper.benchmarks2026.select_configs
python -m paper.benchmarks2026.make_comparison_plan final_v1
python -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/final_v1.json
python -m paper.benchmarks2026.profile_all --phase final_v1
python -m paper.benchmarks2026.audit --phase final_v1
python -m paper.benchmarks2026.report --phase final_v1
```

Each phase freezes all executed Python sources and the required pinned KGDM
field, checksums them, and runs a bounded process-isolated CUDA queue. The final
queue uses at most two workers after a train-only scheduling probe verified
bitwise-equal generator/critic weights and higher throughput. Completed jobs
are never overwritten. Interrupted jobs resume from the last validation
checkpoint with optimizer, method/EMA, and independent target/training RNGs.

Calibration reads no test targets for metric computation. Recipes are chosen
on fire, general, and RNA validation probes and transferred to the other
datasets of the same geometry. The final protocol is frozen before test
scoring: 50,000 generator updates, seeds 0/1/2, checkpoint selection by three
fixed validation draws, and three final test draws per selected checkpoint.
The initial checkpoint is eligible; reporting it prevents hiding degradation.
An evaluation assertion verifies it did not advance any training RNG.

## Metrics and Interpretation

The common primary score is **biased Gaussian embedded kernel MMD squared**,
using a fixed train-median bandwidth. Smaller is better; finite-sample values
need not be zero for a correct generator. This is not RW-Flow's metric called
minimum matching distance. Both are stored with distinct names. Additional
scores are intrinsic symmetric nearest-neighbor distance, directed minimum
matching distance, radius coverage, nearest-neighbor occupancy coverage, and
balanced leave-one-out 1-NNA (ideal near 0.5, not necessarily smaller).

Final tables average Monte Carlo draws within each training seed first, then
report mean and sample standard deviation across the three training seeds.
These deviations do not include uncertainty from alternative data splits.
No NLL is inferred for implicit generators. Constraint residuals and measured
one-call sampling latencies include the geometric output map. Training time
includes warmup but excludes validation; under concurrent scheduling it is a
contended wall time, not isolated per-method training speed. MGM critic updates
are also reported. Sampling latency is reprofiled serially only after all
training has completed.

Equal optimizer-update counts are not equal compute, equal parameter counts,
author training budgets, or a proof of convergence. Validation learning curves
and all configured budgets accompany any scientific comparison.

## Sources

- RW-Flow: https://arxiv.org/html/2609.39271v2
- KGDM: https://arxiv.org/html/2605.10727
- Riemannian MeanFlow: https://arxiv.org/html/2603.10718v3
- Riemannian Flow Matching: https://arxiv.org/abs/2302.03660
