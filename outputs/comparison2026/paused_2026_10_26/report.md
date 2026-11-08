# Paused Comparison: Intermediate Results

User requested GPU release. Training is stopped; this report uses only saved CPU-readable artifacts.

Completed 18/108 main jobs; interrupted 2; not started 88.
No training, sampling or GPU timing is launched by this report. No missing seed or method is imputed.

## Completed Test Scores

Biased embedded Gaussian kernel MMD squared x1000; lower is better. Mean +/- sample SD across training seeds after averaging three draws per seed.
Only cells with all three completed seeds receive an aggregate. Missing cells mean unavailable, not poor performance.

| Dataset | MGM | RW-Flow-C | KGDM | RMF-MT |
| --- | ---: | ---: | ---: | ---: |
| volcano | 13.135 +/- 1.555 | 12.537 +/- 0.277 | 13.900 +/- 1.694 | 11.957 +/- 0.275 |
| earthquake | not complete | not complete | 2.132 +/- 0.115 | 2.814 +/- 0.694 |
| flood | not complete | not complete | not complete | not complete |
| fire | not complete | not complete | not complete | not complete |
| general | not complete | not complete | not complete | not complete |
| glycine | not complete | not complete | not complete | not complete |
| proline | not complete | not complete | not complete | not complete |
| prepro | not complete | not complete | not complete | not complete |
| rna | not complete | not complete | not complete | not complete |

Only Volcano has a complete four-method test comparison at this pause. MGM does not have the smallest observed mean there. This is not a statistical-significance claim; Volcano has only 82 test points.

[Volcano comparison, 4800px](../visualizations/final_v1/test_002/volcano_comparison.png) | [Earthquake completed baselines; missing methods explicit](../visualizations/final_v1/test_002/earthquake_comparison.png) | [Individual high-resolution panels and vector PDFs](../visualizations/final_v1/test_002/panel_index.md)

## Interrupted Jobs

The table shows the latest durable progress/checkpoint, not the last in-flight optimizer update. Forced process termination can discard work after the last saved checkpoint.

| Dataset | Method | Seed | Saved / planned updates | Validation-selected step |
| --- | --- | ---: | ---: | ---: |
| earthquake | MGM | 0 | 40,000 / 50,000 | 30,000 |
| earthquake | MGM | 1 | 20,000 / 50,000 | 20,000 |

[Paused validation curves](paused_validation_curves.png) | [Earthquake validation preview](earthquake_paused_validation.png)

Earthquake preview is validation-selected, single-seed, and partly trained. It is deliberately separate from completed test scores; it must not support a completed-test or convergence claim.

## Completed SE(3) Pilot

The separate two-object Mug/Table pilot finished all nine fixed 10,000-update jobs and three seeds. It is a within-object held-out-pose pilot, not full ACRONYM or unseen-object generalization.

[SE(3) report and three-seed metrics](../grasp/report.md) | [Mug image](../grasp/publication_figures/seed0/mug_comparison.png) | [Table image](../grasp/publication_figures/seed0/table_comparison.png)

MGM improves pose MMD over the one-step RMF adaptation but does not beat 50/100-step RFM. Substantial full-gripper collision remains unresolved; no physical grasp success was measured.

## Scope and Verification

Small-network controlled adaptations, not original-author model sizes or budgets. Common point generators use width128/depth3; RMF has a different time-network parameter count. MGM also performs extra critic updates. Optimizer work, tuning grids and compute are not matched.
Saved historically exposed row splits are not a newly blinded or source-group-held-out benchmark. Calibration/refinement validation results are excluded from the test table.
All completed jobs were checked on CPU: frozen source/data hashes, update/critic counts, selected validation/test arithmetic, three unique draws, one model forward, manifold constraints and independent NumPy kernel-MMD recomputation.
GPU latency profiling was intentionally NOT run after the stop request. This is a partial-artifact audit, not the complete 108-job final audit.
