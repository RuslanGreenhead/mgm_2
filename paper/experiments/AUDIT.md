# Empirical Audit

Scope: local evidence, not an external certification of the author's released
artifacts. Audit date 2026-10-23. No existing trainer, manuscript, or figure was
edited. New experiments and generated artifacts are isolated in the two
authorized directories. `outputs/paper_icml2026/legacy_audit.json` contains
checksums, exact checkpoint paths/configs, counts, and compatibility checks.

## What The Existing Results Support

1. **Test-selection leakage.** `external/riemannian-fm/rmgm_train_real.py:160`
   sets training from train; line 161 sets evaluation from test, not validation.
   Lines 286 and 384 select best.pt by test symmetric NN. Validation is loaded
   at line 988 but never used for selection. `rmgm_summarize_all.py:84` takes
   metric-wise extrema over evaluation steps; line 125 then chooses runs by
   their minimum test MMD. Its final-step columns still inherit test-based run
   selection. Different best metrics need not describe one checkpoint.
2. **Data identity has three levels.** The four Earth CSVs are byte-identical
   to entries in the existing local `data/data.zip`; hashes are recorded.
   Top500/RNA PDB ID lists also match that archive, but angles are locally
   regenerated using downloaded structures (`rmgm_prepare_torus_data.py:35`,
   `:42`, `:54`). Exact identity to the author's processed angle arrays is
   unproven. Bunny/Spot/Maze are locally synthesized from the author recipe,
   not verified identical author sample arrays. Eigenfunction sampling seeds
   NumPy but not Torch (`rmgm_prepare_mesh_data.py:120`, `:129`); eigenvector
   signs are also not canonicalized. Archive identity is not independently
   authenticated against a remote checksum in this audit.
3. **Split membership.** For fixed arrays, `rmgm_train_real.py:1032` matches
   `manifm/datasets.py:445`: floor(N/10) validation and test, random_split with
   effective data_seed = seed if data_seed is null. All 64 audited legacy
   checkpoints have seed 0/effective data_seed 0. Fixed-index membership is
   therefore common between local MGM/RFM runs, not proof of the author's
   exact published processed data/split. Training sampling differs: legacy
   MGM/RFM draw with replacement (`rmgm_train_real.py:143`), whereas the author
   loader shuffles expanded training rows (`manifm/datasets.py:461`).
4. **Molecular generalization is not tested.** Source PDBs overlap the row
   splits: test PDBs also present in train are General 494/494, Glycine 370/370,
   Proline 311/314, Pre-Pro 299/302, RNA 98/98. These are residue-row density
   benchmarks, not held-out-protein/RNA-molecule tests. A group split would be
   a different benchmark and should be separately named.
5. **GMM initialization dominates attribution.** The trainer fits centers,
   scales and mixture counts from train before optimization
   (`rmgm_train_real.py:239`, `:802`, `:842`). Strong Earth heads have 24
   components (Volcano 12); Torus heads actually have General 64, others 32,
   not the often-displayed `components=4` from the summary's wrong field
   (`rmgm_summarize_all.py:102`). These GMM heads have no MLP
   (`manifm/rmgm/generators.py:198`, `:317`). `searchsorted` mixture choices
   at lines 218/349/556 have no pathwise gradient to component_logits, so
   fitted weights stay fixed; centers/scales train. This is confirmed by a
   regression test and saved parameter-change diagnostics. A frozen step-0
   GMM, random-init constrained MLP MGM, and same-MLP MMD control are needed.
6. **Not an author-budget reproduction.** Legacy Earth/Torus use 600/800
   steps and small critics. The author base config uses 1,000,000 steps
   (`configs/train.yaml:43`), with experiment-specific architectures/batches.
   Mesh/appendix configs use 50,000-100,000. No usable official full-budget
   checkpoint was found in `outputs/rfm_official_full`; its comparison table
   explicitly says none completed. A preflight test/loss is not a trained
   author-budget result. No exact MGM NLL is available.

## Counts And Final Checkpoints

Every row is total / train / validation / test. Counts are read from actual
configs and local arrays; seeds are 0 for the legacy inventory.

| Data | Counts | Legacy final steps | MGM run ID | RFM run ID |
|---|---|---:|---|---|
| volcano | 827 / 663 / 82 / 82 | 600 | seed0_20260801_001604 | seed0_20260801_001155 |
| earthquake | 6120 / 4896 / 612 / 612 | 600 | seed0_20260801_001611 | seed0_20260801_001202 |
| fire | 12809 / 10249 / 1280 / 1280 | 600 | seed0_20260801_001621 | seed0_20260801_001217 |
| flood | 4875 / 3901 / 487 / 487 | 600 | seed0_20260801_001616 | seed0_20260801_001209 |
| general | 138208 / 110568 / 13820 / 13820 | 800 | seed0_20260801_002726 | seed0_20260801_002732 |
| glycine | 13283 / 10627 / 1328 / 1328 | 800 | seed0_20260801_002735 | seed0_20260801_002739 |
| proline | 7634 / 6108 / 763 / 763 | 800 | seed0_20260801_002741 | seed0_20260801_002745 |
| prepro | 6910 / 5528 / 691 / 691 | 800 | seed0_20260801_002748 | seed0_20260801_002753 |
| rna | 9473 / 7579 / 947 / 947 | 800 | seed0_20260801_002840 | seed0_20260801_002845 |

Paths are `outputs/rmgm_real_earth/<data>/<method>/<run>/checkpoints/last.pt`
for Earth and `outputs/rmgm_real_torus/...` for Torus. These particular 18
checkpoints strict-load under current code. Four older Earth GMM last.pt files
(001157, 001211, 001204, 001149) lack component_logits and require an explicit,
documented legacy loader; do not silently strict=False them into the new model.

- All six Bunny/Spot datasets: 500000 / 400000 / 50000 / 50000, final step 800.
  All three heads' exact 18 paths are in legacy_audit.json under
  `outputs/rmgm_mesh_bunny_spot_pilot/{projection,sphere_map,sphere_map_gmm}/`.
  They are suitable for final-step reevaluation on the existing split, not
  a new untouched-test claim. The original mesh comparison chooses best.pt
  first (`rmgm_mesh_compare.py:110`) and ignores checkpoint data_seed while
  rebuilding the target (`:124`); that only happens to match these seed-0 runs.
- Wrapped Torus: 200000 / 160000 / 20000 / 20000, 2000 steps for d5 and
  1000 for d10/d20. Wrapped SPD: 10000 / 8000 / 1000 / 1000, 2000 steps for
  d3 and 1000 for d6/d10. Hyperbolic: 20000 / 16000 / 2000 / 2000, 2000 steps.
  These datasets ignore row index and sample dynamically
  (`manifm/datasets.py:240`, `:336`); the real trainer materializes fresh arrays.
  They share a recipe, not finite author train/test samples. Across new model
  seeds, reusing data_seed alone would not freeze their realized arrays.
- Maze3v2/Maze4v2: 100000 / 80000 / 10000 / 10000, 500 steps. Simple Bunny:
  200000 / 160000 / 20000 / 20000, 500 steps. For paired targets the real
  trainer retains x1 only (`rmgm_train_real.py:1049`) and does not retain the
  author's x0 base distribution, so it is not the same transport task.
- All 46 final-status last.pt and 18 Bunny/Spot last.pt files load as tensor
  dictionaries, have finite tensors, and saved step equals requested step.
  This audit did not train or reevaluate the entire standalone toy inventory.
  `rmgm_train.py:101` constructs hand-set toy targets, not author real data.

## Geometry And Cost

- Simple sphere/torus paths use expmap/logmap and JVP
  (`manifm/rmgm/geometry.py:179`); SPD has its own geodesic/JVP branch at :152.
  Hyperbolic velocity uses a projected finite difference with epsilon 0.001
  (:223), not exact JVP. Losses freeze generated samples for critic regression
  (`losses.py:41`) and differentiate through the path for the generator (:73).
- The shared tangent critic/RFM field concatenates raw coordinates and time
  features (`manifm/rmgm/critics.py:94`). Torus angles are not converted to
  circular spatial features, so the network is not seam-periodic even though
  the metric and MMD embedding are. This is preserved and disclosed; the
  supplement does not claim an architecture-optimized torus baseline.
- Mesh uses P(x0+s(x1-x0)) and P_T(chord) (`geometry.py:200`), not the author's
  `Mesh.solve_path` (`manifm/manifolds/mesh.py:163`). Lightweight RFM routes to
  the same surrogate (`rfm_baseline_train.py:291`). At face interiors these
  agree with a local derivative, but edge clamping/switching need not.
  legacy_audit.json saves an explicit two-triangle edge counterexample.
  This is not a proof of intrinsic geodesic or global path correctness.
- Critic tangency is correctly implemented: `critics.py:95` calls proju;
  `mesh.py:116` removes a unit face normal (`:344`). It is only face-tangent
  on a piecewise-flat mesh; edges/vertices have no unique face tangent. The
  supplement records normal-residual probes, not just projection feasibility.
- The trained sphere-map heads use soft angular radius interpolation followed
  by closest-point projection (`generators.py:465`, `:567`). They are not the
  ray-intersection utility and are not certified bijections. Genus zero alone
  is insufficient; non-star-shaped concavities can fold or hide patches.
  Feasibility near 1e-9 does not imply coverage or an inverse map.
- Original MMD truncates to 1024 and fits its kernel to generated+target samples
  (`rmgm_train.py:373`); coverage radius changes with each test draw
  (`rmgm_train_real.py:500`). Most primary legacy numbers use resampled targets
  with replacement and one draw. Mesh/SPD/hyperbolic NN falls back to ambient
  Euclidean distance (`rmgm_train_real.py:562`), not intrinsic geometry.
- Legacy RFM sampling is Euler with midpoint time, one field call per step
  (`rfm_baseline_train.py:267`): 64/128 steps means 64/128 NFE, not 128/256.
  GMM sampling may use zero neural evaluations, so 'one NFE' is not a fair
  network-cost comparison. Mesh closest-point projection is brute force
  O(batch*faces) (`mesh.py:372`); sphere-map soft lookup is O(batch*vertices).
  MGM also uses generator+critic updates, path differentiation, train-data
  preprocessing and GMM fitting. The new results record these costs explicitly.

## Publication Boundary

Primary evidence should be the locked three-seed, final-step supplement with
step-0 GMM and parameterization-matched controls. Old best-test-selected tables
are exploratory appendix evidence only. New use of the historically inspected
test split does not make it pristine. No result here supports 'beats official
RFM', 'beats one-step manifold SOTA', exact likelihood, generic mesh bijection,
or intrinsic mesh-geodesic correctness. Published one-step competitors require
separate faithful reproductions; the MMD control is not one of them.
The primary supplement also discloses a CPU-timing RNG side effect: distinct
initialization/target/latent seeds, but a partly shared hidden-time/base-noise
schedule after fixed evaluation checkpoints. This is not test-score selection;
it limits interpreting the SD as variation over fully independent random-stream
bundles. The original code is archived; the extension's timing helper is fixed.

Reproducible commands and metric definitions are in README.md. Raw results,
seed-level summaries, model states, NPZ draws and index hashes are in
`outputs/paper_icml2026/`. No figure or manuscript claim is automatically edited.
