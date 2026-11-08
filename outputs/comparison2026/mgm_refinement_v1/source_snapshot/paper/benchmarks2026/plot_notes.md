# Comparison Plot Utility

`plots.py` is a standalone CPU-only reader. It imports NumPy, SciPy, and
Matplotlib, not torch, method adapters, author runners, checkpoints, or the
legacy result-generation entry point. It does not train, sample models,
compute new evaluation metrics, or select settings using test results.

## Commands

From the workspace root, final comparison after completed final test scoring:

```powershell
.conda/rmgm/python.exe -B -m paper.benchmarks2026.plots --phase outputs/comparison2026/final_v1 --split test --require-complete
```

Early Fire/General/RNA validation preview, merging completed MGM calibration jobs:

```powershell
.conda/rmgm/python.exe -B -m paper.benchmarks2026.plots --phase outputs/comparison2026/calibration_v2 --extra-phase outputs/comparison2026/mgm_calibration_v1 --split val --datasets fire general rna
```

`--phase` also accepts multiple paths. The first phase names the output folder.
Without `--split`, only all-final_* inputs default to test; other inputs default
to validation. Explicit `--split val` remains available for a final phase.

```powershell
.conda/rmgm/python.exe -B -m paper.benchmarks2026.plots --self-test
```

## Eligibility and Selection

- A run is eligible only when BOTH its `job.json` and `completed.json` exist.
  Root phase completion summaries are not mistaken for individual jobs.
- The completion's identity, declared final budget, and selected checkpoint
  range must match the job. No metrics or samples from unfinished jobs are read.
- Validation previews can merge phases containing several configurations per
  dataset/method/seed. Choose the completed candidate with minimum recorded
  `selected_validation_mmd2`, with lexical path as deterministic tie-breaker.
  This is a post-selection validation preview, not independent evidence.
  All completed candidates, including unselected variants, are retained in
  separate `calibration_candidates_*` CSV/Markdown tables.
- Test mode is restricted to final_* phases, explicit `score_test=True`, and
  completed test evaluation. Non-final test-enabled jobs are rejected even in
  validation mode. Duplicate final jobs for a dataset/method/seed are errors;
  all available final seeds for a dataset/method must share one frozen config,
  resolved defaults, declared budget, and executed worker/method source hashes.
- `--require-complete` requires all four methods and seeds 0,1,2 for every
  requested dataset. Without it, missing cells are explicitly shown or recorded,
  never filled using another seed, phase, checkpoint, or fabricated result.
- The exact selected checkpoint is read, not the latest or visually best one:
  `{split}_step{selected_step}_metrics.json` and matching `_samples.npz`.
- Figures use training seed 0 and `generated_draw_0` only. Metrics average all
  recorded draws within a training seed first, then average across available
  training seeds 0,1,2 within an identical phase/config identity; sample SD uses
  ddof=1. With one seed, training-seed SD is absent, not 0. Per-seed records also
  report separately named within-seed draw SD, absent when there is one draw.
  Seed-wise selected configurations are never pooled into one method mean.
- A full SHA256 config ID covers raw/resolved config, method, update budget and
  executed source hashes, never the training seed. Tables retain phase, variant,
  recipe/run path, raw config and resolved defaults. Figures display a compact
  ID; the full identity and source hashes are in CSVs and the plot manifest.
- Held-out target arrays come from the parent's existing fixed
  `outputs/comparison2026/data/<dataset>/data.npz`. NPZ loading is lazy:
  validation mode indexes ONLY `val_target`. It never extracts or hashes the
  `test_target` member or scores any held-out points itself.

## Scientific Display

Columns are Held-out target, MGM, RWFlow, KGDM, MeanFlow. Every available panel,
including the target, is a density plot with the identical kernel and bandwidth.
Nothing mixes an unsmoothed target scatter with smoothed generated samples.
Each row has ONE logarithmic color scale, from 0.001 times its maximum density
to its maximum, pooling the held-out target and all available method panels.
Different rows are not assigned a supposedly universal density scale.

The kernel conventions are taken from `paper/update_results.py`'s `kde` helper,
which is read but neither imported nor edited:

- Sphere: normalized von Mises-Fisher KDE on unit XYZ, kappa=45, density per
  steradian. Evaluate a 241x121 longitude/latitude grid. Render Mollweide maps
  and the existing Natural Earth 110m land outlines, split at the dateline.
  The KDE itself is spherical; map projection is only a display transform.
- Torus: a 120x120 histogram over wrapped angles in [-pi,pi), smoothed with
  Gaussian sigma=2.4 bins in `mode='wrap'`, density per radian squared. The
  displayed axes are degrees, and periodic padding prevents border seams.
- RNA7: six adjacent angle-pair marginals (1,2), (2,3), ..., (6,7), ensuring all
  seven angles appear. These are NOT full-joint density verification. The table
  retains the parent's full-dimensional RNA metrics without recomputing them.

No isolated absolute metric is presented as proof of convergence. Every
validation preview is prominently headed VALIDATION ONLY; individual panels
show selected update and full declared update budget. Tables identify missing
training seeds. Small validation sets, post-selection bias, and Monte Carlo
variation remain limitations of the input experiments.

## Outputs

All writes stay under
`outputs/comparison2026/visualizations/<first-phase-name>/<split>/`.
If that directory already exists, a new `<split>_002`, `_003`, ... revision is
created. Existing previews and core experiment artifacts are never overwritten.

- One dataset comparison PNG and vector PDF each; RNA contains six pair rows.
- Individual target/method PNGs and vector PDFs under `panels/`. Each is rendered
  natively from the exact cached comparison density array and the same row-wise
  log color limits; no cropping, image editing, resmoothing or independent
  renormalization is used. Examples: `panels/fire_mgm.png`,
  `panels/fire_target.png`, `panels/rna_pair6_7_mgm.png`. All six RNA pair
  marginals are exported separately for each available method and target.
- `panel_index.md` links every dataset comparison and individual zoom export.
- An overview PNG and vector PDF, with four sphere rows, four 2D torus rows,
  and six RNA pair rows when the full set is available. Only datasets with
  completed seed-0 figures are rendered in a partial preview.
  When there are more than six rows, additional sphere/T2/RNA overview sheets
  keep each geometry readable without relying on a very tall overview alone.
- PNG width is 4800 px at the default 240 dpi; minimum allowed dpi is 200,
  guaranteeing comparisons, overviews and curves >=4000 px wide. Individual
  zoom PNGs are 2880 px wide by default, 2400 px at minimum dpi (>=2000 px).
- Density layers, coastline paths, ticks, and text remain vector in PDFs.
  Earth uses a vector QuadMesh; periodic torus densities use vector contours.
- Validation learning curves against generator updates, one line per method
  and training seed, with symlog scale and a 1e-5 linear region.
- Method-specific training-loss panels with independent y-scales. Losses have
  different units and are explicitly not a cross-method ranking.
- Per-seed metrics CSV, seed-mean metrics CSV and Markdown, and per-dataset
  Markdown tables containing each seed 0/1/2 plus mean and training-seed SD for
  every reported metric. Missing values remain `--`/empty CSV cells.
  The primary biased `kernel_mmd2_v` and the directed nearest-neighbor
  `minimum_matching_distance` have distinct names in both CSV and Markdown.
- `plot_manifest.json` records phase inputs, exact selected runs/configs,
  missing cells, source JSON hashes, accessed NPZ keys/array hashes, row scales,
  plotting constants, script hash, and exported figure paths.

Learning curves use only completed selected runs and their saved validation
checkpoints. They do not read intermediate test metrics. Raw training loss is
read from those runs' existing `curve.json` and not rescaled into a common loss.
MGM critic-only warmup entries are excluded from generator-loss curves.
Training seeds have distinct line styles; curves and CSVs retain config IDs.

## Verification and Limitations

The 24 plotting tests exercise wrapped KDE normalization and periodicity,
spherical area integration and radial normalization, separate draw/seed SD,
variant/config/phase aggregation boundaries, validation selection, final
source/config rejection, incomplete-job guards before file access, metric/NFE
consistency, warmup filtering, PNG dimensions, shared scales, individual RNA
exports, uncropped zoom captions, and separation of comparison headings,
panel titles and footer labels. Visual QA uses real completed validation
runs only; no fake result fixture is exported. PDF QA uses Poppler rendering
and checks that the PDF has no embedded density raster images.

The first exploratory renderer revision used projected contour paths for Earth,
which produced unnecessarily large PDFs; it is preserved as an earlier output
revision, not the recommended deliverable. The corrected vector mesh renderer
and wider colorbar margin are used by the current CLI.

No final test figure is claimed until the corresponding final jobs complete.
No GPU action is authorized or performed by this utility.

The original 2026-10-24 check covered seven tests and real completed Fire/RNA
validation layouts. On 2026-10-26 the resumed suite passes 24 plotting tests on
CPU with OMP/OpenBLAS/MKL/NumExpr limited to one thread. Import inspection confirms
that torch is not loaded. No training or GPU operation is performed.

Recommended current preview:
`outputs/comparison2026/visualizations/calibration_v2/val_006/`.
The discovery snapshot contains 26 completed calibration candidates, 10 selected
seed-0 runs and 26 explicitly missing dataset/method/seed cells across Fire,
General and RNA. Fire and General have all four methods; RNA has MGM and KGDM
only in this snapshot. It is not a completed final comparison, and later jobs
are not silently added to an existing revision.

QA passed for all 37 exported PNGs and 37 vector PDFs: comparisons are 4800 px
wide (single-row cases 4800x1320), all 28 individual zoom PNGs are 2880 px wide,
and RNA zooms are 2880x2736. All images are nonblank; each zoom's vmin/vmax
exactly equals its comparison row's recorded scale. Recursive pypdf resource
inspection finds zero embedded raster images in all PDFs. Poppler-rendered
Fire and RNA PDF panels were visually checked alongside actual comparison,
zoom and learning-curve PNGs. No clipped text or heading/footer overlap remains
in the corrected revision. `val_005` is preserved but superseded because its
single-row torus headings overlapped before the margin regression fix.

The manifest confirms that accessed NPZ members are only `val_target` and
`generated_draw_0`, with no test-step JSON or NPZ access. Calibration variants
RMF/RMF-MT remain distinct in the all-candidate tables. The final plotting
command above is ready for the future completed `final_v1` phase; no final test
figure or three-seed result is claimed here.
