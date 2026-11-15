# Real ACRONYM SE(3) Pilot

## Status And Scope

CUDA is locked until the parent explicitly releases the shared queue at the
main-phase boundary. Do not inspect, stop, or interfere with other processes.
No GPU job is launched by preparing data, writing the plan, or running CPU tests.

The pilot uses exactly the official NVIDIA mug/table mesh and HDF5 pairs, pinned
at `NVlabs/acronym@d02f58fb3a556c3b719976e206cce0dce0103199`.
The dataset license is CC BY-NC 4.0; the repository code is MIT. Raw asset URLs,
verified upstream Git blob hashes, and SHA256 hashes are in
`external/acronym-data/public/download_manifest.json`. Optional dependencies are
isolated in `external/acronym-data/python-deps`, leaving the shared environment
unchanged. No toy target poses or synthetic object meshes are substituted.

Full ShapeNetSem mesh access requires license approval. The author clone has an
additional 1Shelves H5/OBJ pair, but no explicit license grant was established
for that redistribution; it is not used. The upstream RFM-Grasp repository has
no data beyond a placeholder, and the author repository has no processed-cache
or checkpoint release. No full 1.6GB grasp archive was downloaded without meshes.

Primary sources:
[ACRONYM and license](https://github.com/NVlabs/acronym),
[RMF paper, Section 5.2](https://arxiv.org/html/2603.10718v3#S5.SS2),
[pinned author code](https://github.com/haolsun/Riemannian_MeanFlow/tree/45d96bcd09a271ea3a770b2014865dabbac9eeeb/grasp),
[ShapeNetSem license gate](https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive).

## Two Separate Protocols

`outputs/comparison2026/grasp/data/public_smoke/manifest.json` retains the
original object-transfer integration split: table train, mug test, no validation.
It is NOT a benchmark and is not the primary learned-density pilot.

`outputs/comparison2026/grasp/data/public_within_object/manifest.json` is the
primary **within-object held-out-pose pilot, NOT unseen-object ACRONYM**.
Split seed 2026 is fixed independently of model seeds. Per-object positive HDF5
source rows are permuted into 80/10/10 train/val/test partitions. All source row
indices, split hashes, raw mesh/H5 hashes and cache hashes are recorded.

| Real object | Source poses | Positive labels | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| Mug | 2000 | 1290 | 1032 | 129 | 129 |
| Table | 2000 | 1559 | 1247 | 155 | 157 |

Training samples objects uniformly and draws only their train poses. Validation
and test references use only their own source rows. The mesh/conditioning is
intentionally fixed and shared across pose partitions. Geometry normalization
uses mesh bounds, not held-out grasp statistics. H5 translations are already
meters: object scale is applied only to the mesh. Tests assert disjointness,
positive-only partition coverage and train-only draws. Every data load checks
raw H5, mesh, cache and license-receipt hashes, reconstructs every cached pose
and the actual mesh-sampled conditioning cloud exactly, verifies mesh-only
bounds/scale, and recomputes the fixed source-row permutation on CPU.

## Methods And Fixed Budget

- MGM: direct conditional generator; five critic updates per generator update,
  100 critic warmup updates, exactly 10000 generator updates, 1 NFE.
- RFM: local common-architecture geodesic matching, exactly 10000 field updates,
  exponential Euler at both 50 and 100 NFE.
- RMF-MT source-loss adaptation (`rmf_mt`): exactly 10000 field updates. The
  author `compute_loss_pc` loss tail and PCGrad execute from the pinned source.
  Only PCGrad's compile decorator is omitted, retaining its mathematics. The
  common PointNet/six-vector architecture, RoMa geodesics, sigma_min=0, consistent
  data(0)-to-noise(1) path/backward sampler, Adam settings and gradient cap are
  explicit adaptations. No forced diagonal times are added, matching the grasp
  loss time sampler. This is NOT the native SDF-network experiment reproduction.
- The earlier single-loss implementation is now `rmf_ccadapt`, explicitly
  connection-corrected RMF adaptation with 25% diagonal times; it is not a
  primary RMF-MT result. G-LSD remains an optional local adaptation.

Model seeds are 0, 1, 2; batch 256, width 128, Adam LR 0.0002. No hyperparameter,
early-stop, checkpoint or NFE selection uses validation/test. Only the fixed
final 10000-update checkpoints are primary. Report wall time and all optimizer
updates: MGM has more updates per iteration. Nine jobs total 90000 generator/
field and 150300 critic updates, or 240300 optimizer updates overall.

## Evaluation And Figures

The released author's reported grasp metric is a **static geometric proxy**:
no intersection with a thin open-gripper mesh plus intersection with the
between-finger box, aggregated as an object having any passing pose among 20.
CPU tests match the local implementation against those author functions on real
grasps. It does not close fingers, simulate friction, lift an object or test a
robot/table scene. Historical `object_in_gripper` labels came from FleX; they
cannot label newly generated poses. Every output keeps physical success `null`.

Report per-object unbiased pose MMD2, SO(3) validity, the separately named static
proxy, official Franka mesh collision fractions, timing and memory. Never call
pose MMD or static feasibility a physical grasp-success rate. Sampling uses 512
draws/object, fixed evaluation seed 96231, and 100 collision checks/object;
any-at-20 always uses the first 20 unfiltered draws. Reference sampling is capped
by the number of held-out poses. NPZ exports include metric-frame SE(3)
`transforms_m` and exact original H5 source indices for references.
Every report also includes a per-object `reference_baseline`: the SAME static
proxy on the first min(100, held-out reference count) unfiltered reference poses,
with exact source indices. Final JSON and CSV show it beside generated proxy
fractions and official mesh collisions. Historical FleX-positive labels do not
imply thin-gripper/box proxy passes. These fixed reference indices are shared
across methods, seeds and NFE; they are not tuned.

Figures show actual object meshes and the official Franka collision mesh, not
just line markers. Every object and all three seeds get PNG+PDF+JSON. The top row
has shared full-range bounds covering ALL evaluated poses and gripper vertices;
all pose origins are shown faintly, with the first 12 unfiltered draws rendered
as meshes. The middle row has mesh-defined object-centered bounds and explicit
off-frame counts, both for the first 12 and for all draws. Only fully contained
grippers are rendered in this explicitly spatial detail row. No pose is removed
from metrics or the full-range view. Every panel uses the identical camera.
JSON records the official mesh hash, source/model/data hashes, camera, bounds,
reference source indices, draw selection and off-frame counts.
An additional first-unfiltered-draw row uses common bounds containing every
first gripper and the object in full, making individual geometry readable
without replacing a draw, hiding outliers or changing any metric.

## CPU Commands

Run from the workspace using Windows PowerShell:

```powershell
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli test
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli pilot-plan
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli pilot-check
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli cpu-smoke --out-root outputs/comparison2026/grasp/verification_ready/cpu_integration_recheck
```

The prepared manifests already exist. To reproduce the within-object split:

```powershell
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli prepare --grasps-root external/acronym-data/public/data/examples/grasps --mesh-root external/acronym-data/public/data/examples --out outputs/comparison2026/grasp/data/public_within_object --license-record external/acronym-data/public/download_manifest.json --smoke-only --protocol within_object_heldout_pose
```

CPU smoke runs require a fresh output directory and never overwrite an existing
run. The completed final verification is in
`verification_ready/cpu_integration_reference/summary.json`: five methods, exact
model/critic/optimizer/all-RNG continuation, six validation-only evaluations
including RFM-50/100, independent float64 NumPy MMD/source-row audits, and both
objects' PNG+PDF figures. No primary test quality was examined. All 27 CPU tests
pass with zero skips, including the held-out reference baseline.
The PDF companions were independently rendered with
Poppler and visually inspected.

## Single GPU Command, Only After Release

The ready-to-execute nine-job plan is
`outputs/comparison2026/grasp/pilot_plan_ready.json`. The historical
`pilot_plan.json` is preserved. Writing/checking a plan launches nothing.
Existing PyTorch reports a CUDA 12.1 build; a compatible driver and explicit queue
release are required. Do not run this until the parent releases the main queue.

```powershell
.conda/rmgm/python.exe -m paper.benchmarks2026.grasp.cli pilot-run --plan outputs/comparison2026/grasp/pilot_plan_ready.json --gpu-release
```

The same command resumes interrupted training and reuses completed evaluations
only after checking final checkpoint, configuration, source and sample hashes.
It executes in one process, sequentially, with two CPU threads and no training
workers/subagents. Before EACH training/evaluation call, `nvidia-smi` must report
no foreign compute PID; missing/unsupported/failed queries fail closed. The
current process's existing CUDA context is allowed. The check is not an OS-level
GPU reservation; the parent must still coordinate the queue.

Each seed0 method first checkpoints at 200 updates for measured runtime
calibration, then continues the identical configuration to 10000. Add
`--calibrate-only` to stop after these three 200-update training probes without
ANY validation/test evaluation. Calibration never changes the fixed budget.
Later checkpoints are every 2000 updates. State resumes verify model/critic,
both optimizers and every RNG stream, including Python PCGrad ordering. Source,
manifest, dependency, device and configuration changes are rejected. Exact CPU
continuation is tested; CUDA continuation has not been executed yet.

There are exactly nine final training jobs and 24 evaluations (val+test, MGM
1 NFE, RMF-MT 1 NFE, RFM 50 and 100 NFE). Only exact final 10000-update
checkpoints reach evaluation. No quality-based tuning or early stopping occurs.

After all jobs/evaluations succeed, the separate report contract is:

- `outputs/comparison2026/grasp/final_summary.json`: per-object, seed and method
  metrics, seed mean/std, measured training updates/time/memory, limitations and
  links to all sample evidence and figures.
- `outputs/comparison2026/grasp/final_metrics.csv`: all 48 per-object metric rows.
- `outputs/comparison2026/grasp/final_audit.json`: independent source-row,
  transform, float64 MMD and unfiltered static-proxy bookkeeping checks.
- `outputs/comparison2026/grasp/figures/`: six object/seed PNG+PDF+JSON comparisons,
  including both RFM 50/100 reports, with fixed first-draw selection.
- `outputs/comparison2026/grasp/public_within_pilot/pilot_status.json` and
  `runtime_estimate.json`: resumable stage and measured CUDA training ETA.

These final files are NOT prefilled or fabricated before GPU execution.
CPU readiness and validation-only integration live separately under
`outputs/comparison2026/grasp/verification_ready/`. Original results are retained.
GPU runtime remains unmeasured until release. The refreshed exact-architecture
CPU profile estimates 1252 seconds (20.9 minutes) for nine training runs,
excluding evaluation/rendering. Each estimate uses only 100 measured updates;
shared-machine timing varies (the historical profile was 42.6 minutes). Neither
is a GPU speed prediction. The refreshed profile is in
`verification_ready/cpu_profile/summary.json`.
