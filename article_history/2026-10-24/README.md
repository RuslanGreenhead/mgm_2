# Midpoint Generative Models for General Geometries

English ICML-format research manuscript. `main_technical_note.tex` and
`references_technical_note.bib` preserve the earlier technical note.

## Build With MiKTeX

From the workspace root:

```powershell
powershell -ExecutionPolicy Bypass -File paper/build.ps1
```

The script uses the existing MiKTeX installation under
`C:\Program Files\MiKTeX\miktex\bin\x64`. It runs PDFLaTeX, BibTeX,
and two further PDFLaTeX passes. The result is `paper/main.pdf`.
The official ICML 2026 style and bibliography style are unchanged.

Official rules: https://icml.cc/Conferences/2026/AuthorInstructions

Official style archive:
https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip

The submission format is anonymous, with at most eight main-text pages.
References and appendices follow in the same PDF. This format choice does not
assert that an already completed ICML 2026 submission is being made.
The official call permits AI assistance but assigns responsibility to human
authors; this draft discloses AI-assisted drafting, code preparation, and
automated internal checking in the appendix. Human scientific review remains
necessary before submission.
https://icml.cc/Conferences/2026/CallForPapers

## Rebuild Figures and Tables

Use the project conda Python with PyTorch and scientific plotting packages:

```powershell
& '.conda/rmgm/python.exe' paper/check_geometry.py
& '.conda/rmgm/python.exe' paper/make_illustrations.py
& '.conda/rmgm/python.exe' paper/experiments/verify_independent.py
& '.conda/rmgm/python.exe' paper/update_results.py --require-complete --run-root controlled_independent
```

`generated/` contains data-derived TeX tables and presentation statistics.
`figures/` contains manuscript PDF figures and larger standalone PNGs.
The Natural Earth land outline under `vendor/` is public-domain geographic
context, downloaded from the Natural Earth vector repository:
https://github.com/nvkelso/natural-earth-vector

## Research Evidence

New controlled data, checkpoints, and result ledgers are under
`outputs/paper_icml2026/` relative to the workspace root. The experiment
implementation and exact commands are in `paper/experiments/README.md`.
All new primary runs use fixed final checkpoints and three training seeds.
Repeated sampling draws are not additional training seeds.

The final primary bundle is `outputs/paper_icml2026/controlled_independent/`.
Its `FINAL_TABLES_READY.json` gate requires all 54 corrected independent-RNG
reruns and the separate float64-normalized sphere metric correction. The
18 unaffected MLP objective-control jobs are copied with provenance hashes.
The original preliminary controlled bundle is retained, not used as final
primary evidence. Frozen training sources, metric-correction sources, raw
metric archives, generated arrays and optimizer/RNG-bearing rerun checkpoints
are preserved separately. Some GPU timing was resource-contended, not an
isolated hardware speed benchmark.

The manuscript distinguishes:

- Exact population results for reversible finite-action paths and true tangent
  velocities, including a Wasserstein-1 lower bound.
- Controlled sphere/torus experiments with a smaller RFM architecture and
  explicitly stated update budgets.
- Exploratory historical mesh and additional-geometry pilots.

Full million-step original-author RFM runs, an exact differentiable mesh path
implementation, and large real EEG SPD experiments are not certified as
completed. The projected mesh surrogate is outside the exact-path theorem.
No sample discrepancy is equated with a published NLL.

The separate true-velocity randomized-torus diagnostic uses only General
training/validation data and three fixed noise amplitudes, with one model seed.
It does not change the primary benchmark. To run in a fresh output directory:

```powershell
& '.conda/rmgm/python.exe' paper/noisy_torus_study.py --check
& '.conda/rmgm/python.exe' paper/noisy_torus_study.py
& '.conda/rmgm/python.exe' paper/verify_noisy_torus.py
```

An earlier interrupted trial is preserved under the suffixed output directory
and excluded. The final sweep restarts all three amplitudes with the same fixed
settings; its verification records source hashes and zero test evaluations.

The reviewer archive is used as a quality rubric. A single independent review
agent examines the manuscript and rendered PDF without spawning descendants.
Review notes and responses are stored under `paper/review/`.

This is a complete research draft, not a certification of ICML acceptance or
submission readiness. Population proofs withstand the independent review;
current controlled experiments do not demonstrate robust practical superiority
of the midpoint objective. The abstract and conclusion disclose this directly.
