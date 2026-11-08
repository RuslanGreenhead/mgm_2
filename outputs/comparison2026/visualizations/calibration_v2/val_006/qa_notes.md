# Validation Preview QA

- Date: 2026-10-26. CPU only; one numerical-library thread; no torch import.
- Plot self-test: 24 tests pass, including a reproduced-and-fixed comparison
  heading overlap regression and RNA zoom heading/caption bounds checks.
- Snapshot: 26 completed candidates; 10 selected seed-0 runs; 26 missing cells.
  Fire and General: all four methods. RNA: MGM and KGDM only. This is not final.
- 37 comparison/overview/curve/individual PNGs plus 37 vector PDFs.
- Comparisons: 4800 px wide; Fire and General: 4800x1320.
- 28 individual zoom PNGs: 2880 px wide; Fire 2880x1728; RNA 2880x2736.
- PNG dimensions and nonblank pixel variation checked for every manifest figure.
- Every zoom's density limits exactly match its corresponding comparison row.
- Recursive PDF resource inspection: 37 pages, zero embedded raster images.
- Actual PNG visual QA: Fire comparison/zoom, General comparison, RNA six-pair
  layout and angle-6/7 zoom, validation learning curves.
- PDF visual QA: Poppler rendering of Fire and RNA angle-6/7 zooms.
- Accessed array members: val_target and generated_draw_0 only. No test-step
  metrics, test samples or test targets were read.
- All-candidate CSV/Markdown tables retain config, resolved defaults, variant,
  phase and run identity. Draw SD and training-seed SD are separate; unavailable
  uncertainty and runs remain missing. Kernel MMD2_V is not matching distance.
- Use panel_index.md for comparisons and every available individual target/method
  zoom, including all six RNA angle pairs. Existing earlier revisions remain.
- Final test rendering was not attempted: final_v1 has not arrived. The CLI's
  --require-complete command requires all 108 dataset/method/seed cells first.
