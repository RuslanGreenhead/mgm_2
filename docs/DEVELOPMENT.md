# Local development setup

## Scope

This setup is for safe local development on Apple Silicon: imports, geometry
checks, short deterministic smoke tests, and code changes. It is not intended
to reproduce the paper-scale CUDA sweeps on the laptop.

## Environment

The project-local environment is `.venv`, created with Python 3.12. Core
packages are pinned to the compatible subset of
`environment/requirements-core-reference.txt`: PyTorch, NumPy, SciPy, pandas,
Matplotlib, Geoopt, and torchdiffeq. Mesh/grasp extras (`libigl`, `python-fcl`,
and related packages) are deliberately not installed because the first smoke
test is sphere-only.

Activate it with:

```sh
source .venv/bin/activate
```

## Output policy

Write all newly generated local artifacts under `artifacts/`; Git ignores the
contents. Do not overwrite `outputs/` or paper-output directories. Preserve the
archive as the baseline evidence.

## Portability status

`paper/experiments/controlled_suite.py` is not laptop-ready: it hard-codes a
CUDA device and imports mesh dependencies even for its sphere datasets. Its
historical code will remain untouched. A separate portable smoke runner should
be added before attempting Table 2 reproduction.

The Fire raw CSV is not present in this archive, so an exact Fire reproduction
also awaits documented dataset restoration.

## Recommended verification ladder

1. Confirm core imports and MPS availability.
2. Run a 20-step synthetic-sphere R-MGM smoke test into `artifacts/smoke/`.
3. Inspect its config copy, seed, device, losses, and a small generated sample
   plot; compare only for finite values and expected geometry.
4. Restore Fire data, then run one 2,000-step seed-0 Table 2 R-MGM-mixture
   cell using a new portable runner and a fresh artifact directory.
5. Scale to matched baselines and further seeds only after the single cell is
   auditable.

Never compare smoke-test metrics with the published tables.
