# Dataset storage

Dataset loading and preprocessing code belongs in `src/rmgm/data/` when
promoted into the maintained package. The historical loaders and expected
paths are documented in [DATA_AND_ASSETS.md](../legacy/docs/DATA_AND_ASSETS.md).
The archive intentionally omits most raw data and binary split arrays.

Keep new dataset manifests here; keep raw data, prepared arrays, and caches
out of Git. Dataset identity, splits, preprocessing version, and hashes must
be recorded in each new experiment protocol. Existing files in `external/`
and `legacy/outputs/` are historical references and should not be relocated merely
to fill this directory.
