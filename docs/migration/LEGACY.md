# Legacy and provenance rules

`legacy/` is an index today. Keep the historical R-MGM implementation in the
modified RFM tree until its broader geometry paths have equivalence coverage.
Keep the controlled suite and SE(3) pilot at their existing locations until
their protocols can be migrated with their data and source hash rules.

New code belongs in `src/rmgm/`, `experiments/`, and `tests/`. Promote one
implementation at a time, document semantic differences, and preserve old
source and result manifests. Do not use old checkpoint files with a new
implementation without an explicit state and output equivalence test.
