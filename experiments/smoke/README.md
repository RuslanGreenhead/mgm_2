# Sphere R-MGM smoke test

`run_sphere_rmgm.py` is an isolated local functional test for the historical
R-MGM implementation. It uses the real R-MGM mixture generator, tangent
critic, midpoint losses, and RFM baseline helpers from
`external/riemannian-fm`; it does not call or modify the CUDA-bound controlled
suite.

Run it from the project root after installing the local environment:

```sh
.venv/bin/python experiments/smoke/run_sphere_rmgm.py \
  --config configs/development/sphere_smoke.json
```

It produces a new, non-overwriting directory below `artifacts/smoke/`.
The generated losses and sample metrics are functional checks only. They are
not comparable with paper tables or the Fire dataset.

For the medium synthetic diagnostic, use
`configs/development/sphere_diagnostic.json` with seeds 0, 1, and 2. It adds
small sphere-MMD and nearest-neighbour diagnostics for within-run monitoring;
they are not the paper evaluation protocol.
