# R-MGM research code

The maintained project code is in `src/rmgm/`. It includes Riemannian
Midpoint Generative Models with MLP and local GMM generators, sphere/torus
geometry, modern comparison methods, and the SE(3) grasp pilot. New protocols
belong in `experiments/` with settings in `configs/`. All new run outputs go
under `outputs/`, which is ignored by Git.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the folder map and
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for local setup. The previous paper,
its code and results, and its original archive guide are preserved in
`legacy/`. The original checkout is also available at Git tag
`archive/pre-restructure-2026-11-15`.

To use the package in an existing environment, run
`python -m pip install -e . --no-deps` or set `PYTHONPATH=src`.
Run the maintained unit checks with
`PYTHONPATH=src:. python -m unittest discover -s tests -v`.
