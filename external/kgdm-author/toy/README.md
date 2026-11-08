# 🧪 Toy experiments

Two pipelines, one per paper experiment:

- 📈 `euclidean/` — Fig. 5.2 (C2ST vs Matérn ν, Euclidean swiss-roll).
- 🌐 `geometric/` — Tab. 5.1 + Fig. 5.1 (+ appendix figures) (drift on S² and H², checkerboard and swiss-roll targets, 2×2 design `geometry × method`).

Both train on CPU. Hyperparameters live in YAML configs:
`euclidean/config.yml` and `geometric/configs/<dataset>.yaml`.

## 📈 Fig. 5.2

```bash
cd euclidean
python run_sweep.py        # 2 methods × 7 ν × 3 seeds → results/swissroll/paper-fig/metrics.json
python make_paper_fig.py   # → results/swissroll/paper-fig/paper_c2st_vs_nu.png
```

## 🌐 Tab. 5.1 + appendix figures

```bash
cd geometric
SEEDS="0 1 2" bash run_all.sh    # 4 datasets × 2 geometries × 2 methods × 3 seeds = 48 runs

python make_compare_fig.py       # → results/compare_grid.png    (+ Tab. 5.1 stats to stdout; Fig. 5.1 in appendix)
python make_compare_fig_main.py  # → results/compare_grid_main.png
python make_evolution_grid.py    # → results/<dataset>/evolution_grid.png  (appendix figures)
```

`swissroll_hyperboloid.yaml` carries an `overrides` block for its two
`manifold-*` cells; `main.py` applies them automatically.
