# 🧲 Kernel-Gradient Drifting Models

> Code for the NeurIPS 2026 submission _Kernel-Gradient Drifting Models_.
> Three independent experiment folders reproduce the paper's three empirical sections.

**New here?** Start with [`demo.ipynb`](demo.ipynb) — a short, self-contained
notebook that demonstrates kernel-gradient drifting on a 2-D toy.

## 📁 Layout

```
demo.ipynb  self-contained intro notebook (no data, CPU/MPS)
toy/        synthetic experiments 
  euclidean/    Matérn-ν sweep on R^2 (Fig. 5.2)
  geometric/    Sphere / hyperboloid drifts (Tab. 5.1 + Fig. 5.1)
earth/      Earth-event experiments on S^2 (Tab. 5.2 / Tab. 5.5)
dna/        Promoter DNA experiments (Tab. 5.4)
environment.yml   conda env shared by all folders
```

Each subfolder has its own `README` with reproduction commands.

## ⚙️ Install

```bash
conda env create -f environment.yml
conda activate kgd
```

The same environment is used for `toy/`, `earth/`, and `dna/`.

## 📦 Data

The two real-data experiments need datasets that we do **not** redistribute.
Download them from the original sources and place the files where each
experiment expects them.

### 🌍 Earth (`earth/`)

The four Earth-event datasets (`earthquake`, `volcano`, `fire`, `flood`) are
the ones used by Chen & Lipman (2024), available from the Riemannian Flow
Matching repository:

```
https://github.com/facebookresearch/riemannian-fm
```

Follow that repository's data instructions to obtain the four CSVs, then
place them under `earth/data/earth_data/`:

```
earth/data/earth_data/
  earthquake.csv
  volcano.csv
  fire.csv
  flood.csv
```

The path is configurable via `data.data_dir` in `earth/configs/config.yaml`
(default: `data/earth_data`).

### 🧬 DNA (`dna/`)

The promoter dataset is from Stark et al. (2024) and is hosted on Zenodo:

```
https://zenodo.org/records/7943307
```

Download the promoter files and place them under `dna/data/dna_promoter/`:

```
dna/data/dna_promoter/
  Homo_sapiens.GRCh38.dna.primary_assembly.fa
  FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv
  agg.plus.bw.bedgraph.bw
  agg.minus.bw.bedgraph.bw
```

Evaluation also needs the pretrained Sei model (Chen et al., 2022) from the
[Sei framework](https://github.com/FunctionLab/sei-framework); place these in
the same directory:

```
dna/data/dna_promoter/
  best.sei.model.pth.tar
  target.sei.names
```

The path is configurable via `data.data_dir` in `dna/configs/config.yaml`
(default: `data/dna_promoter`).

### 🧪 Toy (`toy/`)

No external data — synthetic samplers live in
`toy/euclidean/tasks.py` and `toy/geometric/tasks.py`.

## 📊 Results & reproduction

All numbers below are reproduced from the paper. Each table is followed by
the precise command. Subfolder READMEs document every flag.

### Tab. 5.1 — Synthetic geometric (Laplace kernel, mean ± std over 3 seeds)

`SW` is sliced Wasserstein-2 (lower is better); `Tile` is black-square
accuracy on checkerboard targets (higher is better). **Bold** = best per row.

| Dataset      | Manifold    | Metric | Ambient. Disp    | Ambient. Gradient    | Manif. Disp | Manif. Gradient  |
| ------------ | ----------- | ------ | ------------- | ----------------- | --------------- | ------------------ |
| Checkerboard | Sphere      | SW ↓   | 0.031 ± 0.009 | 0.012 ± 0.000     | 0.032 ± 0.008   | **0.011 ± 0.002**  |
| Checkerboard | Sphere      | Tile ↑ | 0.76 ± 0.03   | 0.86 ± 0.01       | 0.80 ± 0.04     | **0.87 ± 0.01**    |
| Checkerboard | Hyperboloid | SW ↓   | 0.026 ± 0.003 | **0.025 ± 0.001** | 0.063 ± 0.022   | 0.037 ± 0.013      |
| Checkerboard | Hyperboloid | Tile ↑ | 0.79 ± 0.01   | 0.85 ± 0.02       | 0.80 ± 0.02     | **0.90 ± 0.01**    |
| Swissroll    | Sphere      | SW ↓   | 0.018 ± 0.002 | **0.014 ± 0.002** | 0.026 ± 0.000   | **0.014 ± 0.002**  |
| Swissroll    | Hyperboloid | SW ↓   | 0.041 ± 0.001 | **0.030 ± 0.003** | 0.042 ± 0.007   | 0.044 ± 0.015      |

```bash
cd toy/geometric
SEEDS="0 1 2" bash run_all.sh
python make_compare_fig.py            # Fig. 5.1 
python make_compare_fig_main.py       # Tab. 5.1 numbers
python make_evolution_grid.py         # (Paper appendix)
```

### Fig. 5.2 — Matérn ν sweep on R² (C2ST ↓)

Sweep over `methods × ν × 3 seeds` on the Euclidean swissroll task; the
figure is the headline output (no companion table in the paper).

```bash
cd toy/euclidean
python run_sweep.py
python make_paper_fig.py              # Fig. 5.2
```

### Tab. 5.2 — Earth (S²), MMD ↓

9 formulations × 4 datasets (mean over 3 seeds). The first row is the Euclidean
Laplace drift baseline from Deng et al. (2026); remaining rows are our gradient,
spherical, and spectral extensions. Hyperparameters per cell are committed in
`earth/configs/winners.yaml`. **Bold** = best per dataset. The full table with
Sinkhorn / NN1 and std is in `earth/README.md`.

| Formulation                     | Volcano   | Earthquake | Fire      | Flood     |
| ------------------------------- | --------- | ---------- | --------- | --------- |
| Euclidean Laplace (Deng et al.) | 0.121     | 0.039      | 0.031     | 0.058     |
| Euclidean Laplace Gradient      | 0.105     | 0.036      | 0.028     | 0.048     |
| Euclidean Gaussian Gradient     | 0.140     | **0.033**  | **0.027** | 0.052     |
| Spherical Laplace               | 0.116     | 0.045      | 0.028     | 0.062     |
| Spherical Laplace Gradient      | 0.115     | 0.037      | 0.029     | 0.058     |
| Spherical Gaussian Gradient     | **0.094** | 0.034      | 0.028     | 0.051     |
| Spectral (ν = 0.5)              | 0.112     | 0.035      | 0.030     | 0.052     |
| Spectral (ν = 2.5)              | 0.104     | 0.048      | 0.029     | **0.045** |
| Spectral (ν = ∞)                | 0.142     | 0.043      | 0.041     | 0.055     |

Reproduce a single cell (3 seeds); each run writes `results/<run_name>.json`,
then average the final test MMD over the seeds:

```bash
cd earth
for s in 0 1 2; do
  python -m src.main --formulation sphere-laplace-gradient --dataset volcano --seed $s
done
```

### Tab. 5.4 — Promoter DNA (Sei MSE ↓, 6-mer correlation ↑)

Sei activity MSE (lower is better) and 6-mer correlation (higher is better),
mean ± std over 3 seeds at the best-k-mer checkpoint. Flow-based baselines show
their 1-NFE results. **Bold** = best among the drifting (ours) variants;
hyperparameters per formulation are committed in `dna/configs/winners.yaml`.

| Method                              | 6-mer Corr ↑    | Sei MSE ↓         |
| ----------------------------------- | --------------- | ----------------- |
| _Flow distillation (1-NFE)_         |                 |                   |
| G-ESD + v-pred                      | 0.13            | 0.055             |
| G-LSD + v-pred                      | 0.73            | 0.046             |
| G-PSD + v-pred                      | 0.82            | 0.035             |
| E-RMF + v-pred                      | 0.96 ± 0.00     | 0.031 ± 0.001     |
| E-RMF + x-pred                      | 0.96 ± 0.01     | 0.030 ± 0.000     |
| S-RMF + v-pred                      | 0.93 ± 0.02     | 0.030 ± 0.001     |
| S-RMF + x-pred                      | 0.84 ± 0.03     | 0.030 ± 0.001     |
| L-RMF + v-pred                      | 0.85 ± 0.01     | 0.027 ± 0.001     |
| L-RMF + x-pred                      | 0.88 ± 0.00     | 0.027 ± 0.001     |
| _Drifting (ours)_                   |                 |                   |
| Ambient Displacement (Deng et al.)  | 0.56 ± 0.03     | 0.044 ± 0.001     |
| Ambient Gradient                    | 0.62 ± 0.01     | **0.032 ± 0.001** |
| Simplex Displacement                | 0.70 ± 0.10     | 0.047 ± 0.004     |
| Simplex Gradient                    | 0.82 ± 0.02     | 0.045 ± 0.004     |
| Sphere Displacement                 | 0.78 ± 0.08     | 0.044 ± 0.004     |
| Sphere Gradient                     | **0.84 ± 0.04** | 0.043 ± 0.002     |

Train the headline recipe end-to-end (val/test Sei MSE + k-mer correlation are
logged during training):

```bash
cd dna
python -m src.main --formulation sphere-grad
```

## 📄 License

This repository is released under the MIT license — see [`LICENSE`](LICENSE).
The Earth and DNA datasets are governed by the licenses set by their
respective hosts.
