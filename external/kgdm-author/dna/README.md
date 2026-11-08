# 🧬 DNA promoter experiments

Conditional generation of human promoter sequences by **kernel-gradient drifting**.
One dilated-CNN backbone maps `noise + per-position CAGE signal → nucleotide
logits`; the drifting loss transports those logits toward real sequences under a
kernel on the chosen geometry. Conditioning is induced by **activity-stratified
("coherent") batches** rather than by the kernel itself.

Conda env: see `../environment.yml`.

## 📁 Layout

```
configs/
  config.yaml        training defaults (data_dir, geometry, optimizer, objective)
  winners.yaml       tuned per-formulation scalars (committed)
src/
  main.py            entry point: config.yaml + winners.yaml + CLI -> one run
  train.py           training loop, evaluation, 3-checkpoint test eval
  model/
    backbone.py        20-block dilated 1D CNN (time-conditioning removed)
    drift_core.py      drift losses: ambient / simplex / sphere (+ grad form)
    spectral_kernel.py truncated spherical-harmonic Matern kernel on S^3
  data/
    dataset.py         pyfaidx + pybigtools promoter dataset (chromosome splits)
    cluster_sampler.py coherent (activity-stratified) batch sampler
  eval/
    kmer.py            pooled 6-mer frequency correlation
    sei.py sei_eval.py Sei chromatin model + the H3K4me3 activity readout
  utils/logger.py
```

## ⚙️ Setup

Provide a data directory containing the promoter FASTA (+ `.fai`), the
plus/minus CAGE bigWigs, the FANTOM TSS table and the Sei weights
(`best.sei.model.pth.tar`). `data.data_dir` in `configs/config.yaml` defaults to
the relative `data/dna_promoter` (resolved from the `dna/` directory you run
from) — either symlink your copy there (`ln -s /abs/path dna/data`) or override
per-run with `--data_dir /abs/path`. Splits are by chromosome: **train** = all
but chr8/9/10, **valid** = chr10, **test** = chr8/9.

## 🎯 Reproduce a paper recipe

```bash
cd dna
python -m src.main --formulation sphere-grad        # headline winner
```

`--formulation` pins the cell's geometry and loads its tuned scalars from
`configs/winners.yaml`. Available keys:

| formulation       | geometry                         | grad form |
|-------------------|----------------------------------|-----------|
| `ambient-nograd`  | raw logits (unbounded)           | no        |
| `ambient-grad`    | raw logits                       | yes       |
| `simplex-nograd`  | softmax simplex                  | no        |
| `simplex-grad`    | softmax simplex                  | yes       |
| `sphere-nograd`   | Fisher–Rao sphere (√-embedding)  | no        |
| **`sphere-grad`** | Fisher–Rao sphere                | **yes**   |
| `spectral`        | Matern kernel score on `(S³₊)^L` | n/a       |

All cells use coherent batches (`sei_batch_tau 0.1`), 200 epochs, batch 128,
EMA 0.999 and the satisficing objective. Any field can be overridden, e.g.

```bash
python -m src.main --formulation sphere-grad --seed 1 --epochs 200
```

Reported numbers are **test (chr8/9) at the best-k-mer checkpoint**, mean ± std
over seeds 0/1/2 (`wandb` summary keys `test_{kmer_corr,sei_mse}_at_best_kmer`):

| formulation     | 6-mer corr ↑   | Sei MSE ↓        |
|-----------------|----------------|------------------|
| ambient-nograd  | 0.555 ± 0.027  | 0.044 ± 0.001    |
| ambient-grad    | 0.617 ± 0.013  | **0.032 ± 0.001** |
| simplex-nograd  | 0.70 ± 0.10    | 0.047 ± 0.004    |
| simplex-grad    | 0.82 ± 0.02    | 0.045 ± 0.004    |
| sphere-nograd   | 0.78 ± 0.08    | 0.044 ± 0.004    |
| **sphere-grad** | **0.84 ± 0.04**| 0.043 ± 0.002    |
| spectral        | ~0.78          | ~0.049           |


## ⚡ Run a single custom job

Edit `configs/config.yaml` (or override on the CLI) to pick a geometry and
hyperparameters, then:

```bash
cd dna
python -m src.main \
  --sphere true --grad_form true --dist_norm true \
  --lr 1.1e-4 --t_list "0.01,0.04,0.15" \
  --sei_batch_tau 0.1 --epochs 200 --seed 0
```

Exactly one of `--sphere/--simplex/--spectral` may be true (all false = the
ambient/raw-logit cell). `grad_form` does not apply to the spectral kernel.


