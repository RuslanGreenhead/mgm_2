# Conditional R-MGM on human promoters

This experiment uses the sorted FANTOM-CAT Level 3 Robust TSS table and GRCh38
genome and CAGE BigWigs distributed in [Zenodo 7943307](https://zenodo.org/records/7943307).
The archive stores the TSS table, genome, CAGE tracks, and Sei checkpoint as
gzip-compressed files; `download.py` extracts and decompresses inputs needed by
the loader and evaluator.
The first 100,000 TSS rows define the cohort. chr10 is validation; chr8/9 are
test; remaining chromosomes train. The archived table's first 100,000 rows
yield 88,570/3,933/7,497 records. RMF Appendix F.1 reports 88,470/3,933/7,497;
the 100-record train discrepancy should be reported alongside comparisons.
The official RMF loader applies a +1 genomic
offset for minus-strand TSSs, reverses and complements their DNA, reverses the
matched minus CAGE track, and zeros intervals from both supplied blacklists.
We prepare a 10-base flank on each side so train batches can shift sequence and
signal together by an integer from -10 through +10. Validation and test are
fixed at the centered crop.

The source BigWigs already contain the aggregate CAGE signal. Do not apply
another `log1p` transform. Unknown bases are represented by label 4 and map to
the uniform categorical point for training and Sei. K-mer windows containing
an unknown base are excluded and the count is reported in the manifest.

From the repository root, with the project virtual environment active:

```bash
PYTHONPATH=src python experiments/dna/download.py
PYTHONPATH=src python experiments/dna/prepare.py
PYTHONPATH=src python experiments/dna/train.py --config configs/dna/rmgm.json --output outputs/dna/seed0
PYTHONPATH=src python experiments/dna/train.py --config configs/dna/rmgm.json --output outputs/dna/seed0 --resume outputs/dna/seed0/last.pt
PYTHONPATH=src:. python experiments/dna/evaluate.py outputs/dna/seed0/best.pt --split test --output outputs/dna/seed0/test.json
PYTHONPATH=src:. python experiments/dna/summarize.py outputs/dna/seed{0,1,2} --output outputs/dna/three_seeds.json
```

All data paths can be outside the repository. For a cluster data directory,
pass `--data`, `--output`, and, when using Sei, `--sei-data-dir` explicitly.
For example:

```bash
PYTHONPATH=src python experiments/dna/train.py --config configs/dna/rmgm.json \
  --data /absolute/path/processed/promoters_rmf10 \
  --output /absolute/path/runs/rmgm/seed0 \
  --eval-sei --sei-data-dir /absolute/path/raw/promoter_design
```

If the full archive is downloaded manually, save it as `data/raw/data.tar.gz`
and run `PYTHONPATH=src python experiments/dna/download.py --extract-only`.
That command checks the published MD5 before extracting the promoter files.
The required preparation readers are listed in `requirements-prepare.txt`.
For the optional Sei metric, install its small dependency list. On older HPC
systems, require wheels to avoid building pandas or scipy from source:

```bash
python -m pip install --only-binary=:all: -r experiments/dna/requirements-sei.txt
```

The vendored Sei evaluator implements Selene's 1D mean-mode strand averaging
locally, so it does not import Selene's training and genome dependencies. Run a
small GPU Sei forward on the target cluster before relying on that metric in a
full training run.

For a short local test after preparation, use `--max-steps 3` and a smaller
configuration. The default 20-block networks are intended for a GPU. Add
`--eval-sei` only after the PyTorch Sei checkpoint and `target.sei.names` are
available in `sei_data_dir`. `--test` evaluates the selected checkpoint on chr8/9.

The metric code computes pooled six-mer Pearson correlation and the vendored
KGDM Sei evaluator's paired H3K4me3 mean squared error. The latter pads each
1024-base sequence to 4096 bases with 0.25 and averages Sei's H3K4me3 outputs.
The selected checkpoint uses validation Sei MSE when `--eval-sei` is enabled,
otherwise validation six-mer correlation. Do not compare the two selection
rules as though they were the same published protocol.

The training objective is the maintained R-MGM hidden-flip objective. The critic
sees time `t` in [0, 0.5] and a path point at either `t` or `1-t`; it predicts a
product-manifold tangent velocity. The generator minimizes `2<f,a>-<f,f>`
through the frozen critic. Loss inner products use the unit-sphere metric,
averaged over bases and batch. Fisher-Rao distances are twice the product-
sphere distance; this uniform factor would scale both losses by four.

Use `--seed 0`, `--seed 1`, and `--seed 2` with distinct output directories for
the three full runs. The runner uses plain PyTorch and JSON configuration, so the same command can
be placed in a Slurm job once cluster paths, GPU limits, and environment modules
are known. It writes `best.pt`, `last.pt`, `history.jsonl`, `run.json`, and
optionally `test.json` under the selected output directory.
