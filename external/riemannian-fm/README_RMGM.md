# Riemannian MGM experiments

This directory contains a lightweight R-MGM layer on top of the copied
`facebookresearch/riemannian-fm` manifold code.

## Environment

The local Windows conda environment lives at:

```powershell
E:\Codex Projects\Riemmanian Midpoint Generative Models\.conda\rmgm
```

Use `conda run`; directly invoking `python.exe` can miss CUDA DLLs for SPD.

```powershell
& E:\anaconda\Scripts\conda.exe run -p ..\..\.conda\rmgm python -B rmgm_smoke.py
```

Expected smoke output should include `device=cuda` and finite losses for
`torus2d`, `spd`, `hyperbolic`, and `sphere`.

## Training

Run commands from `external\riemannian-fm`.

```powershell
& E:\anaconda\Scripts\conda.exe run -p ..\..\.conda\rmgm python -B rmgm_train.py --experiment torus2d --steps 2500 --batch-size 512 --hidden-dim 320 --num-layers 4 --latent-dim 32 --critic-steps 3 --lr-g 5e-5 --lr-c 2e-4 --log-every 100 --plot-every 250 --eval-samples 4096 --output-dir ../../outputs/rmgm_refine_mod --torus-map mod
```

```powershell
& E:\anaconda\Scripts\conda.exe run -p ..\..\.conda\rmgm python -B rmgm_train.py --experiment spd --steps 2000 --batch-size 256 --hidden-dim 192 --num-layers 4 --latent-dim 12 --log-every 200 --plot-every 500 --eval-samples 4096 --output-dir ../../outputs/rmgm_runs
```

```powershell
& E:\anaconda\Scripts\conda.exe run -p ..\..\.conda\rmgm python -B rmgm_train.py --experiment sphere --steps 2500 --batch-size 512 --hidden-dim 256 --num-layers 4 --latent-dim 16 --log-every 250 --plot-every 625 --eval-samples 4096 --output-dir ../../outputs/rmgm_runs
```

```powershell
& E:\anaconda\Scripts\conda.exe run -p ..\..\.conda\rmgm python -B rmgm_train.py --experiment hyperbolic --steps 1500 --batch-size 384 --hidden-dim 192 --num-layers 4 --latent-dim 16 --log-every 150 --plot-every 375 --eval-samples 4096 --output-dir ../../outputs/rmgm_runs
```

Each run writes:

- `config.json`
- `metrics.jsonl`
- `metrics.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `figs/samples-*.png`

For mixture-like toy targets, the strongest one-step generator is the local GMM
head initialized by manifold-aware k-means on target samples, then trained with
the same MGM objective:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_train.py --experiment torus2d --steps 2000 --batch-size 512 --latent-dim 16 --hidden-dim 256 --num-layers 4 --critic-steps 3 --lr-g 3e-5 --lr-c 2e-4 --log-every 100 --plot-every 250 --eval-samples 4096 --torus-generator gmm --torus-components 4 --torus-residual-scale 0.25 --torus-kmeans-init --output-dir ../../outputs/rmgm_refine_torus_gmm_kmeans
```

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_train.py --experiment sphere --steps 2000 --batch-size 512 --latent-dim 16 --hidden-dim 256 --num-layers 4 --critic-steps 3 --lr-g 3e-5 --lr-c 2e-4 --log-every 100 --plot-every 250 --eval-samples 4096 --generator-type local_gmm --generator-components 4 --generator-residual-scale 0.27 --generator-kmeans-init --output-dir ../../outputs/rmgm_refine_sphere_local_gmm
```

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rmgm_train.py --experiment hyperbolic --steps 2000 --batch-size 512 --latent-dim 16 --hidden-dim 256 --num-layers 4 --critic-steps 3 --lr-g 3e-5 --lr-c 2e-4 --log-every 100 --plot-every 250 --eval-samples 4096 --generator-type local_gmm --generator-components 4 --generator-residual-scale 0.20 --generator-kmeans-init --output-dir ../../outputs/rmgm_refine_hyperbolic_local_gmm
```

The RFM baseline mirror is:

```powershell
& E:\anaconda\Scripts\conda.exe run -p "..\..\.conda\rmgm" python -B rfm_baseline_train.py --experiment torus2d --steps 2500 --batch-size 512 --hidden-dim 256 --num-layers 4 --lr 2e-4 --ode-steps 128 --log-every 250 --plot-every 500 --eval-samples 4096 --output-dir ../../outputs/rfm_baseline
```

Use `--ode-steps 1` for the same-NFE baseline.

## Implemented pieces

- `manifm/rmgm/geometry.py`: midpoint flip sampling and RFM geodesic target
  velocities.
- `manifm/rmgm/generators.py`: constrained one-step generators.
- `manifm/rmgm/critics.py`: tangent-valued critic network.
- `manifm/rmgm/losses.py`: critic regression and variational generator loss.
- `rmgm_train.py`: standalone synthetic experiments and paper-style plots.
- `rfm_baseline_train.py`: matching lightweight RFM baseline.
- `rmgm_compare_results.py`: collects best/final MMD and constraint metrics.
- `rmgm_render_checkpoint.py`: renders figures from saved generator checkpoints.
- `rmgm_make_contact_sheet.py`: combines selected figures into a panel.
- `rmgm_make_method_grid.py`: renders separate Ground truth / R-MGM /
  RFM-full / RFM-1-step panels from checkpoints.

## Current synthetic results

Best MMD², seed 0, 4096 evaluation samples:

| Experiment | full RFM | same-NFE RFM | best R-MGM |
| --- | ---: | ---: | ---: |
| `sphere` | 0.000602 | 0.004631 | **0.000188** |
| `spd` | 0.004529 | 0.007526 | **0.003491** |
| `hyperbolic` | 0.002294 | 0.004861 | **0.000452** |
| `torus2d` | 0.002385 | 0.013952 | **0.000151** |

The current contact sheet is written to
`../../outputs/rmgm_summary/structured_rmgm_contact_sheet.png`.

The method-by-method grid is written to
`../../outputs/rmgm_summary/method_comparison_grid.png`.

## Current scope

The first supported experiments are synthetic simple geometries:

- `torus2d`
- `spd`
- `hyperbolic`
- `sphere`

Mesh and external-data RFM experiments are intentionally left for the next pass
because the Windows stack needs `igl`, `meshplot`, `geopandas`, and the original
downloaded datasets.
