#!/bin/bash

python train.py \
	experiment=wrapped_spd \
	seed=0 \
	use_wandb=False \
	earth_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/earth_data \
	top500_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/top500 \
	rna_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/rna \
	mesh_datadir=E:/Codex Projects/Riemmanian Midpoint Generative Models/external/riemannian-fm/data/mesh \
	hydra.run.dir=E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/rfm_official_full/seed0/wrapped_spd_d3/train \
	wrapped.dim=3
