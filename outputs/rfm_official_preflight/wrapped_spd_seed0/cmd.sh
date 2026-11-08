#!/bin/bash

python train.py \
	experiment=wrapped_spd \
	seed=0 \
	use_wandb=False \
	optim.num_iterations=1 \
	val_every=100000 \
	hydra.run.dir=E:/Codex Projects/Riemmanian Midpoint Generative Models/outputs/rfm_official_preflight/wrapped_spd_seed0
