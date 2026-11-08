#!/usr/bin/env bash
# Run the full grid for Tab. 1 / Fig. 3:
#   datasets × geometries × methods × seeds  =  4 × 2 × 2 × N  runs.
#
# Each run calls main.py with the per-dataset YAML config under configs/
# and writes to results/<dataset>/<geometry>-<method>/seed<n>/.
#
# Usage:
#   bash run_all.sh                    # default seeds=0; 16 runs
#   SEEDS="0 1 2" bash run_all.sh      # 3 seeds; 48 runs
#   DATASETS="swissroll_sphere swissroll_hyperboloid" bash run_all.sh   # subset
#
# All runs are sequential and CPU-friendly. Use & + wait to parallelize across
# cores if you want — left out here to keep output readable.

set -euo pipefail

SEEDS="${SEEDS:-0}"
DATASETS="${DATASETS:-checkerboard_sphere checkerboard_hyperboloid swissroll_sphere swissroll_hyperboloid}"
GEOMS="${GEOMS:-euclidean manifold}"
METHODS="${METHODS:-base gradient}"

cd "$(dirname "$0")"

n=0
for ds in ${DATASETS}; do
  for geom in ${GEOMS}; do
    for meth in ${METHODS}; do
      for seed in ${SEEDS}; do
        out="results/${ds}/${geom}-${meth}/seed${seed}"
        echo ""
        echo "=== ${ds} | ${geom}-${meth} | seed=${seed} -> ${out} ==="
        python main.py \
          --dataset "${ds}" --geometry "${geom}" --method "${meth}" \
          --seed "${seed}" --out-dir "${out}"
        n=$((n+1))
      done
    done
  done
done

echo ""
echo "ran ${n} configurations"
