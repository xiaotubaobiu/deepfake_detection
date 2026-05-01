#!/bin/bash
# Run all EMA experiments sequentially
# Usage: bash scripts/run_ema_experiments.sh [start_from]
# start_from: 1-6 to skip already completed experiments

set -e
export PYTHONPATH=src

EXPERIMENTS=(
    "exp2_ema_s42.yaml"
    "exp2_ema_s123.yaml"
    "exp2_ema_s7.yaml"
    "exp3_ema_s42.yaml"
    "exp3_ema_s123.yaml"
    "exp3_ema_s7.yaml"
)

START=${1:-1}

for i in $(seq $START 6); do
    CONFIG="${EXPERIMENTS[$((i-1))]}"
    echo "============================================"
    echo "[$i/6] Running $CONFIG"
    echo "============================================"
    torchrun --nproc_per_node=8 train.py --config "configs/$CONFIG"
    echo ""
done

echo "All experiments complete!"
