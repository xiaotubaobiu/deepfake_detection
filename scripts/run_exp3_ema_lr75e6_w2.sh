#!/bin/bash
set -e
export PYTHONPATH=src

EXPERIMENTS=(
    "exp3_ema_s42.yaml"
    "exp3_ema_s123.yaml"
    "exp3_ema_s7.yaml"
)

for CONFIG in "${EXPERIMENTS[@]}"; do
    echo "============================================"
    echo "Running $CONFIG"
    echo "============================================"
    torchrun --nproc_per_node=8 train.py --config "configs/$CONFIG"
    echo ""
done

echo "All exp3 lr7.5e-6 warmup2 experiments complete!"
