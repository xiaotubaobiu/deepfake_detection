#!/bin/bash
set -e
export PYTHONPATH=src

CONFIG="exp3_det_s42.yaml"

echo "============================================"
echo "Run 1: $CONFIG"
echo "============================================"
torchrun --nproc_per_node=8 train.py --config "configs/$CONFIG"
echo ""

echo "============================================"
echo "Run 2: $CONFIG (same seed, verify reproducibility)"
echo "============================================"
torchrun --nproc_per_node=8 train.py --config "configs/$CONFIG"
echo ""

echo "Both deterministic runs complete!"
