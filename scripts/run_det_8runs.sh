#!/bin/bash
set -e
export PYTHONPATH=src

# 8 runs: exp2 x2 seeds x2 repeats + exp3 x2 seeds x2 repeats
# Same hyperparameters: lr=1e-5, warmup=1, ema=0.999, epochs=15, batch=128

echo "======== EXP2 seed=123 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s123.yaml
echo ""

echo "======== EXP2 seed=123 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s123.yaml
echo ""

echo "======== EXP2 seed=7 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s7.yaml
echo ""

echo "======== EXP2 seed=7 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s7.yaml
echo ""

echo "======== EXP3 seed=123 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s123.yaml
echo ""

echo "======== EXP3 seed=123 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s123.yaml
echo ""

echo "======== EXP3 seed=7 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s7.yaml
echo ""

echo "======== EXP3 seed=7 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s7.yaml
echo ""

echo "All 8 deterministic runs complete!"
