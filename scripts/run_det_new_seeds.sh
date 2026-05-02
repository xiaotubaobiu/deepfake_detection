#!/bin/bash
set -e
export PYTHONPATH=src

# 8 runs: exp2 x2 new seeds x2 repeats + exp3 x2 new seeds x2 repeats

echo "======== EXP2 seed=999 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s999.yaml
echo ""

echo "======== EXP2 seed=999 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s999.yaml
echo ""

echo "======== EXP2 seed=2048 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s2048.yaml
echo ""

echo "======== EXP2 seed=2048 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s2048.yaml
echo ""

echo "======== EXP3 seed=999 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s999.yaml
echo ""

echo "======== EXP3 seed=999 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s999.yaml
echo ""

echo "======== EXP3 seed=2048 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s2048.yaml
echo ""

echo "======== EXP3 seed=2048 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s2048.yaml
echo ""

echo "All 8 new-seed runs complete!"
