#!/bin/bash
set -e
export PYTHONPATH=src

# Remaining from interrupted batch + 2 new seeds, all x2 runs
# exp3 s999, exp3 s2048 (exp2 already done)
# exp2+exp3 s555, s1024 (new seeds)

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

echo "======== EXP2 seed=555 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s555.yaml
echo ""
echo "======== EXP2 seed=555 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s555.yaml
echo ""

echo "======== EXP2 seed=1024 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s1024.yaml
echo ""
echo "======== EXP2 seed=1024 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp2_det_s1024.yaml
echo ""

echo "======== EXP3 seed=555 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s555.yaml
echo ""
echo "======== EXP3 seed=555 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s555.yaml
echo ""

echo "======== EXP3 seed=1024 Run 1 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s1024.yaml
echo ""
echo "======== EXP3 seed=1024 Run 2 ========"
torchrun --nproc_per_node=8 train.py --config configs/exp3_det_s1024.yaml
echo ""

echo "All 12 remaining runs complete!"
