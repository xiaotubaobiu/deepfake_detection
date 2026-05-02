#!/bin/bash
set -e
export PYTHONPATH=src

EXPERIMENTS=(
  "configs/exp2_fullffpp_ema_s42.yaml"
  "configs/exp2_fullffpp_ema_s123.yaml"
  "configs/exp2_fullffpp_ema_s7.yaml"
  "configs/exp2_fullffpp_ema_s999.yaml"
  "configs/exp2_fullffpp_ema_s2048.yaml"
  "configs/exp3_fullffpp_ema_s42.yaml"
  "configs/exp3_fullffpp_ema_s123.yaml"
  "configs/exp3_fullffpp_ema_s7.yaml"
  "configs/exp3_fullffpp_ema_s999.yaml"
  "configs/exp3_fullffpp_ema_s2048.yaml"
)

for i in "${!EXPERIMENTS[@]}"; do
  config="${EXPERIMENTS[$i]}"
  echo "============================================"
  echo "[$((i+1))/${#EXPERIMENTS[@]}] Running $config"
  echo "============================================"
  torchrun --nproc_per_node=8 train.py --config "$config"
  echo ""
done

echo "All 10 full-FF++ EMA runs complete!"
