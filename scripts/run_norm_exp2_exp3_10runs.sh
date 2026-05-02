#!/bin/bash
set -e
export PYTHONPATH=src

START=${1:-1}

EXPERIMENTS=(
  "configs/exp2_norm_fullffpp_ema_s42.yaml"
  "configs/exp2_norm_fullffpp_ema_s123.yaml"
  "configs/exp2_norm_fullffpp_ema_s7.yaml"
  "configs/exp2_norm_fullffpp_ema_s999.yaml"
  "configs/exp2_norm_fullffpp_ema_s2048.yaml"
  "configs/exp3_norm_fullffpp_ema_s42.yaml"
  "configs/exp3_norm_fullffpp_ema_s123.yaml"
  "configs/exp3_norm_fullffpp_ema_s7.yaml"
  "configs/exp3_norm_fullffpp_ema_s999.yaml"
  "configs/exp3_norm_fullffpp_ema_s2048.yaml"
)

for i in "${!EXPERIMENTS[@]}"; do
  run_no=$((i+1))
  if [ "$run_no" -lt "$START" ]; then
    continue
  fi
  config="${EXPERIMENTS[$i]}"
  echo "============================================"
  echo "[$run_no/${#EXPERIMENTS[@]}] Running $config"
  echo "============================================"
  torchrun --nproc_per_node=8 train.py --config "$config"
  echo ""
done

echo "Normalized exp2/exp3 10 runs complete!"
