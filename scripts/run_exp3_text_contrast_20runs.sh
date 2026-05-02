#!/bin/bash
set -e
export PYTHONPATH=src

START=${1:-1}

EXPERIMENTS=(
  "configs/exp3_itc_freeze_all_s42.yaml"
  "configs/exp3_itc_freeze_all_s123.yaml"
  "configs/exp3_itc_freeze_all_s7.yaml"
  "configs/exp3_itc_freeze_all_s999.yaml"
  "configs/exp3_itc_freeze_all_s2048.yaml"
  "configs/exp3_itc_freeze_partial_s42.yaml"
  "configs/exp3_itc_freeze_partial_s123.yaml"
  "configs/exp3_itc_freeze_partial_s7.yaml"
  "configs/exp3_itc_freeze_partial_s999.yaml"
  "configs/exp3_itc_freeze_partial_s2048.yaml"
  "configs/exp3_itc_train_all_s42.yaml"
  "configs/exp3_itc_train_all_s123.yaml"
  "configs/exp3_itc_train_all_s7.yaml"
  "configs/exp3_itc_train_all_s999.yaml"
  "configs/exp3_itc_train_all_s2048.yaml"
  "configs/exp3_prompt_init_s42.yaml"
  "configs/exp3_prompt_init_s123.yaml"
  "configs/exp3_prompt_init_s7.yaml"
  "configs/exp3_prompt_init_s999.yaml"
  "configs/exp3_prompt_init_s2048.yaml"
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

echo "All 20 exp3 text contrast runs complete!"
