#!/bin/bash
set -e
export PYTHONPATH=src

START=${1:-1}

EXPERIMENTS=(
  "configs/exp3_itc_freeze_all_l003_s42.yaml"
  "configs/exp3_itc_freeze_all_l003_s123.yaml"
  "configs/exp3_itc_freeze_all_l003_s7.yaml"
  "configs/exp3_itc_freeze_all_l003_s999.yaml"
  "configs/exp3_itc_freeze_all_l003_s2048.yaml"
  "configs/exp3_itc_freeze_all_l001_s42.yaml"
  "configs/exp3_itc_freeze_all_l001_s123.yaml"
  "configs/exp3_itc_freeze_all_l001_s7.yaml"
  "configs/exp3_itc_freeze_all_l001_s999.yaml"
  "configs/exp3_itc_freeze_all_l001_s2048.yaml"
  "configs/exp3_itc_partial_l001_s42.yaml"
  "configs/exp3_itc_partial_l001_s123.yaml"
  "configs/exp3_itc_partial_l001_s7.yaml"
  "configs/exp3_itc_partial_l001_s999.yaml"
  "configs/exp3_itc_partial_l001_s2048.yaml"
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

echo "Conservative text-alignment runs complete!"
