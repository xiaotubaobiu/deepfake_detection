#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
START=${1:-1}
CONFIGS=(
  configs/exp3_rawcls_normprompt_fullffpp_ema_s42.yaml
  configs/exp3_rawcls_normprompt_fullffpp_ema_s123.yaml
  configs/exp3_rawcls_normprompt_fullffpp_ema_s7.yaml
  configs/exp3_rawcls_normprompt_fullffpp_ema_s999.yaml
  configs/exp3_rawcls_normprompt_fullffpp_ema_s2048.yaml
)

for i in "${!CONFIGS[@]}"; do
  run_idx=$((i + 1))
  if (( run_idx < START )); then
    continue
  fi
  cfg="${CONFIGS[$i]}"
  echo "===== [$run_idx/${#CONFIGS[@]}] $cfg ====="
  torchrun --nproc_per_node=8 train.py --config "$cfg"
done
