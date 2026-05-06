#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 123 7 999 2048)
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
LOG_EVERY="${LOG_EVERY:-100}"

train_one() {
  local config="$1"
  PYTHONPATH=src torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py --config "${config}"
}

eval_raw_checkpoint() {
  local seed="$1"
  local checkpoint="$2"
  PYTHONPATH=src torchrun --nproc_per_node="${NPROC_PER_NODE}" evaluate_norm_correction.py \
    --config "experiments/configs/norm_shortcut_raw_s${seed}.yaml" \
    --checkpoint "${checkpoint}" \
    --mode raw,feature_norm,partial,fusion \
    --splits ffpp,cdf \
    --alphas 0,0.25,0.5,0.75,1 \
    --betas 0,0.25,0.5,0.75,1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --log-every "${LOG_EVERY}" \
    --write-sample-rows \
    --output-dir "experiments/outputs/norm_shortcut_eval/raw_s${seed}"
}

eval_normtrain_checkpoint() {
  local seed="$1"
  local checkpoint="$2"
  PYTHONPATH=src torchrun --nproc_per_node="${NPROC_PER_NODE}" evaluate_norm_correction.py \
    --config "experiments/configs/norm_shortcut_normtrain_s${seed}.yaml" \
    --checkpoint "${checkpoint}" \
    --mode feature_norm \
    --splits ffpp,cdf \
    --alphas 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --log-every "${LOG_EVERY}" \
    --write-sample-rows \
    --output-dir "experiments/outputs/norm_shortcut_eval/normtrain_s${seed}"
}

case "${1:-train}" in
  train)
    for seed in "${SEEDS[@]}"; do
      train_one "experiments/configs/norm_shortcut_raw_s${seed}.yaml"
      train_one "experiments/configs/norm_shortcut_normtrain_s${seed}.yaml"
    done
    ;;
  eval-raw)
    eval_raw_checkpoint "${2:?seed required}" "${3:?checkpoint required}"
    ;;
  eval-normtrain)
    eval_normtrain_checkpoint "${2:?seed required}" "${3:?checkpoint required}"
    ;;
  *)
    echo "Usage: $0 train | eval-raw <seed> <checkpoint> | eval-normtrain <seed> <checkpoint>" >&2
    exit 2
    ;;
esac
