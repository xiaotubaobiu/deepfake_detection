#!/usr/bin/env bash
set -euo pipefail
conda run -n deepfake-detection torchrun --nproc_per_node=8 train.py --config configs/exp1_efficientnet.yaml
