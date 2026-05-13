#!/usr/bin/env python3
"""Generate and optionally launch all training configs for multi-backbone experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BACKBONES = {
    "efficientnet": {
        "base_config": "experiments/configs/efficientnet_base.yaml",
        "model_name": "efficientnet_b0",
        "lr": None,
    },
    "dinov2": {
        "base_config": "experiments/configs/dinov2_base.yaml",
        "model_name": "dinov2_vitb14",
        "lr": None,
    },
    "xception": {
        "base_config": "experiments/configs/xception_base.yaml",
        "model_name": "xception",
        "lr": None,
    },
}

SEEDS = [42, 7, 123, 999, 2048]
VARIANTS = [
    ("raw", False),
    ("normtrain", True),
]


def generate_config(backbone, variant, seed, model_name, normalize, lr, base_config, output_dir):
    experiment_name = f"multibb_{variant}_s{seed}_{backbone}"
    config_text = f"""_base_: {base_config}
experiment_name: {experiment_name}
model:
  name: {model_name}
  normalize_features: {str(normalize).lower()}
train:
  lr: {lr}
  seed: {seed}
"""
    config_path = output_dir / f"{experiment_name}.yaml"
    config_path.write_text(config_text)
    return config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--efficientnet-lr", type=float, required=True)
    parser.add_argument("--dinov2-lr", type=float, required=True)
    parser.add_argument("--xception-lr", type=float, required=True)
    parser.add_argument("--run", action="store_true", help="Launch training (otherwise just generate configs)")
    parser.add_argument("--config-dir", default="experiments/configs/multibb")
    args = parser.parse_args()

    BACKBONES["efficientnet"]["lr"] = args.efficientnet_lr
    BACKBONES["dinov2"]["lr"] = args.dinov2_lr
    BACKBONES["xception"]["lr"] = args.xception_lr

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for bb_name, bb_cfg in BACKBONES.items():
        for variant_name, normalize in VARIANTS:
            for seed in SEEDS:
                path = generate_config(
                    backbone=bb_name, variant=variant_name, seed=seed,
                    model_name=bb_cfg["model_name"], normalize=normalize,
                    lr=bb_cfg["lr"], base_config=bb_cfg["base_config"],
                    output_dir=config_dir,
                )
                configs.append(path)

    print(f"Generated {len(configs)} configs in {config_dir}/")
    for c in sorted(configs):
        print(f"  {c.name}")

    if args.run:
        for i, config_path in enumerate(sorted(configs)):
            print(f"\n[{i+1}/{len(configs)}] Training {config_path.name}")
            subprocess.run(
                ["torchrun", "--nproc_per_node=8", "train.py", "--config", str(config_path)],
                check=True,
            )
        print("All training complete.")


if __name__ == "__main__":
    main()
