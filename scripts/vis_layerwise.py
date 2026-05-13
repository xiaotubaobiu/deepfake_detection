#!/usr/bin/env python3
"""Layer-wise analysis: norm-AUC and direction-AUC per CLIP transformer block."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

NUM_BLOCKS = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features/clip")
    parser.add_argument("--output-dir", default="figures/layerwise")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = Path(args.features_dir) / f"{args.variant}_{args.seed}_layers"

    for domain in args.domains:
        norm_aucs, dir_aucs = [], []
        for i in range(NUM_BLOCKS):
            npz_path = layers_dir / f"{domain}_layer{i}.npz"
            if not npz_path.exists():
                norm_aucs.append(np.nan)
                dir_aucs.append(np.nan)
                continue
            data = np.load(npz_path, allow_pickle=True)
            features, labels = data["features"], data["labels"]
            norms = np.linalg.norm(features, axis=1)
            norm_aucs.append(roc_auc_score(labels, norms) if len(set(labels)) > 1 else np.nan)
            real_mean = features[labels == 0].mean(axis=0)
            fake_mean = features[labels == 1].mean(axis=0)
            direction = fake_mean - real_mean
            direction /= np.linalg.norm(direction) + 1e-8
            scores = features @ direction
            dir_aucs.append(roc_auc_score(labels, scores) if len(set(labels)) > 1 else np.nan)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(NUM_BLOCKS), norm_aucs, "o-", label="Norm AUC", color="#F44336")
        ax.plot(range(NUM_BLOCKS), dir_aucs, "s-", label="Direction AUC", color="#2196F3")
        ax.set_xlabel("Transformer Block Index"); ax.set_ylabel("AUC")
        ax.set_title(f"CLIP Layer-wise Analysis — {domain.upper()}")
        ax.set_xticks(range(NUM_BLOCKS)); ax.set_ylim(0.4, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / f"clip_{domain}_layerwise.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
