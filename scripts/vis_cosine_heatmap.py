#!/usr/bin/env python3
"""Cosine similarity heatmap between class centroids across domains."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output-dir", default="figures/cosine_heatmap")
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids = {}
    for domain in args.domains:
        npz_path = Path(args.features_dir) / args.backbone / f"{args.variant}_{args.seed}" / f"{domain}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        features, labels = data["features"], data["labels"]
        centroids[f"{domain}_real"] = features[labels == 0].mean(axis=0)
        centroids[f"{domain}_fake"] = features[labels == 1].mean(axis=0)

    if len(centroids) < 2:
        print("Not enough data.")
        return

    names = list(centroids.keys())
    vectors = np.stack([centroids[n] for n in names])
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim_matrix = vectors_norm @ vectors_norm.T

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", xticklabels=names, yticklabels=names,
                cmap="RdYlBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"{args.backbone} — Cross-Domain Cosine Similarity")
    fig.tight_layout()
    out_path = output_dir / f"{args.backbone}_cross_domain_cosine.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
