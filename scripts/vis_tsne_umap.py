#!/usr/bin/env python3
"""t-SNE / UMAP visualization of features colored by label or norm magnitude."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_embedding(embedding_2d, labels, norms, title, output_path, color_by="label"):
    fig, ax = plt.subplots(figsize=(8, 7))
    if color_by == "label":
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.3, s=2)
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', label='Real'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', label='Fake'),
        ])
    elif color_by == "norm":
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=norms, cmap="viridis", alpha=0.3, s=2)
        plt.colorbar(scatter, ax=ax, label="Feature Norm")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output-dir", default="figures/tsne")
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features, all_labels, all_norms = [], [], []
    for domain in args.domains:
        npz_path = Path(args.features_dir) / args.backbone / f"{args.variant}_{args.seed}" / f"{domain}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        feats = data["features"]
        n = args.max_samples // len(args.domains)
        if len(feats) > n:
            idx = np.random.choice(len(feats), n, replace=False)
            feats, labels, norms = feats[idx], data["labels"][idx], data["norms"][idx]
        else:
            labels, norms = data["labels"], data["norms"]
        all_features.append(feats)
        all_labels.append(labels)
        all_norms.append(norms)

    if not all_features:
        print("No data found.")
        return

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    norms = np.concatenate(all_norms)
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_normalized = features / feat_norms

    for feat_type, feats in [("raw", features), ("normalized", features_normalized)]:
        print(f"Computing {args.method} for {feat_type} ({feats.shape})...")
        if args.method == "tsne":
            embedding = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats)
        else:
            import umap
            embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)

        for color_by in ["label", "norm"]:
            out_path = output_dir / f"{args.backbone}_{args.variant}_{feat_type}_{color_by}.pdf"
            plot_embedding(embedding, labels, norms, f"{args.backbone} ({feat_type}) — {color_by}", out_path, color_by)
            print(f"  Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
