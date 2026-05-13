#!/usr/bin/env python3
"""Plot feature norm distributions: real vs fake, per backbone, per dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_norm_histogram(npz_path, output_path, title):
    data = np.load(npz_path, allow_pickle=True)
    norms = data["norms"]
    labels = data["labels"]
    real_norms = norms[labels == 0]
    fake_norms = norms[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(real_norms, ax=ax, label="Real", color="#2196F3", fill=True, alpha=0.3)
    sns.kdeplot(fake_norms, ax=ax, label="Fake", color="#F44336", fill=True, alpha=0.3)
    ax.set_xlabel("Feature L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output-dir", default="figures/norm_histogram")
    parser.add_argument("--backbones", nargs="+", default=["clip", "efficientnet", "dinov2", "xception"])
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for backbone in args.backbones:
        for domain in args.domains:
            npz_path = Path(args.features_dir) / backbone / f"{args.variant}_{args.seed}" / f"{domain}.npz"
            if not npz_path.exists():
                print(f"Skip {npz_path} (not found)")
                continue
            out_path = output_dir / f"{backbone}_{domain}_norm_hist.pdf"
            plot_norm_histogram(npz_path, out_path, f"{backbone} — {domain.upper()} Feature Norm Distribution")
            print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
