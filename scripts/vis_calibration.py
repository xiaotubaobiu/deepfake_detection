#!/usr/bin/env python3
"""Calibration analysis: ECE and reliability diagrams."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(probs[mask].mean())
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    ece /= max(len(probs), 1)
    return ece, np.array(bin_accs), np.array(bin_confs)


def plot_reliability(bin_accs, bin_confs, ece, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#2196F3", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output-dir", default="figures/calibration")
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        npz_path = Path(args.features_dir) / args.backbone / f"{args.variant}_{args.seed}" / f"{domain}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        ece, bin_accs, bin_confs = compute_ece(data["probs"], data["labels"])
        out_path = output_dir / f"{args.backbone}_{domain}_reliability.pdf"
        plot_reliability(bin_accs, bin_confs, ece, f"{args.backbone} — {domain.upper()} Reliability Diagram", out_path)
        print(f"{domain}: ECE={ece:.4f} → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
