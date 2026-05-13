#!/usr/bin/env python3
"""Threshold sensitivity: accuracy/balanced accuracy/TPR/FPR vs threshold."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--output-dir", default="figures/threshold")
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--variant", default="raw")
    parser.add_argument("--seed", default="s42")
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf", "dfd"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0, 1, 201)

    for domain in args.domains:
        npz_path = Path(args.features_dir) / args.backbone / f"{args.variant}_{args.seed}" / f"{domain}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        probs, labels = data["probs"], data["labels"]

        accs, bal_accs, tprs, fprs = [], [], [], []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            accs.append((preds == labels).mean())
            bal_accs.append(balanced_accuracy_score(labels, preds))
            real_mask = labels == 0
            fake_mask = labels == 1
            tprs.append(preds[fake_mask].mean() if fake_mask.sum() > 0 else 0)
            fprs.append(preds[real_mask].mean() if real_mask.sum() > 0 else 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, accs, label="Accuracy")
        ax.plot(thresholds, bal_accs, label="Balanced Accuracy")
        ax.plot(thresholds, tprs, label="TPR (Fake Recall)", linestyle="--")
        ax.plot(thresholds, fprs, label="FPR (Real → Fake)", linestyle="--")
        ax.axvline(0.5, color="gray", linestyle=":", label="Default threshold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title(f"{args.backbone} — {domain.upper()} Threshold Sensitivity")
        ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        out_path = output_dir / f"{args.backbone}_{domain}_threshold_curve.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
