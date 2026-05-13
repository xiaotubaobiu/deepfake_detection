#!/usr/bin/env python3
"""Summarize AUC results across all backbones, variants, and seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BACKBONES = ["clip", "efficientnet", "dinov2", "xception"]
SEEDS = ["s42", "s7", "s123", "s999", "s2048"]
VARIANTS = ["raw", "normtrain"]
DOMAINS = ["ffpp", "cdf", "dfd"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="features")
    args = parser.parse_args()

    print(f"{'Backbone':<15} {'Variant':<12} {'Seeds':>5} {'FF++':>12} {'CDF':>12} {'DFD':>12}")
    print("-" * 72)

    for backbone in BACKBONES:
        for variant in VARIANTS:
            aucs_by_domain = {d: [] for d in DOMAINS}
            for seed in SEEDS:
                seed_dir = Path(args.features_dir) / backbone / f"{variant}_{seed}"
                if not seed_dir.exists():
                    continue
                for domain in DOMAINS:
                    npz_path = seed_dir / f"{domain}.npz"
                    if not npz_path.exists():
                        continue
                    data = np.load(npz_path, allow_pickle=True)
                    if len(set(data["labels"])) > 1:
                        aucs_by_domain[domain].append(roc_auc_score(data["labels"], data["probs"]))

            n_seeds = max((len(v) for v in aucs_by_domain.values()), default=0)
            if n_seeds == 0:
                continue
            ff = f"{np.mean(aucs_by_domain['ffpp']):.4f}±{np.std(aucs_by_domain['ffpp']):.4f}" if aucs_by_domain["ffpp"] else "—"
            cd = f"{np.mean(aucs_by_domain['cdf']):.4f}±{np.std(aucs_by_domain['cdf']):.4f}" if aucs_by_domain["cdf"] else "—"
            df = f"{np.mean(aucs_by_domain['dfd']):.4f}±{np.std(aucs_by_domain['dfd']):.4f}" if aucs_by_domain["dfd"] else "—"
            print(f"{backbone:<15} {variant:<12} {n_seeds:>5} {ff:>12} {cd:>12} {df:>12}")


if __name__ == "__main__":
    main()
