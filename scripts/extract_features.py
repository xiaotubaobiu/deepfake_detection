#!/usr/bin/env python3
"""Extract features, norms, labels, predictions from trained checkpoints to .npz files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_eval_loader


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = Path(config_path).parent.parent.parent / cfg.pop("_base_")
        if not base_path.exists():
            base_path = Path(config_path).parent / cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        for k, v in cfg.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
        return base
    return cfg


DOMAINS = ["ffpp", "cdf", "dfd"]


@torch.no_grad()
def extract_domain(model, dataloader, device):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    all_features, all_norms, all_labels, all_probs, all_video_ids = [], [], [], [], []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        features = raw_model.extract_features(images.float()).float()
        logits = raw_model.classifier(features)
        prob_fake = torch.softmax(logits, dim=1)[:, 1]
        norms = features.norm(dim=-1)
        all_features.append(features.cpu().numpy())
        all_norms.append(norms.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_probs.append(prob_fake.cpu().numpy())
        all_video_ids.extend(batch["video_id"])
    return {
        "features": np.concatenate(all_features),
        "norms": np.concatenate(all_norms),
        "labels": np.concatenate(all_labels),
        "probs": np.concatenate(all_probs),
        "video_ids": np.array(all_video_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        print(f"Extracting {domain}...")
        loader = build_eval_loader(cfg, domain=domain, distributed=False)
        data = extract_domain(model, loader, args.device)
        out_path = output_dir / f"{domain}.npz"
        np.savez(out_path, **data)
        print(f"  Saved {out_path}: {data['features'].shape[0]} samples, dim={data['features'].shape[1]}")

    print("Done.")


if __name__ == "__main__":
    main()
