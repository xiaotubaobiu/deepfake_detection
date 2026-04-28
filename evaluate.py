from __future__ import annotations

import argparse
import os

import yaml
import torch

from deepfake_detection.engine.ddp import init_ddp, is_main_process
from deepfake_detection.engine.trainers import run_eval_epoch
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_eval_loader
from train import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", default="both", choices=["ffpp", "cdf", "both"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    distributed = torch.distributed.is_initialized()

    if args.domain in ("ffpp", "both"):
        loader = build_eval_loader(cfg, domain="ffpp", distributed=distributed)
        metrics = run_eval_epoch(model, loader, device, cfg)
        if is_main_process():
            print(f"FF++: AUC={metrics['auc']:.4f} EER={metrics['eer']:.4f} ACC={metrics['acc']:.4f}")

    if args.domain in ("cdf", "both"):
        loader = build_eval_loader(cfg, domain="cdf", distributed=distributed)
        metrics = run_eval_epoch(model, loader, device, cfg)
        if is_main_process():
            print(f"CDF: AUC={metrics['auc']:.4f} EER={metrics['eer']:.4f} ACC={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
