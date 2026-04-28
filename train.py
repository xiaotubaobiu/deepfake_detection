from __future__ import annotations

import argparse
import os
import time

import yaml
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from deepfake_detection.engine.ddp import init_ddp, is_main_process, barrier
from deepfake_detection.engine.trainers import run_train_epoch, run_eval_epoch
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_train_loader, build_eval_loader


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        merged = {**base, **cfg}
        for k in set(list(base.keys()) + list(cfg.keys())):
            if k in base and k in cfg and isinstance(base[k], dict) and isinstance(cfg[k], dict):
                merged[k] = {**base[k], **cfg[k]}
        cfg = merged
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp_name = cfg.get("experiment_name", "default")
    epochs = cfg["train"].get("epochs", 30)
    lr = cfg["train"].get("lr", 1e-4)
    weight_decay = cfg["train"].get("weight_decay", 1e-5)
    patience = cfg["train"].get("patience", 5)
    output_dir = cfg["train"].get("output_dir", "outputs")
    seed = cfg["train"].get("seed", 42)

    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_main_process():
        print(f"Experiment: {exp_name}, Device: {device}, Config: {args.config}")

    model = build_model(cfg["model"]).to(device)
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    train_loader = build_train_loader(cfg, distributed=torch.distributed.is_initialized())
    ffpp_eval_loader = build_eval_loader(cfg, domain="ffpp", distributed=torch.distributed.is_initialized())
    cdf_eval_loader = build_eval_loader(cfg, domain="cdf", distributed=torch.distributed.is_initialized())

    best_auc = 0
    patience_counter = 0
    save_dir = os.path.join(output_dir, exp_name)

    for epoch in range(epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, optimizer, scaler, device, cfg)
        ffpp_metrics = run_eval_epoch(model, ffpp_eval_loader, device, cfg)
        elapsed = time.time() - t0

        if is_main_process():
            print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) loss={train_loss:.4f} "
                  f"ffpp_auc={ffpp_metrics['auc']:.4f} ffpp_eer={ffpp_metrics['eer']:.4f}")

        if ffpp_metrics["auc"] > best_auc:
            best_auc = ffpp_metrics["auc"]
            patience_counter = 0
            if is_main_process():
                os.makedirs(save_dir, exist_ok=True)
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"epoch": epoch, "model_state_dict": state, "auc": best_auc}, os.path.join(save_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience and is_main_process():
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        barrier()

    if is_main_process():
        cdf_metrics = run_eval_epoch(model, cdf_eval_loader, device, cfg)
        print(f"\nFinal CDF eval: auc={cdf_metrics['auc']:.4f} eer={cdf_metrics['eer']:.4f} acc={cdf_metrics['acc']:.4f}")
        print(f"Best FF++ AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
