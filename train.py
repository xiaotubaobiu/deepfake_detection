from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

import math

import yaml
import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from deepfake_detection.engine.ddp import init_ddp, is_main_process, barrier
from deepfake_detection.engine.trainers import run_train_epoch, run_eval_epoch
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_train_loader, build_eval_loader, build_val_loader


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


def setup_experiment_log(exp_name, cfg, output_dir="outputs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, exp_name, ts)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "experiment.log")
    config_path = os.path.join(run_dir, "config.yaml")
    meta_path = os.path.join(run_dir, "meta.json")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    meta = {
        "experiment_name": exp_name,
        "start_time": ts,
        "config_file": cfg.get("_base_", "N/A"),
        "model": cfg.get("model", {}).get("name", "N/A"),
        "per_gpu_batch": cfg.get("train", {}).get("per_gpu_batch", "N/A"),
        "epochs": cfg.get("train", {}).get("epochs", "N/A"),
        "lr": cfg.get("train", {}).get("lr", "N/A"),
        "loss": cfg.get("loss", {}).get("name", "N/A"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir, log_path


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._file = open(log_path, "a")

    def log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        for k, v in state.items():
            self.shadow[k].copy_(v)


def build_scheduler(optimizer, epochs, warmup_epochs):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # epoch 0 → 1/warmup, not 0
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)
    ema_decay = cfg["train"].get("ema_decay", 0)
    output_dir = cfg["train"].get("output_dir", "outputs")
    seed = cfg["train"].get("seed", 42)

    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- 全局确定性种子 ----
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    logger = None
    run_dir = os.path.join(output_dir, exp_name)
    if is_main_process():
        run_dir, log_path = setup_experiment_log(exp_name, cfg, output_dir)
        logger = Logger(log_path)
        logger.log(f"Experiment: {exp_name}, Device: {device}, Config: {args.config}")
        logger.log(f"Per-GPU batch: {cfg['train'].get('per_gpu_batch')}, Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}, EMA: {ema_decay}")
        logger.log(f"Log dir: {run_dir}")
    if torch.distributed.is_initialized():
        run_dir_holder = [run_dir]
        torch.distributed.broadcast_object_list(run_dir_holder, src=0)
        run_dir = run_dir_holder[0]
    barrier()

    model = build_model(cfg["model"]).to(device)
    init_ckpt_path = cfg.get("train", {}).get("init_checkpoint")
    if init_ckpt_path:
        ckpt = torch.load(init_ckpt_path, map_location=device)
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main_process():
            print(f"Initialized from {init_ckpt_path}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if torch.distributed.is_initialized():
        loss_name = cfg.get("loss", {}).get("name", "")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=loss_name == "cross_entropy_plus_prompt")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name == "cross_entropy_plus_bgface_contrast":
        from deepfake_detection.data.builders import build_bgface_train_loader
        train_loader = build_bgface_train_loader(cfg, distributed=torch.distributed.is_initialized())
    else:
        train_loader = build_train_loader(cfg, distributed=torch.distributed.is_initialized())
    val_loader = build_val_loader(cfg, distributed=torch.distributed.is_initialized())
    ffpp_eval_loader = build_eval_loader(cfg, domain="ffpp", distributed=torch.distributed.is_initialized())
    cdf_eval_loader = build_eval_loader(cfg, domain="cdf", distributed=torch.distributed.is_initialized())

    if logger:
        logger.log(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, "
                   f"FF++ test: {len(ffpp_eval_loader.dataset)}, CDF test: {len(cdf_eval_loader.dataset)}")

    best_auc = 0
    patience_counter = 0
    best_epoch = 0
    save_dir = run_dir

    for epoch in range(epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, optimizer, scaler, device, cfg, ema=ema)

        # Evaluate with EMA if enabled
        if ema is not None:
            ema.apply(model)
        val_metrics = run_eval_epoch(model, val_loader, device, cfg)
        if ema is not None:
            ema.restore(model)
        elapsed = time.time() - t0

        if logger:
            logger.log(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) loss={train_loss:.4f} "
                       f"val_auc={val_metrics['auc']:.4f} val_eer={val_metrics['eer']:.4f} val_acc={val_metrics['acc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            patience_counter = 0
            if logger:
                os.makedirs(save_dir, exist_ok=True)
                if ema is not None:
                    ema.apply(model)
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"epoch": epoch, "model_state_dict": state, "auc": best_auc,
                            "eer": val_metrics["eer"], "acc": val_metrics["acc"]},
                           os.path.join(save_dir, "best_model.pth"))
                if ema is not None:
                    ema.restore(model)
                logger.log(f"  -> Best model saved (val AUC={best_auc:.4f})")
        else:
            patience_counter += 1

        should_stop = patience_counter >= patience
        if should_stop:
            if logger:
                logger.log(f"Early stopping at epoch {epoch+1}")
            barrier()
            break

        scheduler.step()
        barrier()

    barrier()
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])

    if logger:
        logger.log("Loading best model for final test evaluation...")

    val_final = run_eval_epoch(model, val_loader, device, cfg)
    ffpp_metrics = run_eval_epoch(model, ffpp_eval_loader, device, cfg)
    cdf_metrics = run_eval_epoch(model, cdf_eval_loader, device, cfg)

    if logger:
        logger.log(f"Val (best): auc={val_final['auc']:.4f} eer={val_final['eer']:.4f} acc={val_final['acc']:.4f}")
        logger.log(f"FF++ test: auc={ffpp_metrics['auc']:.4f} eer={ffpp_metrics['eer']:.4f} acc={ffpp_metrics['acc']:.4f}")
        logger.log(f"CDF test: auc={cdf_metrics['auc']:.4f} eer={cdf_metrics['eer']:.4f} acc={cdf_metrics['acc']:.4f}")

        logger.log(f"Best epoch: {best_epoch}, Best val AUC: {best_auc:.4f}")

        end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = os.path.join(os.path.dirname(logger.log_path), "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["end_time"] = end_ts
        meta["best_epoch"] = best_epoch
        meta["best_val_auc"] = best_auc
        meta["final_val_auc"] = val_final["auc"]
        meta["final_val_eer"] = val_final["eer"]
        meta["final_val_acc"] = val_final["acc"]
        meta["ffpp_test_auc"] = ffpp_metrics["auc"]
        meta["ffpp_test_eer"] = ffpp_metrics["eer"]
        meta["ffpp_test_acc"] = ffpp_metrics["acc"]
        meta["cdf_test_auc"] = cdf_metrics["auc"]
        meta["cdf_test_eer"] = cdf_metrics["eer"]
        meta["cdf_test_acc"] = cdf_metrics["acc"]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.log(f"Meta saved to {meta_path}")

        results_path = os.path.join(save_dir, "results_summary.txt")
        with open(results_path, "w") as f:
            f.write(f"Exp: {exp_name}\n")
            f.write(f"Model: {cfg.get('model', {}).get('name', 'N/A')}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Per-GPU batch: {cfg['train'].get('per_gpu_batch')}\n")
            f.write(f"Val:  AUC={val_final['auc']:.4f} EER={val_final['eer']:.4f} ACC={val_final['acc']:.4f}\n")
            f.write(f"FF++: AUC={ffpp_metrics['auc']:.4f} EER={ffpp_metrics['eer']:.4f} ACC={ffpp_metrics['acc']:.4f}\n")
            f.write(f"CDF:  AUC={cdf_metrics['auc']:.4f} EER={cdf_metrics['eer']:.4f} ACC={cdf_metrics['acc']:.4f}\n")
        logger.log(f"Results saved to {results_path}")
        logger.close()


if __name__ == "__main__":
    main()
