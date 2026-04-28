from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from deepfake_detection.engine.metrics import compute_auc, compute_eer, compute_acc, aggregate_video_predictions


def classification_step(model, batch, device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def contrastive_step(model, batch, device, lambda_align=0.1, temperature=0.07):
    from deepfake_detection.losses.contrastive import same_frame_contrastive_loss
    bg = batch["background"].to(device)
    real_face = batch["real_face"].to(device)
    fake_face = batch["fake_face"].to(device)
    labels = batch["label"].to(device)
    cls_logits = model.forward_classification(bg)
    cls_loss = F.cross_entropy(cls_logits, labels)
    bg_feat, real_feat, fake_feat = model.forward_contrastive(bg, real_face, fake_face)
    nce_loss = same_frame_contrastive_loss(bg_feat, real_feat, fake_feat, temperature)
    total_loss = cls_loss + lambda_align * nce_loss
    return total_loss, cls_logits


def run_train_epoch(model, dataloader, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0
    n = 0
    is_contrastive = cfg.get("loss", {}).get("name") == "cross_entropy_plus_contrastive"
    lambda_align = cfg.get("loss", {}).get("lambda_align", 0.1)
    temperature = cfg.get("loss", {}).get("temperature", 0.07)
    for batch in dataloader:
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=True):
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            else:
                loss, _ = classification_step(model, batch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def run_eval_epoch(model, dataloader, device, cfg):
    model.eval()
    all_rows = []
    is_contrastive = cfg.get("loss", {}).get("name") == "cross_entropy_plus_contrastive"
    for batch in dataloader:
        images = batch["image"].to(device) if not is_contrastive else batch["background"].to(device)
        labels = batch["label"]
        video_ids = batch["video_id"]
        with autocast(device_type="cuda", enabled=True):
            logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for i in range(len(probs)):
            all_rows.append({
                "video_id": video_ids[i] if isinstance(video_ids, list) else video_ids[i],
                "score": float(probs[i]),
                "label": int(labels[i]),
            })
    labels, scores = aggregate_video_predictions(all_rows)
    if len(set(labels)) < 2:
        return {"auc": 0.0, "eer": 0.0, "acc": 0.0, "loss": 0.0}
    auc = compute_auc(labels, scores)
    eer = compute_eer(labels, scores)
    acc = compute_acc(labels, scores)
    return {"auc": auc, "eer": eer, "acc": acc}
