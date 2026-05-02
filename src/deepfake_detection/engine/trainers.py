from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

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


def prompt_contrast_step(model, batch, device, beta=0.1):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    cls_logits, prompt_logits = model(images)
    cls_loss = F.cross_entropy(cls_logits, labels)
    prompt_loss = F.cross_entropy(prompt_logits, labels)
    total_loss = cls_loss + beta * prompt_loss
    return total_loss, cls_logits, prompt_logits


def prompt_itc_step(model, batch, device, lambda_itc=0.1, temperature=0.07):
    from deepfake_detection.losses.contrastive import image_text_contrastive_loss
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    raw_model = model.module if hasattr(model, "module") else model
    cls_logits, _, image_features, text_features = raw_model.forward_with_features(images)
    cls_loss = F.cross_entropy(cls_logits, labels)
    itc_loss = image_text_contrastive_loss(image_features, text_features, labels, temperature)
    total_loss = cls_loss + lambda_itc * itc_loss
    return total_loss, cls_logits


def bgface_contrast_step(model, batch, device, lambda_align=0.1, temperature=0.07):
    bg_images = batch["background"].to(device)
    real_face = batch["real_face"].to(device)
    fake_face = batch["fake_face"].to(device)
    labels = batch["label"].to(device)

    cls_logits = model.forward_classification(fake_face)
    cls_loss = F.cross_entropy(cls_logits, labels)

    from deepfake_detection.losses.contrastive import infonce_bg_face_loss
    bg_proj, real_face_proj = model.forward_contrastive(bg_images, real_face)
    _, fake_face_proj = model.forward_contrastive(bg_images, fake_face)
    contrastive_loss = infonce_bg_face_loss(bg_proj, real_face_proj, fake_face_proj, temperature)

    total_loss = cls_loss + lambda_align * contrastive_loss
    return total_loss, cls_logits


def run_train_epoch(model, dataloader, optimizer, scaler, device, cfg, ema=None):
    model.train()
    total_loss = 0
    n = 0
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    is_prompt_itc = loss_name == "cross_entropy_plus_prompt_itc"
    is_bgface = loss_name == "cross_entropy_plus_bgface_contrast"
    lambda_align = cfg.get("loss", {}).get("lambda_align", 0.1)
    lambda_itc = cfg.get("loss", {}).get("lambda_itc", 0.1)
    temperature = cfg.get("loss", {}).get("temperature", 0.07)
    beta = cfg.get("loss", {}).get("beta", 0.1)
    for batch in dataloader:
        optimizer.zero_grad()
        with autocast(enabled=True):
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            elif is_prompt_itc:
                loss, _ = prompt_itc_step(model, batch, device, lambda_itc, temperature)
            elif is_prompt:
                loss, _, _ = prompt_contrast_step(model, batch, device, beta)
            elif is_bgface:
                loss, _ = bgface_contrast_step(model, batch, device, lambda_align, temperature)
            else:
                loss, _ = classification_step(model, batch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def run_eval_epoch(model, dataloader, device, cfg):
    model.eval()
    all_rows = []
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    is_prompt_itc = loss_name == "cross_entropy_plus_prompt_itc"
    alpha = cfg.get("loss", {}).get("alpha", 0.3)
    for batch in dataloader:
        images = batch["image"].to(device) if not is_contrastive else batch["background"].to(device)
        labels = batch["label"]
        video_ids = batch["video_id"]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            output = model(images.float())
        if is_prompt and not is_prompt_itc:
            cls_logits, prompt_logits = output
            cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]
            prompt_prob = torch.softmax(prompt_logits, dim=1)[:, 1]
            probs = ((1 - alpha) * cls_prob + alpha * prompt_prob).cpu().numpy()
        else:
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for i in range(len(probs)):
            all_rows.append({
                "video_id": video_ids[i] if isinstance(video_ids, list) else video_ids[i],
                "score": float(probs[i]),
                "label": int(labels[i]),
            })

    if torch.distributed.is_initialized():
        gathered_rows = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_rows, all_rows)
        all_rows = [row for rows in gathered_rows for row in rows]

    labels, scores = aggregate_video_predictions(all_rows)
    if len(set(labels)) < 2:
        return {"auc": 0.0, "eer": 0.0, "acc": 0.0, "loss": 0.0}
    auc = compute_auc(labels, scores)
    eer = compute_eer(labels, scores)
    acc = compute_acc(labels, scores)
    return {"auc": auc, "eer": eer, "acc": acc}
