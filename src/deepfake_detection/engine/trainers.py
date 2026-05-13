from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from deepfake_detection.engine.metrics import (
    compute_auc,
    compute_eer,
    compute_acc,
    merge_video_aggregates,
    update_video_aggregate,
    video_aggregates_to_predictions,
)
from deepfake_detection.evaluation.sample_eval import (
    build_sample_rows,
    gather_rows_across_ranks,
    summarize_sample_rows,
    video_metrics_from_rows,
    write_eval_summary,
    write_sample_rows_csv,
)


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
def run_sample_eval_epoch(model, dataloader, device, cfg, output_dir: str | None = None, split_name: str = "eval"):
    model.eval()
    rows = []
    raw_model = model.module if hasattr(model, "module") else model
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            features = raw_model.extract_features(images.float()).float()
            classifier_features = F.normalize(features, dim=-1) if getattr(raw_model, "normalize_features", False) else features
            logits = raw_model.classifier(classifier_features.float())
            prob_fake = torch.softmax(logits, dim=1)[:, 1]
            feature_norm = features.norm(dim=-1)
        rows.extend(build_sample_rows(batch, logits, prob_fake, feature_norm))

    rows = gather_rows_across_ranks(rows)
    summary = summarize_sample_rows(rows)
    metrics_05 = video_metrics_from_rows(rows, threshold=0.5)
    result = {**metrics_05, **summary, "loss": 0.0}
    if output_dir is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        write_sample_rows_csv(rows, f"{output_dir}/{split_name}_sample_rows.csv")
        write_eval_summary(summary, metrics_05, f"{output_dir}/{split_name}_summary.json")
    return result


@torch.no_grad()
def run_eval_epoch(model, dataloader, device, cfg):
    return run_sample_eval_epoch(model, dataloader, device, cfg)
