from __future__ import annotations

import torch
import torch.nn.functional as F


def same_frame_contrastive_loss(background, real_face, fake_face, temperature: float = 0.07):
    background = F.normalize(background, dim=1)
    real_face = F.normalize(real_face, dim=1)
    fake_face = F.normalize(fake_face, dim=1)
    pos = (background * real_face).sum(dim=1, keepdim=True) / temperature
    neg = (background * fake_face).sum(dim=1, keepdim=True) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(background.size(0), dtype=torch.long, device=background.device)
    return F.cross_entropy(logits, labels)


def infonce_bg_face_loss(bg_features, real_face_features, fake_face_features, temperature=0.07):
    """InfoNCE loss with multiple bg patches per sample.

    Args:
        bg_features: (B, K, D) — K bg patches per sample, already L2-normalized
        real_face_features: (B, D) — real face features, already L2-normalized
        fake_face_features: (B, D) — fake face features, already L2-normalized
        temperature: softmax temperature

    Returns:
        scalar loss. Positive: (bg_i, real_face). Negatives: (bg_i, fake_face) + cross-sample.
    """
    B, K, D = bg_features.shape
    bg_flat = bg_features.reshape(B * K, D)

    # Positive: each bg patch with its own sample's real face
    real_expanded = real_face_features.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
    pos_sim = (bg_flat * real_expanded).sum(dim=1) / temperature  # (B*K,)

    # Negative: each bg patch with its own sample's fake face
    fake_expanded = fake_face_features.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
    neg_same = (bg_flat * fake_expanded).sum(dim=1) / temperature  # (B*K,)

    # Cross-sample negatives: bg with all other samples' faces
    cross_real = torch.mm(bg_flat, real_face_features.t()) / temperature  # (B*K, B)
    cross_fake = torch.mm(bg_flat, fake_face_features.t()) / temperature  # (B*K, B)

    # Mask out self-pairs in cross_real
    sample_idx = torch.arange(B, device=bg_features.device).unsqueeze(1).expand(B, K).reshape(B * K)
    cross_real[sample_idx, sample_idx[:B]] = float('-inf')

    # All negatives: same-sample fake + cross-sample real + cross-sample fake
    all_neg = torch.cat([neg_same.unsqueeze(1), cross_real, cross_fake], dim=1)  # (B*K, 1+2B)

    # Logits: positive + all negatives
    logits = torch.cat([pos_sim.unsqueeze(1), all_neg], dim=1)  # (B*K, 2+2B)

    # Labels: positive is always index 0
    labels = torch.zeros(B * K, dtype=torch.long, device=bg_features.device)

    return F.cross_entropy(logits, labels)
