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
