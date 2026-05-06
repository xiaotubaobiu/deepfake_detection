from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_partial_norm(features: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    norms = features.norm(dim=-1, keepdim=True).clamp_min(eps)
    return features / norms.pow(alpha)


def apply_mean_scale_norm(features: torch.Tensor, mean_scale: float, eps: float = 1e-12) -> torch.Tensor:
    return mean_scale * F.normalize(features, dim=-1, eps=eps)


def fuse_scores(raw_scores: torch.Tensor, norm_scores: torch.Tensor, beta: float) -> torch.Tensor:
    return (1 - beta) * raw_scores + beta * norm_scores


def best_accuracy_threshold(labels: torch.Tensor, scores: torch.Tensor) -> tuple[float, float]:
    thresholds = torch.unique(scores).sort().values
    best_threshold = float(thresholds[0].item())
    best_acc = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).long()
        acc = (preds == labels.long()).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold.item())
    return best_threshold, best_acc
