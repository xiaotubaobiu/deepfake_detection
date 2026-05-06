from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_inference_norm(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(features, dim=-1, eps=eps)


def apply_partial_norm(features: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    norms = features.norm(dim=-1, keepdim=True).clamp_min(eps)
    return features / norms.pow(alpha)


def fuse_scores(raw_scores: torch.Tensor, norm_scores: torch.Tensor, beta: float) -> torch.Tensor:
    return (1 - beta) * raw_scores + beta * norm_scores


def classifier_scores_from_features(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mode: str,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    feature_norm = features.norm(dim=-1)
    if mode == "raw":
        corrected = features
    elif mode == "norm":
        corrected = apply_inference_norm(features)
    elif mode == "partial":
        corrected = apply_partial_norm(features, alpha)
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    logits = corrected @ weight.t().to(features.device) + bias.to(features.device)
    prob_fake = torch.softmax(logits, dim=1)[:, 1]
    return {"logits": logits, "prob_fake": prob_fake, "feature_norm": feature_norm}
