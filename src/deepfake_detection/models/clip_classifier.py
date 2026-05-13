from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_classifier_logits(features, weight, bias=None, scale: float = 16.0):
    features = F.normalize(features, dim=-1)
    weight = F.normalize(weight, dim=-1)
    return scale * features @ weight.t()


class CLIPFineTuneBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", normalize_features: bool = False):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.visual = clip_model.visual
        self.classifier = nn.Linear(self.visual.output_dim, 2)
        self.normalize_features = normalize_features

    def extract_features(self, images):
        return self.visual(images)

    def forward(self, images):
        image_features = self.extract_features(images)
        if self.normalize_features:
            image_features = F.normalize(image_features, dim=-1)
        return self.classifier(image_features.float())
