from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2BinaryClassifier(nn.Module):
    def __init__(self, normalize_features: bool = False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.feature_dim = self.backbone.embed_dim  # 768
        self.classifier = nn.Linear(self.feature_dim, 2)
        self.normalize_features = normalize_features

    def extract_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        features = self.extract_features(images)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.classifier(features)
