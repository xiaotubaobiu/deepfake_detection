from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import timm


class XceptionBinaryClassifier(nn.Module):
    def __init__(self, normalize_features: bool = False):
        super().__init__()
        self.backbone = timm.create_model("legacy_xception", pretrained=True)
        self.feature_dim = self.backbone.num_features  # 2048
        self.backbone.reset_classifier(0)
        self.classifier = nn.Linear(self.feature_dim, 2)
        self.normalize_features = normalize_features

    def extract_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        features = self.extract_features(images)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.classifier(features)
