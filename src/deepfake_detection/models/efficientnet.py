from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, images):
        return self.backbone(images)
