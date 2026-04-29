from __future__ import annotations

import clip
import torch.nn as nn


class CLIPFineTuneBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16"):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.visual = clip_model.visual
        self.classifier = nn.Linear(self.visual.output_dim, 2)

    def forward(self, images):
        image_features = self.visual(images)
        return self.classifier(image_features.float())
