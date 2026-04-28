from __future__ import annotations

import torch
import torch.nn as nn
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS
from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts


class CLIPPromptBackgroundContrastModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", projection_dim: int = 256):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        width = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(width, 2)
        self.face_projection = nn.Linear(width, projection_dim)
        self.background_projection = nn.Linear(width, projection_dim)
        real_texts, fake_texts = build_fixed_prompt_texts()
        self.register_buffer("_real_tokens", clip.tokenize(real_texts))
        self.register_buffer("_fake_tokens", clip.tokenize(fake_texts))

    def encode_image(self, images):
        return self.clip_model.encode_image(images)

    def forward(self, images):
        return self.forward_classification(images)

    def forward_classification(self, images):
        image_features = self.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        real_features = self.clip_model.encode_text(self._real_tokens)
        real_features = real_features / real_features.norm(dim=-1, keepdim=True)
        fake_features = self.clip_model.encode_text(self._fake_tokens)
        fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
        real_sim = (image_features @ real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ fake_features.T).mean(dim=1, keepdim=True)
        logits = torch.cat([real_sim, fake_sim], dim=1) * 100.0
        return logits

    def forward_contrastive(self, background_images, real_face_images, fake_face_images):
        bg_feat = self.background_projection(self.encode_image(background_images).float())
        real_feat = self.face_projection(self.encode_image(real_face_images).float())
        fake_feat = self.face_projection(self.encode_image(fake_face_images).float())
        return bg_feat, real_feat, fake_feat
