from __future__ import annotations

import torch
import torch.nn as nn
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def build_fixed_prompt_texts():
    return REAL_PROMPTS, FAKE_PROMPTS


class CLIPPromptBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        real_texts, fake_texts = build_fixed_prompt_texts()
        self.register_buffer("_real_tokens", clip.tokenize(real_texts))
        self.register_buffer("_fake_tokens", clip.tokenize(fake_texts))

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        real_features = self.clip_model.encode_text(self._real_tokens)
        real_features = real_features / real_features.norm(dim=-1, keepdim=True)
        fake_features = self.clip_model.encode_text(self._fake_tokens)
        fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
        real_sim = (image_features @ real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ fake_features.T).mean(dim=1, keepdim=True)
        logits = torch.cat([real_sim, fake_sim], dim=1) * 100.0
        return logits
