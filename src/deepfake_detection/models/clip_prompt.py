from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def build_fixed_prompt_texts():
    return REAL_PROMPTS, FAKE_PROMPTS


class CLIPPromptBinaryClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        tau: float = 0.07,
        freeze_visual: str = "none",
        freeze_visual_layers: int = 9,
        projection_dim: int = 512,
        classifier_normalize_features: bool = True,
    ):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        self.visual = clip_model.visual
        self.classifier = nn.Linear(self.visual.output_dim, 2)
        self.classifier_normalize_features = classifier_normalize_features
        self.image_projection = nn.Linear(self.visual.output_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(self.visual.output_dim, projection_dim, bias=False)
        self.register_buffer("tau", torch.tensor(tau))

        real_texts, fake_texts = build_fixed_prompt_texts()
        real_tokens = clip.tokenize(real_texts)
        fake_tokens = clip.tokenize(fake_texts)
        with torch.no_grad():
            real_features = clip_model.encode_text(real_tokens)
            fake_features = clip_model.encode_text(fake_tokens)
        self.register_buffer("_real_features", F.normalize(real_features, dim=-1))
        self.register_buffer("_fake_features", F.normalize(fake_features, dim=-1))

        self.apply_visual_freeze(freeze_visual, freeze_visual_layers)

    def apply_visual_freeze(self, mode: str, freeze_visual_layers: int):
        if mode == "none":
            return
        if mode == "all":
            for p in self.visual.parameters():
                p.requires_grad = False
            return
        if mode == "partial":
            for p in self.visual.parameters():
                p.requires_grad = False
            for p in self.visual.ln_post.parameters():
                p.requires_grad = True
            if self.visual.proj is not None:
                self.visual.proj.requires_grad = True
            blocks = self.visual.transformer.resblocks
            for block in blocks[freeze_visual_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            return
        raise ValueError(f"Unknown freeze_visual mode: {mode}")

    def encode_image_features(self, images):
        image_features = self.visual(images)
        return F.normalize(image_features, dim=-1)

    def encode_raw_image_features(self, images):
        return self.visual(images)

    def prompt_logits_from_features(self, image_features):
        real_sim = (image_features @ self._real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ self._fake_features.T).mean(dim=1, keepdim=True)
        return torch.cat([real_sim, fake_sim], dim=1) / self.tau

    def forward(self, images):
        raw_features = self.encode_raw_image_features(images)
        image_features = F.normalize(raw_features, dim=-1)
        classifier_features = image_features if self.classifier_normalize_features else raw_features
        cls_logits = self.classifier(classifier_features.float())
        prompt_logits = self.prompt_logits_from_features(image_features)
        return cls_logits, prompt_logits

    def forward_with_features(self, images):
        raw_features = self.encode_raw_image_features(images)
        image_features = F.normalize(raw_features, dim=-1)
        classifier_features = image_features if self.classifier_normalize_features else raw_features
        cls_logits = self.classifier(classifier_features.float())
        prompt_logits = self.prompt_logits_from_features(image_features)
        text_features = torch.stack([
            self._real_features.mean(dim=0),
            self._fake_features.mean(dim=0),
        ], dim=0)
        image_features = self.image_projection(image_features.float())
        text_features = self.text_projection(text_features.float())
        return cls_logits, prompt_logits, image_features, text_features
