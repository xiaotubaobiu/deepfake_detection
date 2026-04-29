from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS
from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts


class CLIPBgFaceContrastModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", projection_dim: int = 256):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        self.visual = clip_model.visual
        feat_dim = self.visual.output_dim

        self.classifier = nn.Linear(feat_dim, 2)
        self.bg_projection = nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.face_projection = nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Pre-compute prompt features
        real_texts, fake_texts = build_fixed_prompt_texts()
        real_tokens = clip.tokenize(real_texts)
        fake_tokens = clip.tokenize(fake_texts)
        with torch.no_grad():
            real_features = clip_model.encode_text(real_tokens)
            fake_features = clip_model.encode_text(fake_tokens)
        real_features = F.normalize(real_features, dim=-1)
        fake_features = F.normalize(fake_features, dim=-1)
        self.register_buffer("_real_features", real_features)
        self.register_buffer("_fake_features", fake_features)
        self.register_buffer("tau", torch.tensor(0.07))

    def encode_face(self, face_images):
        """Encode face crop(s). Accepts (B, 3, H, W)."""
        return F.normalize(self.visual(face_images), dim=-1)

    def encode_bg(self, bg_images):
        """Encode background patches. Accepts (B, K, 3, H, W) or (B, 3, H, W)."""
        if bg_images.dim() == 5:
            B, K, C, H, W = bg_images.shape
            flat = bg_images.reshape(B * K, C, H, W)
            feats = F.normalize(self.visual(flat), dim=-1)
            return feats.reshape(B, K, -1)
        return F.normalize(self.visual(bg_images), dim=-1)

    def forward(self, images):
        """Standard classification forward (for eval without bg)."""
        image_features = self.encode_face(images)
        return self.classifier(image_features.float())

    def forward_classification(self, face_images):
        """Classification logits from face crop."""
        face_feat = self.encode_face(face_images)
        return self.classifier(face_feat.float())

    def forward_contrastive(self, bg_images, face_images):
        """Project bg and face features for contrastive loss.

        Args:
            bg_images: (B, K, 3, H, W) background patches
            face_images: (B, 3, H, W) face crops

        Returns:
            bg_proj: (B, K, proj_dim) L2-normalized
            face_proj: (B, proj_dim) L2-normalized
        """
        bg_feat = self.encode_bg(bg_images)
        face_feat = self.encode_face(face_images)

        B, K, D = bg_feat.shape
        bg_proj = F.normalize(self.bg_projection(bg_feat.reshape(B * K, D).float()), dim=-1)
        bg_proj = bg_proj.reshape(B, K, -1)
        face_proj = F.normalize(self.face_projection(face_feat.float()), dim=-1)
        return bg_proj, face_proj
