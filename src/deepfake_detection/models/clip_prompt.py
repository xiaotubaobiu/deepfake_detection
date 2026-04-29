from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def build_fixed_prompt_texts():
    return REAL_PROMPTS, FAKE_PROMPTS


class CLIPPromptBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", tau: float = 0.07):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        # 只保留 visual encoder（可训练）
        self.visual = clip_model.visual

        # 预计算 prompt 特征（冻结 text encoder，用完即弃）
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
        # text encoder 随 clip_model 一起释放，不保留引用

        # 分类头
        self.classifier = nn.Linear(self.visual.output_dim, 2)

        # 温度参数（固定，不可训练，随模型保存/迁移）
        self.register_buffer("tau", torch.tensor(tau))

    def forward(self, images):
        # 1. 视觉特征（原始，不归一化）
        raw_features = self.visual(images)

        # 2. 分类头 logits（用原始特征，和 exp2 一致）
        cls_logits = self.classifier(raw_features.float())

        # 3. Prompt logits（归一化后做余弦相似度）
        image_features = F.normalize(raw_features, dim=-1)
        real_sim = (image_features @ self._real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ self._fake_features.T).mean(dim=1, keepdim=True)
        prompt_logits = torch.cat([real_sim, fake_sim], dim=1) / self.tau

        return cls_logits, prompt_logits
