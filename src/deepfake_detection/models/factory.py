from __future__ import annotations

from deepfake_detection.models.efficientnet import EfficientNetBinaryClassifier
from deepfake_detection.models.clip_classifier import CLIPFineTuneBinaryClassifier
from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
from deepfake_detection.models.clip_bgcontrast import CLIPBgFaceContrastModel


def build_model(model_cfg: dict):
    name = model_cfg["name"]
    clip_name = model_cfg.get("clip_model_name", "ViT-B/16")
    if name == "efficientnet_b0":
        return EfficientNetBinaryClassifier()
    if name == "clip_finetune":
        return CLIPFineTuneBinaryClassifier(clip_name)
    if name == "clip_prompt":
        return CLIPPromptBinaryClassifier(clip_name)
    if name == "clip_prompt_bgcontrast":
        proj_dim = model_cfg.get("projection_dim", 256)
        return CLIPBgFaceContrastModel(clip_name, proj_dim)
    raise ValueError(f"Unknown model name: {name}")
