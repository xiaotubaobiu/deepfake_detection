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
        normalize_features = model_cfg.get("normalize_features", False)
        return CLIPFineTuneBinaryClassifier(clip_name, normalize_features)
    if name == "clip_prompt":
        tau = model_cfg.get("tau", 0.07)
        freeze_visual = model_cfg.get("freeze_visual", "none")
        freeze_visual_layers = model_cfg.get("freeze_visual_layers", 9)
        projection_dim = model_cfg.get("projection_dim", 512)
        classifier_normalize_features = model_cfg.get("classifier_normalize_features", True)
        return CLIPPromptBinaryClassifier(
            clip_name,
            tau,
            freeze_visual,
            freeze_visual_layers,
            projection_dim,
            classifier_normalize_features,
        )
    if name == "clip_prompt_bgcontrast":
        proj_dim = model_cfg.get("projection_dim", 256)
        return CLIPBgFaceContrastModel(clip_name, proj_dim)
    raise ValueError(f"Unknown model name: {name}")
