import torch

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def test_build_fixed_prompt_texts_returns_real_and_fake_prompts():
    from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts

    real, fake = build_fixed_prompt_texts()

    assert real == REAL_PROMPTS
    assert fake == FAKE_PROMPTS


def test_prompt_counts():
    assert len(REAL_PROMPTS) == 3
    assert len(FAKE_PROMPTS) == 3


def test_clip_prompt_forward_returns_tuple_of_logits():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    images = torch.randn(2, 3, 224, 224)
    cls_logits, prompt_logits = model(images)
    assert cls_logits.shape == (2, 2)
    assert prompt_logits.shape == (2, 2)


def test_clip_prompt_prompt_features_are_frozen_buffers():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    assert not hasattr(model, "clip_model"), "clip_model should not be kept"
    for name, buf in model.named_buffers():
        if "real_features" in name or "fake_features" in name:
            assert not buf.requires_grad


def test_clip_prompt_has_classifier_head():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    assert hasattr(model, "classifier")
    assert model.classifier.out_features == 2
