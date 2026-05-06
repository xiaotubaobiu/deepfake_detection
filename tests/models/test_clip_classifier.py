import torch
import torch.nn.functional as F

from deepfake_detection.models.clip_classifier import cosine_classifier_logits


def test_cosine_classifier_logits_normalizes_features_and_weights_without_bias():
    features = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([
        [6.0, 8.0],
        [-8.0, 6.0],
    ])
    bias = torch.tensor([100.0, -100.0])

    logits = cosine_classifier_logits(features, weight, bias=bias, scale=16.0)

    expected = 16.0 * F.normalize(features, dim=-1) @ F.normalize(weight, dim=-1).t()
    assert torch.allclose(logits, expected)
