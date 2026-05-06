import pytest
import torch

from deepfake_detection.evaluation.norm_modes import (
    apply_inference_norm,
    apply_partial_norm,
    classifier_scores_from_features,
    fuse_scores,
)


def test_apply_inference_norm_makes_unit_features():
    features = torch.tensor([[3.0, 4.0], [5.0, 12.0]])

    normalized = apply_inference_norm(features)

    assert torch.allclose(normalized.norm(dim=-1), torch.ones(2))


def test_apply_partial_norm_alpha_zero_keeps_features():
    features = torch.tensor([[3.0, 4.0]])

    corrected = apply_partial_norm(features, alpha=0.0)

    assert torch.allclose(corrected, features)


def test_apply_partial_norm_alpha_one_normalizes_features():
    features = torch.tensor([[3.0, 4.0]])

    corrected = apply_partial_norm(features, alpha=1.0)

    assert torch.allclose(corrected, torch.tensor([[0.6, 0.8]]))


def test_fuse_scores_interpolates_raw_and_norm():
    raw = torch.tensor([0.2, 0.8])
    norm = torch.tensor([0.6, 0.4])

    fused = fuse_scores(raw, norm, beta=0.25)

    assert torch.allclose(fused, torch.tensor([0.3, 0.7]))


def test_classifier_scores_from_features_returns_logits_probs_and_norms():
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([0.0, 0.0])

    rows = classifier_scores_from_features(features, weight, bias, mode="raw")

    assert torch.allclose(rows["logits"], torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    assert torch.allclose(rows["feature_norm"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(rows["prob_fake"], torch.softmax(rows["logits"], dim=1)[:, 1])


def test_classifier_scores_from_features_supports_norm_mode():
    features = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([0.0, 0.0])

    rows = classifier_scores_from_features(features, weight, bias, mode="norm")

    assert torch.allclose(rows["logits"], torch.tensor([[0.6, 0.8]]))
    assert torch.allclose(rows["feature_norm"], torch.tensor([5.0]))


def test_classifier_scores_from_features_supports_partial_mode():
    features = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([0.0, 0.0])

    rows = classifier_scores_from_features(features, weight, bias, mode="partial", alpha=0.5)

    expected = features / torch.tensor([[5.0]]).pow(0.5)
    assert torch.allclose(rows["logits"], expected)
    assert torch.allclose(rows["feature_norm"], torch.tensor([5.0]))


def test_classifier_scores_from_features_rejects_unknown_mode():
    features = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([0.0, 0.0])

    with pytest.raises(ValueError, match="Unknown norm mode"):
        classifier_scores_from_features(features, weight, bias, mode="bad")
