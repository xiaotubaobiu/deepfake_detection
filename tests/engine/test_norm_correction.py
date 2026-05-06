import torch

from deepfake_detection.evaluation.norm_correction import (
    apply_partial_norm,
    apply_mean_scale_norm,
    fuse_scores,
    best_accuracy_threshold,
)


def test_apply_partial_norm_uses_alpha_power_of_feature_norm():
    features = torch.tensor([[3.0, 4.0]])

    corrected = apply_partial_norm(features, alpha=0.5)

    assert torch.allclose(corrected, features / (torch.tensor([[5.0]]) ** 0.5))


def test_apply_mean_scale_norm_restores_global_scale():
    features = torch.tensor([[3.0, 4.0]])

    corrected = apply_mean_scale_norm(features, mean_scale=10.0)

    assert torch.allclose(corrected, torch.tensor([[6.0, 8.0]]))


def test_fuse_scores_interpolates_raw_and_norm_scores():
    raw = torch.tensor([0.2, 0.8])
    norm = torch.tensor([0.6, 0.4])

    fused = fuse_scores(raw, norm, beta=0.25)

    assert torch.allclose(fused, torch.tensor([0.3, 0.7]))


def test_best_accuracy_threshold_selects_threshold_from_validation_scores():
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.4, 0.35, 0.9])

    threshold, acc = best_accuracy_threshold(labels, scores)

    assert abs(threshold - 0.35) < 1e-6
    assert acc == 0.75
