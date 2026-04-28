import torch
from deepfake_detection.losses.contrastive import same_frame_contrastive_loss


def test_same_frame_contrastive_loss_returns_scalar():
    background = torch.randn(4, 8)
    real_face = torch.randn(4, 8)
    fake_face = torch.randn(4, 8)
    loss = same_frame_contrastive_loss(background, real_face, fake_face, temperature=0.07)
    assert loss.ndim == 0


def test_same_frame_contrastive_loss_is_non_negative():
    background = torch.randn(4, 8)
    real_face = torch.randn(4, 8)
    fake_face = torch.randn(4, 8)
    loss = same_frame_contrastive_loss(background, real_face, fake_face, temperature=0.07)
    assert loss.item() >= 0
