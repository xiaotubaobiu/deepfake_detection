import numpy as np
import torch
import torch.nn as nn
from deepfake_detection.engine.metrics import mean_video_score, compute_auc, compute_eer, compute_acc, aggregate_video_predictions


def test_mean_video_score_averages_eight_frame_scores():
    score = mean_video_score([0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5])
    assert abs(score - 0.40625) < 1e-6


def test_compute_auc_on_perfect_classifier():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(labels, scores) == 1.0


def test_aggregate_video_predictions():
    rows = [
        {"video_id": "v0", "score": 0.1, "label": 0},
        {"video_id": "v0", "score": 0.3, "label": 0},
        {"video_id": "v1", "score": 0.8, "label": 1},
    ]
    labels, scores = aggregate_video_predictions(rows)
    assert len(labels) == 2
    assert len(scores) == 2


class _FakePromptModel(nn.Module):
    """用于测试 prompt_contrast_step 的 mock 模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, images):
        cls_logits = self.linear(images.mean(dim=[1, 2, 3]).unsqueeze(1).expand(-1, 10))
        prompt_logits = cls_logits * 2.0
        return cls_logits, prompt_logits


def test_prompt_contrast_step_returns_total_loss_and_logits():
    from deepfake_detection.engine.trainers import prompt_contrast_step
    model = _FakePromptModel()
    batch = {
        "image": torch.randn(4, 3, 32, 32),
        "label": torch.tensor([0, 1, 0, 1]),
    }
    total_loss, cls_logits, prompt_logits = prompt_contrast_step(model, batch, "cpu", beta=0.1)
    assert total_loss.ndim == 0
    assert cls_logits.shape == (4, 2)
    assert prompt_logits.shape == (4, 2)
    assert total_loss.item() > 0
