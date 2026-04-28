import numpy as np
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
