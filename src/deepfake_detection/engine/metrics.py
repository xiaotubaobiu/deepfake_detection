from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def mean_video_score(scores):
    return float(np.mean(scores))


def compute_auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = np.argmin(np.abs(fnr - fpr))
    return float((fpr[eer_threshold] + fnr[eer_threshold]) / 2)


def compute_acc(labels, scores, threshold=0.5):
    preds = (np.array(scores) >= threshold).astype(int)
    return float(accuracy_score(labels, preds))


def aggregate_video_predictions(rows: list[dict]) -> tuple[list[int], list[float]]:
    from collections import defaultdict
    by_video: dict[str, dict] = defaultdict(lambda: {"scores": [], "label": None})
    for row in rows:
        by_video[row["video_id"]]["scores"].append(row["score"])
        by_video[row["video_id"]]["label"] = row["label"]
    labels, scores = [], []
    for vid, data in by_video.items():
        labels.append(data["label"])
        scores.append(mean_video_score(data["scores"]))
    return labels, scores
