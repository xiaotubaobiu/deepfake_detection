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


def update_video_aggregate(aggregates: dict, video_id, score: float, label: int) -> None:
    data = aggregates.setdefault(video_id, {"score_sum": 0.0, "count": 0, "label": label})
    data["score_sum"] += score
    data["count"] += 1
    data["label"] = label


def merge_video_aggregates(target: dict, source: dict) -> None:
    for video_id, data in source.items():
        current = target.setdefault(video_id, {"score_sum": 0.0, "count": 0, "label": data["label"]})
        current["score_sum"] += data["score_sum"]
        current["count"] += data["count"]
        current["label"] = data["label"]


def video_aggregates_to_predictions(aggregates: dict) -> tuple[list[int], list[float]]:
    labels, scores = [], []
    for data in aggregates.values():
        labels.append(data["label"])
        scores.append(float(data["score_sum"] / max(data["count"], 1)))
    return labels, scores


def aggregate_video_predictions(rows: list[dict]) -> tuple[list[int], list[float]]:
    by_video: dict[str, dict] = {}
    for row in rows:
        update_video_aggregate(by_video, row["video_id"], float(row["score"]), int(row["label"]))
    return video_aggregates_to_predictions(by_video)
