from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import torch

from deepfake_detection.engine.metrics import compute_acc, compute_auc, compute_eer


SAMPLE_ROW_FIELDS = [
    "sample_id",
    "image_path",
    "video_id",
    "label",
    "logit_real",
    "logit_fake",
    "prob_fake",
    "feature_norm",
]


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def build_sample_rows(batch: dict, logits: torch.Tensor, prob_fake: torch.Tensor, feature_norm: torch.Tensor) -> list[dict]:
    sample_ids = _as_list(batch["sample_id"])
    image_paths = _as_list(batch["image_path"])
    video_ids = _as_list(batch["video_id"])
    labels = batch["label"].detach().cpu().tolist()
    logits_cpu = logits.detach().cpu().float()
    probs_cpu = prob_fake.detach().cpu().float()
    norms_cpu = feature_norm.detach().cpu().float()
    rows = []
    for i in range(len(sample_ids)):
        rows.append({
            "sample_id": sample_ids[i],
            "image_path": image_paths[i],
            "video_id": video_ids[i],
            "label": int(labels[i]),
            "logit_real": float(logits_cpu[i, 0]),
            "logit_fake": float(logits_cpu[i, 1]),
            "prob_fake": float(probs_cpu[i]),
            "feature_norm": float(norms_cpu[i]),
        })
    return rows


def deduplicate_sample_rows(rows: Iterable[dict]) -> list[dict]:
    by_key = OrderedDict()
    for row in rows:
        key = row.get("sample_id") or row["image_path"]
        if key not in by_key:
            by_key[key] = row
    return list(by_key.values())


def summarize_sample_rows(rows: list[dict]) -> dict:
    deduped = deduplicate_sample_rows(rows)
    return {
        "rows_before_dedup": len(rows),
        "rows_after_dedup": len(deduped),
        "unique_sample_ids_before": len({row.get("sample_id") for row in rows if row.get("sample_id")}),
        "unique_image_paths_before": len({row["image_path"] for row in rows}),
        "unique_videos_after": len({row["video_id"] for row in deduped}),
    }


def sample_rows_to_video_predictions(rows: list[dict]) -> tuple[list[int], list[float]]:
    grouped = defaultdict(lambda: {"score_sum": 0.0, "count": 0, "label": 0})
    for row in rows:
        data = grouped[row["video_id"]]
        data["score_sum"] += float(row["prob_fake"])
        data["count"] += 1
        data["label"] = int(row["label"])
    labels = []
    scores = []
    for data in grouped.values():
        labels.append(data["label"])
        scores.append(data["score_sum"] / max(data["count"], 1))
    return labels, scores


def video_metrics_from_rows(rows: list[dict], threshold: float = 0.5) -> dict:
    labels, scores = sample_rows_to_video_predictions(deduplicate_sample_rows(rows))
    if len(set(labels)) < 2:
        return {"auc": 0.0, "eer": 0.0, "acc": compute_acc(labels, scores, threshold)}
    return {
        "auc": compute_auc(labels, scores),
        "eer": compute_eer(labels, scores),
        "acc": compute_acc(labels, scores, threshold),
    }


def gather_rows_across_ranks(rows: list[dict]) -> list[dict]:
    if not torch.distributed.is_initialized():
        return rows
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, rows)
    merged = []
    for rank_rows in gathered:
        merged.extend(rank_rows)
    return merged


def write_sample_rows_csv(rows: list[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SAMPLE_ROW_FIELDS})


def write_eval_summary(summary: dict, metrics: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "metrics": metrics}, f, indent=2, sort_keys=True)
