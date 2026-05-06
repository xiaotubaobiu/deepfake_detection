import json

import torch

from deepfake_detection.evaluation import sample_eval
from deepfake_detection.evaluation.sample_eval import (
    build_sample_rows,
    deduplicate_sample_rows,
    gather_rows_across_ranks,
    sample_rows_to_video_predictions,
    summarize_sample_rows,
    video_metrics_from_rows,
    write_eval_summary,
    write_sample_rows_csv,
)


def test_build_sample_rows_keeps_logits_probs_norms_and_paths():
    batch = {
        "sample_id": ["s1", "s2"],
        "image_path": ["/x/1.jpg", "/x/2.jpg"],
        "video_id": ["v1", "v2"],
        "label": torch.tensor([0, 1]),
    }
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    prob_fake = torch.softmax(logits, dim=1)[:, 1]
    feature_norm = torch.tensor([11.0, 12.0])

    rows = build_sample_rows(batch, logits, prob_fake, feature_norm)

    assert rows == [
        {
            "sample_id": "s1",
            "image_path": "/x/1.jpg",
            "video_id": "v1",
            "label": 0,
            "logit_real": 2.0,
            "logit_fake": 0.0,
            "prob_fake": float(prob_fake[0]),
            "feature_norm": 11.0,
        },
        {
            "sample_id": "s2",
            "image_path": "/x/2.jpg",
            "video_id": "v2",
            "label": 1,
            "logit_real": 0.0,
            "logit_fake": 2.0,
            "prob_fake": float(prob_fake[1]),
            "feature_norm": 12.0,
        },
    ]


def test_deduplicate_sample_rows_prefers_sample_id():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "b", "image_path": "/x/b.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.3},
    ]

    deduped = deduplicate_sample_rows(rows)

    assert [row["sample_id"] for row in deduped] == ["a", "b"]


def test_deduplicate_sample_rows_falls_back_to_image_path_without_sample_id():
    rows = [
        {"image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.9},
        {"sample_id": None, "image_path": "/x/b.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.8},
    ]

    deduped = deduplicate_sample_rows(rows)

    assert [row["image_path"] for row in deduped] == ["/x/a.jpg", "/x/b.jpg"]
    assert [row["prob_fake"] for row in deduped] == [0.1, 0.8]


def test_deduplicate_sample_rows_preserves_first_occurrence_order_and_values():
    rows = [
        {"sample_id": "b", "image_path": "/x/b-first.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.2},
        {"sample_id": "a", "image_path": "/x/a-first.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.7},
        {"sample_id": "b", "image_path": "/x/b-second.jpg", "video_id": "v9", "label": 1, "prob_fake": 0.9},
        {"sample_id": "a", "image_path": "/x/a-second.jpg", "video_id": "v8", "label": 0, "prob_fake": 0.1},
    ]

    deduped = deduplicate_sample_rows(rows)

    assert deduped == [rows[0], rows[1]]


def test_summarize_sample_rows_counts_before_and_after_dedup():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "c", "image_path": "/x/c.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.9},
    ]

    summary = summarize_sample_rows(rows)

    assert summary == {
        "rows_before_dedup": 3,
        "rows_after_dedup": 2,
        "unique_sample_ids_before": 2,
        "unique_image_paths_before": 2,
        "unique_videos_after": 2,
    }


def test_sample_rows_to_video_predictions_averages_frame_scores():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.2},
        {"sample_id": "b", "image_path": "/x/b.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.4},
        {"sample_id": "c", "image_path": "/x/c.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.8},
    ]

    labels, scores = sample_rows_to_video_predictions(rows)

    assert labels == [0, 1]
    assert scores == [0.30000000000000004, 0.8]


def test_video_metrics_from_rows_supports_fixed_threshold():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "b", "image_path": "/x/b.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.9},
    ]

    metrics = video_metrics_from_rows(rows, threshold=0.5)

    assert metrics["auc"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["acc"] == 1.0


def test_video_metrics_from_rows_deduplicates_before_aggregation():
    rows = [
        {"sample_id": "real", "image_path": "/x/real.jpg", "video_id": "real-video", "label": 0, "prob_fake": 0.1},
        {"sample_id": "fake", "image_path": "/x/fake.jpg", "video_id": "fake-video", "label": 1, "prob_fake": 0.9},
        {"sample_id": "fake", "image_path": "/x/fake-pad.jpg", "video_id": "fake-video", "label": 1, "prob_fake": 0.0},
    ]

    metrics = video_metrics_from_rows(rows, threshold=0.5)

    assert metrics["acc"] == 1.0
    assert metrics["auc"] == 1.0


def test_video_metrics_from_rows_computes_acc_for_single_class():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 1, "prob_fake": 0.9},
        {"sample_id": "b", "image_path": "/x/b.jpg", "video_id": "v2", "label": 1, "prob_fake": 0.8},
    ]

    metrics = video_metrics_from_rows(rows, threshold=0.5)

    assert metrics == {"auc": 0.0, "eer": 0.0, "acc": 1.0}


def test_gather_rows_across_ranks_non_distributed_returns_input_unchanged(monkeypatch):
    rows = [{"sample_id": "a"}]
    monkeypatch.setattr(sample_eval.torch.distributed, "is_initialized", lambda: False)

    gathered = gather_rows_across_ranks(rows)

    assert gathered is rows


def test_gather_rows_across_ranks_distributed_merges_rank_rows(monkeypatch):
    rows = [{"sample_id": "rank0"}]

    def fake_all_gather_object(gathered, local_rows):
        assert local_rows is rows
        gathered[0] = [{"sample_id": "rank0"}]
        gathered[1] = [{"sample_id": "rank1"}]

    monkeypatch.setattr(sample_eval.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(sample_eval.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(sample_eval.torch.distributed, "all_gather_object", fake_all_gather_object)

    gathered = gather_rows_across_ranks(rows)

    assert gathered == [{"sample_id": "rank0"}, {"sample_id": "rank1"}]


def test_write_sample_rows_csv_creates_parent_dirs_and_stable_header(tmp_path):
    output_path = tmp_path / "nested" / "sample_rows.csv"
    rows = [
        {
            "sample_id": "a",
            "image_path": "/x/a.jpg",
            "video_id": "v1",
            "label": 0,
            "logit_real": 1.25,
            "logit_fake": -0.5,
            "prob_fake": 0.15,
            "feature_norm": 3.0,
            "extra": "ignored",
        }
    ]

    write_sample_rows_csv(rows, str(output_path))

    assert output_path.read_text() == (
        "sample_id,image_path,video_id,label,logit_real,logit_fake,prob_fake,feature_norm\n"
        "a,/x/a.jpg,v1,0,1.25,-0.5,0.15,3.0\n"
    )


def test_write_eval_summary_creates_parent_dirs_and_stable_json(tmp_path):
    output_path = tmp_path / "nested" / "eval_summary.json"
    summary = {"rows_before_dedup": 2, "rows_after_dedup": 1}
    metrics = {"eer": 0.2, "auc": 0.8, "acc": 0.75}

    write_eval_summary(summary, metrics, str(output_path))

    assert json.loads(output_path.read_text()) == {"summary": summary, "metrics": metrics}
    assert output_path.read_text() == (
        '{\n'
        '  "metrics": {\n'
        '    "acc": 0.75,\n'
        '    "auc": 0.8,\n'
        '    "eer": 0.2\n'
        '  },\n'
        '  "summary": {\n'
        '    "rows_after_dedup": 1,\n'
        '    "rows_before_dedup": 2\n'
        '  }\n'
        '}'
    )
