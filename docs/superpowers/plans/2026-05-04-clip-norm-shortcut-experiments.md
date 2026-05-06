# CLIP Norm Shortcut Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, auditable CLIP-based deepfake detection experiment pipeline to verify `Direction ≥ Direction + Magnitude ≫ Magnitude` on FF++/CDF/DFD.

**Architecture:** First create a trustworthy sample-level evaluation layer that gathers DDP predictions, deduplicates padded samples, saves audit rows, and computes video-level metrics. Then prune unrelated experimental code, keep only the raw/norm train-test paths, and run three detector settings plus magnitude-only probe: raw train + raw test, raw train + inference-time norm test, norm train + norm test, and raw train + norm-only probe.

**Tech Stack:** Python, PyTorch DDP, CLIP, sklearn metrics/logistic regression, pandas/csv, pytest.

---

## File Structure

- `src/deepfake_detection/data/index_ffpp.py`
  - Keep DF40 FF++/CDF JSON indexing.
  - Ensure records expose stable `sample_id`, `frame_path`, `video_id`, `label`, and method/domain metadata.
- `src/deepfake_detection/data/index_external.py`
  - Keep only DFD external folder indexing if needed for DFD test; remove FaceShifter-specific assumptions from experiment scripts.
- `src/deepfake_detection/data/datasets.py`
  - Return `sample_id` and `image_path` in every classification batch.
- `src/deepfake_detection/evaluation/sample_eval.py`
  - New core evaluation module: collect sample rows, all-gather rows, deduplicate, aggregate to video metrics, write CSV/JSON summaries.
- `src/deepfake_detection/evaluation/norm_modes.py`
  - New small module for raw, inference norm, partial norm, score fusion, and feature norm extraction.
- `src/deepfake_detection/engine/trainers.py`
  - Replace legacy eval path with calls into `sample_eval.py`; keep training path minimal.
- `evaluate.py`
  - Use the sample-level evaluator for FF++/CDF.
- `evaluate_norm_correction.py`
  - Slim to raw checkpoint inference sweeps: raw, inference norm, partial norm, fusion, FF++ val threshold calibration.
- `evaluate_norm_probe.py`
  - Keep magnitude-only probe; route its feature norm collection through the same sample metadata assumptions.
- `train.py`
  - Keep 8-GPU DDP training; ensure configs can train raw and norm models.
- `configs/exp_norm_raw_s*.yaml`
  - Create/keep raw-train configs for seeds.
- `configs/exp_norm_train_s*.yaml`
  - Create norm-train configs for seeds.
- `scripts/run_norm_shortcut_experiments.sh`
  - Orchestrate training and evaluation for seeds.
- `tests/data/test_sample_metadata.py`
  - Verify dataset records/batches include stable metadata.
- `tests/evaluation/test_sample_eval.py`
  - Verify deduplication, video aggregation, threshold metrics, and DDP-padding-like duplicates.
- `tests/evaluation/test_norm_modes.py`
  - Verify raw/norm/partial/fusion math.

---

### Task 1: Add Stable Sample Metadata

**Files:**
- Modify: `src/deepfake_detection/data/index_ffpp.py:7-72`
- Modify: `src/deepfake_detection/data/index_external.py:12-93`
- Modify: `src/deepfake_detection/data/datasets.py:47-65`
- Create: `tests/data/test_sample_metadata.py`

- [ ] **Step 1: Write failing tests for record and batch metadata**

Create `tests/data/test_sample_metadata.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from deepfake_detection.data.datasets import FrameClassificationDataset
from deepfake_detection.data.index_external import ExternalFrameRecord
from deepfake_detection.data.index_ffpp import FrameRecord


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_frame_record_sample_id_uses_path_label_and_video_id(tmp_path):
    frame = tmp_path / "video_a" / "frame_001.png"
    record = FrameRecord(
        method="Deepfakes",
        pair_id="video_a",
        frame_name="frame_001.png",
        frame_path=str(frame),
        landmark_path=None,
        label=1,
    )

    assert record.video_id == "video_a"
    assert record.sample_id == f"{frame}::1::video_a"


def test_external_record_sample_id_uses_path_label_and_video_id(tmp_path):
    frame = tmp_path / "fake" / "video_b" / "frame_001.jpg"
    record = ExternalFrameRecord(
        frame_path=str(frame),
        label=1,
        video_id="dfd/fake/video_b",
    )

    assert record.sample_id == f"{frame}::1::dfd/fake/video_b"


def test_classification_dataset_returns_audit_metadata(tmp_path):
    frame = tmp_path / "real" / "video_c" / "frame_001.jpg"
    _write_image(frame)
    record = ExternalFrameRecord(
        frame_path=str(frame),
        label=0,
        video_id="dfd/real/video_c",
    )
    dataset = FrameClassificationDataset([record], augment=False)

    item = dataset[0]

    assert item["label"].item() == 0
    assert item["video_id"] == "dfd/real/video_c"
    assert item["image_path"] == str(frame)
    assert item["sample_id"] == f"{frame}::0::dfd/real/video_c"
    assert tuple(item["image"].shape) == (3, 224, 224)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/data/test_sample_metadata.py -q
```

Expected: FAIL because `sample_id`, `video_id`, or `image_path` properties/fields are not yet available in all returned items.

- [ ] **Step 3: Add metadata properties to record classes**

Modify `src/deepfake_detection/data/index_ffpp.py` so `FrameRecord` becomes:

```python
@dataclass(frozen=True)
class FrameRecord:
    method: str
    pair_id: str
    frame_name: str
    frame_path: str
    landmark_path: str | None
    label: int

    @property
    def video_id(self) -> str:
        return self.pair_id

    @property
    def sample_id(self) -> str:
        return f"{self.frame_path}::{self.label}::{self.video_id}"
```

Modify `src/deepfake_detection/data/index_external.py` so `ExternalFrameRecord` becomes:

```python
@dataclass(frozen=True)
class ExternalFrameRecord:
    frame_path: str
    label: int
    video_id: str

    @property
    def sample_id(self) -> str:
        return f"{self.frame_path}::{self.label}::{self.video_id}"
```

- [ ] **Step 4: Return metadata from classification dataset**

Modify `FrameClassificationDataset.__getitem__` in `src/deepfake_detection/data/datasets.py` to return:

```python
    def __getitem__(self, idx):
        rec = self.records[idx]
        img = _load_frame(rec.frame_path)
        img = cv2_resize(img, self.img_size)
        img = self.transform(image=img)["image"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        video_id = rec.video_id if hasattr(rec, "video_id") else rec.pair_id
        sample_id = rec.sample_id if hasattr(rec, "sample_id") else f"{rec.frame_path}::{rec.label}::{video_id}"
        return {
            "image": tensor,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "video_id": video_id,
            "image_path": rec.frame_path,
            "sample_id": sample_id,
        }
```

- [ ] **Step 5: Run metadata tests**

Run:

```bash
PYTHONPATH=src pytest tests/data/test_sample_metadata.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/deepfake_detection/data/index_ffpp.py src/deepfake_detection/data/index_external.py src/deepfake_detection/data/datasets.py tests/data/test_sample_metadata.py
git commit -m "fix: add stable sample metadata for evaluation"
```

---

### Task 2: Implement Sample-Level Deduplicated Evaluation

**Files:**
- Create: `src/deepfake_detection/evaluation/sample_eval.py`
- Create: `tests/evaluation/test_sample_eval.py`

- [ ] **Step 1: Write failing tests for deduplication and metrics**

Create `tests/evaluation/test_sample_eval.py`:

```python
from deepfake_detection.evaluation.sample_eval import (
    deduplicate_sample_rows,
    sample_rows_to_video_predictions,
    summarize_sample_rows,
    video_metrics_from_rows,
)


def test_deduplicate_sample_rows_prefers_sample_id():
    rows = [
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "a", "image_path": "/x/a.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.1},
        {"sample_id": "b", "image_path": "/x/b.jpg", "video_id": "v1", "label": 0, "prob_fake": 0.3},
    ]

    deduped = deduplicate_sample_rows(rows)

    assert [row["sample_id"] for row in deduped] == ["a", "b"]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_sample_eval.py -q
```

Expected: FAIL because `sample_eval.py` does not exist.

- [ ] **Step 3: Implement sample evaluation helpers**

Create `src/deepfake_detection/evaluation/sample_eval.py`:

```python
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
        return {"auc": 0.0, "eer": 0.0, "acc": 0.0}
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
        json.dump({"summary": summary, "metrics": metrics}, f, indent=2)
```

- [ ] **Step 4: Run sample eval tests**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_sample_eval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/evaluation/sample_eval.py tests/evaluation/test_sample_eval.py
git commit -m "feat: add deduplicated sample-level evaluation"
```

---

### Task 3: Add Norm Mode Math

**Files:**
- Create: `src/deepfake_detection/evaluation/norm_modes.py`
- Create: `tests/evaluation/test_norm_modes.py`

- [ ] **Step 1: Write failing tests for norm modes**

Create `tests/evaluation/test_norm_modes.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_norm_modes.py -q
```

Expected: FAIL because `norm_modes.py` does not exist.

- [ ] **Step 3: Implement norm mode helpers**

Create `src/deepfake_detection/evaluation/norm_modes.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_inference_norm(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(features, dim=-1, eps=eps)


def apply_partial_norm(features: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    norms = features.norm(dim=-1, keepdim=True).clamp_min(eps)
    return features / norms.pow(alpha)


def fuse_scores(raw_scores: torch.Tensor, norm_scores: torch.Tensor, beta: float) -> torch.Tensor:
    return (1 - beta) * raw_scores + beta * norm_scores


def classifier_scores_from_features(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mode: str,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    feature_norm = features.norm(dim=-1)
    if mode == "raw":
        corrected = features
    elif mode == "norm":
        corrected = apply_inference_norm(features)
    elif mode == "partial":
        corrected = apply_partial_norm(features, alpha)
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    logits = corrected @ weight.t().to(features.device) + bias.to(features.device)
    prob_fake = torch.softmax(logits, dim=1)[:, 1]
    return {"logits": logits, "prob_fake": prob_fake, "feature_norm": feature_norm}
```

- [ ] **Step 4: Run norm mode tests**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_norm_modes.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/evaluation/norm_modes.py tests/evaluation/test_norm_modes.py
git commit -m "feat: add CLIP feature norm evaluation modes"
```

---

### Task 4: Wire Sample-Level Evaluator Into Runtime Evaluation

**Files:**
- Modify: `src/deepfake_detection/engine/trainers.py:117-157`
- Modify: `evaluate.py:15-42`
- Test: `tests/evaluation/test_sample_eval.py`

- [ ] **Step 1: Add a test for building rows from batch metadata**

Append to `tests/evaluation/test_sample_eval.py`:

```python
import torch

from deepfake_detection.evaluation.sample_eval import build_sample_rows


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_sample_eval.py::test_build_sample_rows_keeps_logits_probs_norms_and_paths -q
```

Expected: FAIL because `build_sample_rows` does not exist.

- [ ] **Step 3: Add `build_sample_rows` to sample eval**

Append this function to `src/deepfake_detection/evaluation/sample_eval.py`:

```python
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
```

- [ ] **Step 4: Add `run_sample_eval_epoch` to trainers**

Modify `src/deepfake_detection/engine/trainers.py` imports to include:

```python
from deepfake_detection.evaluation.sample_eval import (
    build_sample_rows,
    gather_rows_across_ranks,
    summarize_sample_rows,
    video_metrics_from_rows,
    write_eval_summary,
    write_sample_rows_csv,
)
```

Add this function above `run_eval_epoch`:

```python
@torch.no_grad()
def run_sample_eval_epoch(model, dataloader, device, cfg, output_dir: str | None = None, split_name: str = "eval"):
    model.eval()
    rows = []
    raw_model = model.module if hasattr(model, "module") else model
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            features = raw_model.visual(images.float()).float()
            logits = raw_model.classifier(features)
            prob_fake = torch.softmax(logits, dim=1)[:, 1]
            feature_norm = features.norm(dim=-1)
        rows.extend(build_sample_rows(batch, logits, prob_fake, feature_norm))

    rows = gather_rows_across_ranks(rows)
    summary = summarize_sample_rows(rows)
    metrics_05 = video_metrics_from_rows(rows, threshold=0.5)
    result = {**metrics_05, **summary, "loss": 0.0}
    if output_dir is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        write_sample_rows_csv(rows, f"{output_dir}/{split_name}_sample_rows.csv")
        write_eval_summary(summary, metrics_05, f"{output_dir}/{split_name}_summary.json")
    return result
```

Replace the body of existing `run_eval_epoch` with:

```python
@torch.no_grad()
def run_eval_epoch(model, dataloader, device, cfg):
    return run_sample_eval_epoch(model, dataloader, device, cfg)
```

- [ ] **Step 5: Run evaluation tests and existing metrics tests**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_sample_eval.py tests/engine/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-test `evaluate.py` on FF++ using existing checkpoint**

Run:

```bash
PYTHONPATH=src python evaluate.py \
  --config configs/exp2_fullffpp_ema_s2048.yaml \
  --checkpoint outputs/repro_exp2_to_exp3/checkpoints/exp2_fullffpp_ema_s2048_20260501_163927/best_model.pth \
  --domain ffpp
```

Expected: prints FF++ AUC/EER/ACC and does not crash.

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/engine/trainers.py src/deepfake_detection/evaluation/sample_eval.py evaluate.py tests/evaluation/test_sample_eval.py
git commit -m "fix: evaluate with deduplicated sample rows"
```

---

### Task 5: Add Trusted Norm-Correction Evaluation Script

**Files:**
- Modify: `evaluate_norm_correction.py`
- Test: `tests/evaluation/test_norm_modes.py`

- [ ] **Step 1: Define required CLI behavior**

Replace `evaluate_norm_correction.py` with a script that supports:

```text
--config
--checkpoint
--splits ffpp,cdf,dfd
--mode raw,norm,partial,fusion
--alphas 0,0.25,0.5,0.75,1
--betas 0,0.25,0.5,0.75,1
--ffpp-val-threshold
--output-dir
--batch-size
--num-workers
--prefetch-factor
```

The script must always write:

```text
metrics.csv
<split>_<variant>_sample_rows.csv
run_meta.json
```

- [ ] **Step 2: Implement score collection with visual features**

Use this core function in `evaluate_norm_correction.py`:

```python
@torch.no_grad()
def collect_norm_mode_rows(model, loader, device, weight, bias, scorer_name: str, scorer: dict):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            features = raw_model.visual(images.float()).float()
            if scorer["mode"] == "fusion":
                raw = classifier_scores_from_features(features, weight, bias, mode="raw")
                norm = classifier_scores_from_features(features, weight, bias, mode="norm")
                prob_fake = fuse_scores(raw["prob_fake"], norm["prob_fake"], scorer["beta"])
                logits = torch.stack([1 - prob_fake, prob_fake], dim=1)
                feature_norm = raw["feature_norm"]
            else:
                outputs = classifier_scores_from_features(
                    features,
                    weight,
                    bias,
                    mode=scorer["mode"],
                    alpha=scorer.get("alpha", 1.0),
                )
                logits = outputs["logits"]
                prob_fake = outputs["prob_fake"]
                feature_norm = outputs["feature_norm"]
        rows.extend(build_sample_rows(batch, logits, prob_fake, feature_norm))
    return gather_rows_across_ranks(rows)
```

- [ ] **Step 3: Compute FF++ val threshold only from validation rows**

Add this helper:

```python
def best_threshold_from_rows(rows: list[dict]) -> tuple[float, float]:
    labels, scores = sample_rows_to_video_predictions(deduplicate_sample_rows(rows))
    best_threshold = 0.5
    best_acc = -1.0
    for threshold in sorted(set(float(score) for score in scores)):
        acc = compute_acc(labels, scores, threshold)
        if acc > best_acc:
            best_threshold = threshold
            best_acc = acc
    return best_threshold, best_acc
```

- [ ] **Step 4: Emit metrics rows with dedup statistics**

For every split/variant append:

```python
summary = summarize_sample_rows(rows)
metrics_05 = video_metrics_from_rows(rows, threshold=0.5)
metrics_cal = video_metrics_from_rows(rows, threshold=ffpp_val_threshold)
result_rows.append({
    "split": split,
    "variant": variant,
    "param": param,
    "auc": metrics_05["auc"],
    "eer": metrics_05["eer"],
    "acc_0.5": metrics_05["acc"],
    "ffpp_val_threshold": ffpp_val_threshold,
    "ffpp_val_acc_at_threshold": ffpp_val_acc,
    "acc_ffpp_val_threshold": metrics_cal["acc"],
    **summary,
})
```

- [ ] **Step 5: Run unit tests**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_sample_eval.py tests/evaluation/test_norm_modes.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-test raw and norm modes on FF++**

Run:

```bash
PYTHONPATH=src torchrun --nproc_per_node=8 evaluate_norm_correction.py \
  --config configs/exp2_fullffpp_ema_s2048.yaml \
  --checkpoint outputs/repro_exp2_to_exp3/checkpoints/exp2_fullffpp_ema_s2048_20260501_163927/best_model.pth \
  --splits ffpp \
  --mode raw,norm \
  --batch-size 128 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --output-dir outputs/smoke_norm_eval
```

Expected: output directory contains `metrics.csv`, sample rows CSVs, and dedup stats where `rows_after_dedup <= rows_before_dedup`.

- [ ] **Step 7: Commit**

```bash
git add evaluate_norm_correction.py src/deepfake_detection/evaluation/norm_modes.py src/deepfake_detection/evaluation/sample_eval.py tests/evaluation/test_norm_modes.py tests/evaluation/test_sample_eval.py
git commit -m "feat: add auditable norm correction evaluation"
```

---

### Task 6: Clean Unrelated Experiment Code

**Files:**
- Inspect and possibly delete:
  - `evaluate_cosine_inference.py`
  - `evaluate_norm_probe.py`
  - `scripts/eval_alpha_sweep_rawcls_normprompt.py`
  - `scripts/run_exp*.sh`
  - stale configs under `configs/exp3_*` and `configs/exp4_*`
- Keep:
  - `train.py`
  - `evaluate.py`
  - `evaluate_norm_correction.py`
  - `evaluate_norm_probe.py` if it is slimmed to magnitude-only probe
  - `scripts/download_faceforensics.py` while DFD video download is needed

- [ ] **Step 1: List unrelated files before deleting**

Run:

```bash
git status --short
find configs scripts -maxdepth 1 -type f | sort
```

Expected: identify files unrelated to this paper experiment. Do not delete user data directories.

- [ ] **Step 2: Keep cosine inference only as negative-result script or delete it**

If cosine inference is not needed as a separate script, remove `evaluate_cosine_inference.py`. If kept, rename/trim its output as a negative-result baseline and ensure it uses sample-level evaluation.

Recommended command:

```bash
git rm evaluate_cosine_inference.py
```

Expected: file staged for deletion.

- [ ] **Step 3: Delete stale exp3/exp4 shell scripts and configs**

Delete only files that are not part of the final experiment matrix:

```bash
git rm -f configs/exp3_*.yaml configs/exp4_*.yaml scripts/run_exp3*.sh scripts/run_exp4*.sh scripts/eval_alpha_sweep_rawcls_normprompt.py
```

Expected: stale prompt/background/old alpha scripts removed from git.

- [ ] **Step 4: Run tests after cleanup**

Run:

```bash
PYTHONPATH=src pytest tests -q
```

Expected: PASS, or only failures from tests that explicitly reference deleted experimental files. If a test references deleted files, update the test to the new experiment script names.

- [ ] **Step 5: Commit cleanup**

```bash
git add -u configs scripts evaluate_cosine_inference.py tests
git commit -m "chore: remove unrelated experiment code"
```

---

### Task 7: Add Final Experiment Configs and Runner

**Files:**
- Create: `configs/norm_shortcut_raw_s42.yaml`
- Create: `configs/norm_shortcut_norm_s42.yaml`
- Create analogous seed configs for `7,123,999,2048`
- Create: `scripts/run_norm_shortcut_experiments.sh`

- [ ] **Step 1: Create raw-train seed config template**

Create `configs/norm_shortcut_raw_s42.yaml`:

```yaml
_base_: configs/exp2_fullffpp_ema_s2048.yaml
experiment_name: norm_shortcut_raw_s42
model:
  name: clip_vit_b16
  normalize_features: false
train:
  seed: 42
  per_gpu_batch: 128
  num_workers: 8
  prefetch_factor: 4
  output_dir: outputs/norm_shortcut
```

- [ ] **Step 2: Create norm-train seed config template**

Create `configs/norm_shortcut_norm_s42.yaml`:

```yaml
_base_: configs/exp2_fullffpp_ema_s2048.yaml
experiment_name: norm_shortcut_norm_s42
model:
  name: clip_vit_b16
  normalize_features: true
train:
  seed: 42
  per_gpu_batch: 128
  num_workers: 8
  prefetch_factor: 4
  output_dir: outputs/norm_shortcut
```

- [ ] **Step 3: Create remaining seed configs**

Create these files by replacing `42` with each seed in both `experiment_name` and `train.seed`:

```text
configs/norm_shortcut_raw_s7.yaml
configs/norm_shortcut_raw_s123.yaml
configs/norm_shortcut_raw_s999.yaml
configs/norm_shortcut_raw_s2048.yaml
configs/norm_shortcut_norm_s7.yaml
configs/norm_shortcut_norm_s123.yaml
configs/norm_shortcut_norm_s999.yaml
configs/norm_shortcut_norm_s2048.yaml
```

- [ ] **Step 4: Create runner script**

Create `scripts/run_norm_shortcut_experiments.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 7 123 999 2048)
NPROC=8
BATCH=128
WORKERS_TRAIN=8
WORKERS_EVAL=8
PREFETCH=4

for seed in "${SEEDS[@]}"; do
  PYTHONPATH=src torchrun --nproc_per_node="$NPROC" train.py \
    --config "configs/norm_shortcut_raw_s${seed}.yaml"

  PYTHONPATH=src torchrun --nproc_per_node="$NPROC" train.py \
    --config "configs/norm_shortcut_norm_s${seed}.yaml"
done
```

- [ ] **Step 5: Add eval runner commands to script**

Append to `scripts/run_norm_shortcut_experiments.sh`:

```bash
for seed in "${SEEDS[@]}"; do
  RAW_CKPT=$(find outputs/norm_shortcut/norm_shortcut_raw_s"${seed}" -name best_model.pth | sort | tail -1)
  NORM_CKPT=$(find outputs/norm_shortcut/norm_shortcut_norm_s"${seed}" -name best_model.pth | sort | tail -1)

  PYTHONPATH=src torchrun --nproc_per_node="$NPROC" evaluate_norm_correction.py \
    --config "configs/norm_shortcut_raw_s${seed}.yaml" \
    --checkpoint "$RAW_CKPT" \
    --splits ffpp,cdf,dfd \
    --mode raw,norm,partial,fusion \
    --batch-size "$BATCH" \
    --num-workers "$WORKERS_EVAL" \
    --prefetch-factor "$PREFETCH" \
    --output-dir "outputs/norm_shortcut_eval/raw_s${seed}"

  PYTHONPATH=src torchrun --nproc_per_node="$NPROC" evaluate_norm_correction.py \
    --config "configs/norm_shortcut_norm_s${seed}.yaml" \
    --checkpoint "$NORM_CKPT" \
    --splits ffpp,cdf,dfd \
    --mode norm \
    --batch-size "$BATCH" \
    --num-workers "$WORKERS_EVAL" \
    --prefetch-factor "$PREFETCH" \
    --output-dir "outputs/norm_shortcut_eval/norm_s${seed}"

  PYTHONPATH=src torchrun --nproc_per_node="$NPROC" evaluate_norm_probe.py \
    --config "configs/norm_shortcut_raw_s${seed}.yaml" \
    --checkpoint "$RAW_CKPT" \
    --batch-size "$BATCH" \
    --num-workers "$WORKERS_EVAL" \
    --prefetch-factor "$PREFETCH" \
    --output-dir "outputs/norm_shortcut_probe/raw_s${seed}"
done
```

- [ ] **Step 6: Make script executable and run syntax checks**

Run:

```bash
chmod +x scripts/run_norm_shortcut_experiments.sh
bash -n scripts/run_norm_shortcut_experiments.sh
```

Expected: no syntax errors.

- [ ] **Step 7: Commit**

```bash
git add configs/norm_shortcut_*.yaml scripts/run_norm_shortcut_experiments.sh
git commit -m "feat: add norm shortcut experiment configs"
```

---

### Task 8: Generate Paper Tables and Analysis Outputs

**Files:**
- Create: `scripts/summarize_norm_shortcut_results.py`
- Output:
  - `outputs/norm_shortcut_tables/main_results.csv`
  - `outputs/norm_shortcut_tables/ablation_results.csv`
  - `outputs/norm_shortcut_tables/probe_results.csv`
  - `outputs/norm_shortcut_tables/paired_comparison.csv`

- [ ] **Step 1: Create summarizer script**

Create `scripts/summarize_norm_shortcut_results.py`:

```python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.glob("**/metrics.csv"):
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_method_column(df: pd.DataFrame) -> pd.DataFrame:
    def method(row):
        variant = str(row.get("variant", ""))
        source = str(row.get("source_file", ""))
        if "norm_s" in source and variant == "norm":
            return "norm_train_norm_test"
        if variant == "raw":
            return "direction_plus_magnitude"
        if variant == "norm":
            return "direction"
        if variant.startswith("partial"):
            return "partial_norm"
        if variant.startswith("fusion"):
            return "score_fusion"
        return variant
    df = df.copy()
    df["method"] = df.apply(method, axis=1)
    return df


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["auc", "eer", "acc_0.5", "acc_ffpp_val_threshold"]
    grouped = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped


def paired_direction_gain(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["method"].isin(["direction", "direction_plus_magnitude"])]
    pivot = main.pivot_table(index=["seed", "split"], columns="method", values="auc", aggfunc="first").reset_index()
    pivot["auc_gain_direction_minus_raw"] = pivot["direction"] - pivot["direction_plus_magnitude"]
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="outputs/norm_shortcut_eval")
    parser.add_argument("--output-dir", default="outputs/norm_shortcut_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_metrics(Path(args.eval_root))
    if df.empty:
        raise SystemExit(f"No metrics.csv found under {args.eval_root}")
    df = add_method_column(df)
    df.to_csv(output_dir / "all_metrics.csv", index=False)

    main_df = df[df["method"].isin(["direction", "direction_plus_magnitude", "norm_train_norm_test"])]
    summarize_mean_std(main_df, ["split", "method"]).to_csv(output_dir / "main_results.csv", index=False)

    ablation_df = df[df["method"].isin(["partial_norm", "score_fusion"])]
    summarize_mean_std(ablation_df, ["split", "method", "param"]).to_csv(output_dir / "ablation_results.csv", index=False)

    paired_direction_gain(df).to_csv(output_dir / "paired_comparison.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run summarizer after experiment outputs exist**

Run:

```bash
PYTHONPATH=src python scripts/summarize_norm_shortcut_results.py \
  --eval-root outputs/norm_shortcut_eval \
  --output-dir outputs/norm_shortcut_tables
```

Expected: CSV files are written to `outputs/norm_shortcut_tables`.

- [ ] **Step 3: Inspect paired comparison**

Run:

```bash
python - <<'PY'
import pandas as pd
p = 'outputs/norm_shortcut_tables/paired_comparison.csv'
df = pd.read_csv(p)
print(df.groupby('split')['auc_gain_direction_minus_raw'].agg(['mean', 'std', 'min', 'max']))
PY
```

Expected: prints per-split paired AUC gains.

- [ ] **Step 4: Commit summarizer**

```bash
git add scripts/summarize_norm_shortcut_results.py
git commit -m "feat: summarize norm shortcut experiment results"
```

---

### Task 9: Run the Final Experiments

**Files:**
- Use: `scripts/run_norm_shortcut_experiments.sh`
- Use: `scripts/summarize_norm_shortcut_results.py`

- [ ] **Step 1: Stop unrelated background dataset preparation if it is still running**

Check background tasks and stop only DFD download/prep tasks if the user confirms they are not needed right now. Do not stop training/evaluation tasks without confirmation.

- [ ] **Step 2: Run all training and evaluation jobs**

Run:

```bash
scripts/run_norm_shortcut_experiments.sh
```

Expected: raw and norm checkpoints for seeds `42,7,123,999,2048`, then eval outputs for FF++/CDF/DFD.

- [ ] **Step 3: Summarize results**

Run:

```bash
PYTHONPATH=src python scripts/summarize_norm_shortcut_results.py \
  --eval-root outputs/norm_shortcut_eval \
  --output-dir outputs/norm_shortcut_tables
```

Expected: final tables written.

- [ ] **Step 4: Report required outputs**

Report:

```text
1. Code audit and removed unrelated code.
2. Per-dataset, per-seed, per-method AUC/EER/ACC@0.5/ACC@FF++ val threshold.
3. Dedup before/after sample counts.
4. Main table, ablation table, norm-only probe result.
5. Conclusion: CLIP feature direction is more robust cross-domain than feature magnitude; feature norm can carry shortcut; inference-time norm correction improves cross-domain generalization.
```

---

## Self-Review

- Spec coverage: The plan covers dataset metadata, DDP eval deduplication, sample row saving, raw/norm/partial/fusion modes, threshold calibration, code cleanup, three detector settings plus norm-only probe, 5-seed runner, and result summaries.
- Placeholder scan: No TBD/TODO placeholders remain. Commands and expected outputs are explicit.
- Type consistency: `sample_id`, `image_path`, `video_id`, `label`, `logit_real`, `logit_fake`, `prob_fake`, and `feature_norm` are consistently used across tasks.
