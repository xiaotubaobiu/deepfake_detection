# Fix Data Leakage & JSON Index Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three data leakage bugs (val=train real, CDF domain mismatch, FF++ identity overlap) and replace directory scanning with JSON-based indexing for 15x speedup.

**Architecture:** Replace all `Path.iterdir()` + `glob("*.png")` indexing with JSON pre-built indexes from `dataset_json/`. JSON files already contain correct real data sources (FF++ original for FF++ domain, Celeb-DF-v2 for CDF domain). Apply path remapping and video-range filtering to split train/val.

**Tech Stack:** Python 3, PyTorch, existing codebase

---

### Task 1: Add JSON-based indexing functions

**Files:**
- Modify: `src/deepfake_detection/data/index_ffpp.py`

- [ ] **Step 1: Add `load_json_index` function and path remapping**

Add these functions at the top of `index_ffpp.py` (after imports, before existing `FrameRecord`):

```python
import json

def _remap_path(json_path: str, root: str) -> str:
    """Remap JSON path to actual filesystem path.

    JSON paths start with 'deepfakes_detection_datasets/'.
    'DF40/' in test paths maps to 'DF40_test/' on disk.
    """
    rel = json_path.replace("deepfakes_detection_datasets/", "")
    rel = rel.replace("DF40/", "DF40_test/", 1)  # only first occurrence
    return str(Path(root) / rel)


def load_json_index(
    root: str,
    method: str,
    domain: str,
    split: str,
    label: int,
    max_videos: int | None = None,
    video_range: tuple[int, int] | None = None,
) -> list[FrameRecord]:
    """Load frame records from a pre-built JSON index file.

    Args:
        root: dataset root (e.g. /Dataset/deepfake_detection/DF40_all)
        method: forgery method name (e.g. 'simswap')
        domain: 'ff' or 'cdf'
        split: 'train', 'val', or 'test'
        label: 0 for real, 1 for fake
        max_videos: take first N videos (by sorted order)
        video_range: (start, end) tuple to slice video list by index
    """
    json_dir = Path(root) / "dataset_json"
    json_path = json_dir / f"{method}_{domain}.json"
    if not json_path.exists():
        return []

    with open(json_path) as f:
        data = json.load(f)

    top_key = f"{method}_{domain}"
    category = f"{method}_Real" if label == 0 else f"{method}_Fake"
    section = data.get(top_key, {}).get(category, {}).get(split, {})
    if not section:
        return []

    video_ids = sorted(section.keys())
    if video_range is not None:
        video_ids = video_ids[video_range[0]:video_range[1]]
    elif max_videos is not None:
        video_ids = video_ids[:max_videos]

    records = []
    for vid_id in video_ids:
        vid_data = section[vid_id]
        frames = vid_data.get("frames", [])
        landmarks = vid_data.get("landmarks", [])
        for i, frame_path in enumerate(frames):
            actual_path = _remap_path(frame_path, root)
            lm_path = None
            if i < len(landmarks) and landmarks[i]:
                lm_path = _remap_path(landmarks[i], root)
            records.append(FrameRecord(
                method=method if label == 1 else "real",
                pair_id=vid_id,
                frame_name=Path(actual_path).name,
                frame_path=actual_path,
                landmark_path=lm_path,
                label=label,
            ))
    return records
```

- [ ] **Step 2: Verify the function works**

Run: `python3 -c "from deepfake_detection.data.index_ffpp import load_json_index; recs = load_json_index('/Dataset/deepfake_detection/DF40_all', 'simswap', 'ff', 'train', 0, video_range=(0,5)); print(f'{len(recs)} records'); print(recs[0])"`
Expected: ~160 records (5 videos × 32 frames), showing FrameRecord with correct paths

- [ ] **Step 3: Verify fake index and CDF index**

Run:
```bash
python3 -c "
from deepfake_detection.data.index_ffpp import load_json_index
root = '/Dataset/deepfake_detection/DF40_all'
# FF++ fake
fake = load_json_index(root, 'simswap', 'ff', 'train', 1, max_videos=5)
print(f'FF fake: {len(fake)} records, sample: {fake[0]}')
# CDF real
cdf_real = load_json_index(root, 'simswap', 'cdf', 'test', 0)
print(f'CDF real: {len(cdf_real)} records, sample: {cdf_real[0]}')
# Verify paths exist
import os
print(f'FF fake path exists: {os.path.exists(fake[0].frame_path)}')
print(f'CDF real path exists: {os.path.exists(cdf_real[0].frame_path)}')
"
```
Expected: All paths exist=True, CDF real comes from Celeb-DF-v2 directory

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/data/index_ffpp.py
git commit -m "feat: add JSON-based indexing with path remapping"
```

---

### Task 2: Rewrite builders.py to use JSON indexes

**Files:**
- Modify: `src/deepfake_detection/data/builders.py`

- [ ] **Step 1: Replace `build_train_loader` real/fake indexing**

Replace the entire `build_train_loader` function body with JSON-based indexing:

```python
def build_train_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    require_triplets = cfg.get("data", {}).get("require_aligned_triplets", False)

    if require_triplets:
        all_triplets = []
        for method in methods:
            triplets = index_train_aligned_triplets(root, method, max_videos)
            all_triplets.extend(triplets)
        dataset = AlignedTripletDataset(all_triplets, augment=True)
    else:
        # Collect fake records from JSON: first max_videos per method
        fake_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 1, max_videos=max_videos)
            fake_records.extend(recs)
        fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

        # Collect real records from JSON: FF++ original, aligned range
        real_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 0, max_videos=max_videos)
            real_records.extend(recs)
        real_deduped = _deduplicate_by_pair_id(real_records)
        real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
        target_real_count = len(fake_by_video)
        rng = random.Random(42)
        real_balanced = []
        while len(real_balanced) < target_real_count:
            real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

        all_records = real_balanced[:target_real_count] + fake_by_video
        dataset = FrameClassificationDataset(all_records, augment=True)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True)
```

Also add the `_deduplicate_by_pair_id` helper:

```python
def _deduplicate_by_pair_id(records):
    seen = set()
    result = []
    for r in records:
        if r.pair_id not in seen:
            seen.add(r.pair_id)
            result.append(r)
    return result
```

- [ ] **Step 2: Replace `build_val_loader`**

```python
def build_val_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    val_videos = cfg["dataset"].get("val_videos_per_method", 100)
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)

    # Val fake: videos [max_videos : max_videos + val_videos] per method
    max_videos = cfg["dataset"]["train_videos_per_method"]
    fake_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 1,
                               video_range=(max_videos, max_videos + val_videos))
        fake_records.extend(recs)
    fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

    # Val real: FF++ original, same range — NO overlap with training real
    real_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 0,
                               video_range=(max_videos, max_videos + val_videos))
        real_records.extend(recs)
    real_deduped = _deduplicate_by_pair_id(real_records)
    real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
    target_real_count = len(fake_by_video)
    rng = random.Random(42)
    real_balanced = []
    while len(real_balanced) < target_real_count:
        real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

    all_records = real_balanced[:target_real_count] + fake_by_video
    dataset = FrameClassificationDataset(all_records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
```

- [ ] **Step 3: Replace `build_eval_loader` — use CDF JSON for CDF domain**

```python
def build_eval_loader(cfg, domain="ffpp", distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)

    test_domain = "ff" if domain == "ffpp" else "cdf"

    # Real: from JSON test split (CDF JSON uses Celeb-DF-v2 real)
    real_records = []
    for method in methods:
        recs = load_json_index(root, method, test_domain, "test", 0)
        real_records.extend(recs)
    real_deduped = _deduplicate_by_pair_id(real_records)

    # Fake: from JSON test split
    fake_records = []
    for method in methods:
        recs = load_json_index(root, method, test_domain, "test", 1)
        fake_records.extend(recs)

    all_records = real_deduped + fake_records
    dataset = FrameClassificationDataset(all_records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
```

- [ ] **Step 4: Update imports in builders.py**

Replace the old imports with:

```python
from deepfake_detection.data.index_ffpp import load_json_index
from deepfake_detection.data.index_ffpp import (
    index_train_aligned_triplets,
)
```

- [ ] **Step 5: Verify imports resolve**

Run: `python3 -c "from deepfake_detection.data.builders import build_train_loader, build_val_loader, build_eval_loader; print('OK')"`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add src/deepfake_detection/data/builders.py
git commit -m "feat: rewrite builders to use JSON indexes, fix data leakage"
```

---

### Task 3: Verify data correctness end-to-end

**Files:**
- No code changes, verification only

- [ ] **Step 1: Load all three splits and verify no overlap**

```bash
python3 -c "
import yaml, random
from deepfake_detection.data.builders import build_train_loader, build_val_loader, build_eval_loader

with open('configs/base.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model'] = {'name': 'efficientnet_b0'}

# Check train/val pair_id overlap
from deepfake_detection.data.index_ffpp import load_json_index
root = cfg['dataset']['root']
methods = ['simswap']  # spot check one method

train_real = load_json_index(root, 'simswap', 'ff', 'train', 0, max_videos=200)
val_real = load_json_index(root, 'simswap', 'ff', 'train', 0, video_range=(200, 300))
train_ids = set(r.pair_id for r in train_real)
val_ids = set(r.pair_id for r in val_real)
overlap = train_ids & val_ids
print(f'Train real pair_ids: {len(train_ids)}, Val real pair_ids: {len(val_ids)}, Overlap: {len(overlap)}')
assert len(overlap) == 0, f'Train/Val real overlap: {overlap}'

# Check CDF test real source
cdf_real = load_json_index(root, 'simswap', 'cdf', 'test', 0)
print(f'CDF real: {len(cdf_real)} records')
print(f'CDF real sample path: {cdf_real[0].frame_path}')
assert 'Celeb-DF-v2' in cdf_real[0].frame_path, 'CDF real should be from Celeb-DF-v2'

print('All checks passed!')
"
```
Expected: Overlap=0, CDF real path contains Celeb-DF-v2

- [ ] **Step 2: Verify frame paths are all valid**

```bash
python3 -c "
import os, yaml
from deepfake_detection.data.index_ffpp import load_json_index

with open('configs/base.yaml') as f:
    cfg = yaml.safe_load(f)
root = cfg['dataset']['root']

# Check a sample of paths from each source
checks = [
    ('simswap', 'ff', 'train', 0, 5),
    ('simswap', 'ff', 'train', 1, 5),
    ('simswap', 'cdf', 'test', 0, 5),
    ('simswap', 'cdf', 'test', 1, 5),
]
for method, domain, split, label, n in checks:
    recs = load_json_index(root, method, domain, split, label, max_videos=n)
    for r in recs[:3]:
        exists = os.path.exists(r.frame_path)
        if not exists:
            print(f'MISSING: {r.frame_path}')
    print(f'{method}/{domain}/{split}/label={label}: {len(recs)} records, paths OK')

print('Path verification done')
"
```
Expected: All paths exist

- [ ] **Step 3: Commit verification results**

No commit needed — this is a verification step only.

---

### Task 4: Clean up old indexing functions

**Files:**
- Modify: `src/deepfake_detection/data/index_ffpp.py`

- [ ] **Step 1: Remove unused old functions**

Remove these functions that are no longer called:
- `index_train_method_frames`
- `index_val_method_frames`
- `index_train_real_frames`
- `index_test_real_frames`
- `index_test_method_frames`

Keep:
- `FrameRecord` dataclass
- `build_aligned_pair_key`
- `index_train_aligned_triplets` (still used for triplet mode)
- `load_json_index`
- `_remap_path`

- [ ] **Step 2: Verify nothing is broken**

Run: `python3 -c "from deepfake_detection.data.builders import build_train_loader, build_val_loader, build_eval_loader; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/data/index_ffpp.py src/deepfake_detection/data/builders.py
git commit -m "chore: remove old directory-scan indexing functions"
```

---

### Task 5: Re-run exp1 with fixed data pipeline

**Files:**
- No code changes, experiment execution

- [ ] **Step 1: Run exp1 training**

```bash
torchrun --nproc_per_node=8 --master_port 29500 train.py --config configs/exp1_efficientnet.yaml 2>&1 | tee /tmp/exp1_fixed.log
```

- [ ] **Step 2: Verify results are realistic**

After training completes, check `outputs/exp1_efficientnet/results_summary.txt`:
- Val AUC should be < 1.0 (previously was exactly 1.0 due to leakage)
- CDF AUC should be significantly < 1.0 (previously was 1.0 due to domain mismatch)
- FF++ AUC should be lower than 0.9995 (previously inflated by identity overlap in real class)

- [ ] **Step 3: Commit results**

```bash
git add -f outputs/exp1_efficientnet/results_summary.txt outputs/exp1_efficientnet/best_model.pth
git commit -m "feat: exp1 results with fixed data pipeline (no leakage)"
```
