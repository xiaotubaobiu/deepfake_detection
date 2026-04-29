# Validation Set + Exp1/Exp2 Re-run Design

## Problem

Current training uses FF++ test set for early stopping (monitoring AUC each epoch), then evaluates on the same FF++ test set. This means the model selection is biased toward FF++ test performance. Exp1 reached AUC=0.9996 on FF++ which is likely inflated because early stopping was done on the same data.

## Solution

Add a proper validation set split from training data. Use validation AUC for early stopping, then independently evaluate the best model on FF++ and CDF test sets.

## Data Split

Each method's training videos (sorted, 700-988 available per method):

| Split | Videos | Source |
|-------|--------|--------|
| Train | 1-200 per method | `DF40_train/<method>/frames/` (first 200) |
| Val | 201-300 per method | `DF40_train/<method>/frames/` (next 100) |
| Test FF++ | official test split | `DF40_test/<method>/ff/frames/` |
| Test CDF | official test split | `DF40_test/<method>/cdf/frames/` |

Val set size: 16 methods x 100 videos x 8 frames = 12800 frames (6400 fake + 6400 real balanced)

## Training Flow

1. Each epoch: train on train set, evaluate on **val set** for AUC
2. Val AUC no improvement for `patience` epochs → early stop, save best model
3. After training: load best model, independently evaluate on FF++ test and CDF test

## Experiments

### Exp1: EfficientNet-B0 (re-run)
- Add validation set
- Train with val-based early stopping
- Save best model based on val AUC
- Evaluate best model on FF++ and CDF test sets independently
- Save results

### Exp2: CLIP ViT-B/16 fine-tune
- Same validation set and early stopping
- First run: measure GPU memory with small batch
- Calculate optimal per_gpu_batch to fill 8xV100 32GB (target ~28GB per GPU)
- Re-run with optimized batch
- Debug until training runs normally
- Evaluate best model on FF++ and CDF test sets
- Save results

## Code Changes

### 1. `src/deepfake_detection/data/index_ffpp.py`

Add function:
```python
def index_val_method_frames(root, method, val_videos_per_method=100):
    frames_dir = Path(root) / "DF40_train" / method / "frames"
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    # Skip first 200 (train), take next 100 (val)
    val_dirs = video_dirs[200:200 + val_videos_per_method]
    # ... same pattern as index_train_method_frames
```

### 2. `src/deepfake_detection/data/builders.py`

Add function:
```python
def build_val_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    val_videos = cfg["dataset"].get("val_videos_per_method", 100)
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)

    fake_records = []
    for method in methods:
        recs = index_val_method_frames(root, method, val_videos)
        fake_records.extend(recs)
    fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

    real_records = index_train_real_frames(root)
    real_by_video_full = _subsample_records_to_frames(real_records, frames_per_video)
    target_real_count = len(fake_by_video)
    rng = random.Random(42)
    real_balanced = []
    while len(real_balanced) < target_real_count:
        real_balanced.extend(rng.sample(real_by_video_full, min(len(real_by_video_full), target_real_count - len(real_balanced))))

    all_records = real_balanced[:target_real_count] + fake_by_video
    dataset = FrameClassificationDataset(all_records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
```

### 3. `train.py`

Replace ffpp_eval_loader in training loop with val_loader:
```python
val_loader = build_val_loader(cfg, distributed=torch.distributed.is_initialized())

# In training loop:
val_metrics = run_eval_epoch(model, val_loader, device, cfg)
# Early stop on val AUC instead of ffpp AUC

# After training (on main process):
# Load best model
# Run FF++ test eval
# Run CDF test eval
# Save all results
```

### 4. `configs/base.yaml`

Add:
```yaml
val_videos_per_method: 100
```

## Results Format

For each experiment, save to `outputs/<exp_name>/results_summary.txt`:
```
ExpN <model_name>
Val: AUC=X.XXXX EER=X.XXXX ACC=X.XXXX
FF++: AUC=X.XXXX EER=X.XXXX ACC=X.XXXX
CDF:  AUC=X.XXXX EER=X.XXXX ACC=X.XXXX
Best epoch: N
Per-GPU batch: X
```
