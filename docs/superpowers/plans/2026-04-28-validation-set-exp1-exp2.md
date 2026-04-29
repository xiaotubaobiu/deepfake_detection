# Validation Set + Exp1/Exp2 Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proper validation set (videos 201-300 per method) for early stopping, re-run exp1 and exp2 with unbiased evaluation on FF++ and CDF test sets.

**Architecture:** Split each method's training videos into train (1-200) and val (201-300). Modify training loop to use val AUC for early stopping. After training, load best checkpoint and independently evaluate on FF++ and CDF test sets. For exp2, profile GPU memory first, then run with optimal batch size.

**Tech Stack:** PyTorch DDP, 8xV100 32GB, EfficientNet-B0 (exp1), CLIP ViT-B/16 (exp2)

---

### Task 1: Add validation set indexing to index_ffpp.py

**Files:**
- Modify: `src/deepfake_detection/data/index_ffpp.py`

- [ ] **Step 1: Add `index_val_method_frames` function**

Add the following function after `index_train_method_frames` (after line 43):

```python
def index_val_method_frames(root: str, method: str, val_videos_per_method: int = 100) -> list[FrameRecord]:
    frames_dir = Path(root) / "DF40_train" / method / "frames"
    if not frames_dir.is_dir():
        return []
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    val_dirs = video_dirs[200:200 + val_videos_per_method]
    records = []
    for video_dir in val_dirs:
        pair_id = video_dir.name
        lm_dir = Path(root) / "DF40_train" / method / "landmarks" / pair_id
        for frame_path in sorted(video_dir.glob("*.png")):
            lm_file = lm_dir / (frame_path.stem + ".npy") if lm_dir.is_dir() else None
            records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=str(lm_file) if lm_file and lm_file.exists() else None,
                label=1,
            ))
    return records
```

- [ ] **Step 2: Verify the function works**

Run:
```bash
PYTHONPATH=src /home/z/anaconda3/envs/deepfake-detection/bin/python -c "
from deepfake_detection.data.index_ffpp import index_val_method_frames
recs = index_val_method_frames('/Dataset/deepfake_detection/DF40_all', 'simswap', 100)
print(f'simswap val: {len(recs)} frames from {len(set(r.pair_id for r in recs))} videos')
assert len(set(r.pair_id for r in recs)) == 100, 'Should have 100 videos'
recs2 = index_val_method_frames('/Dataset/deepfake_detection/DF40_all', 'fomm', 100)
print(f'fomm val: {len(recs2)} frames from {len(set(r.pair_id for r in recs2))} videos')
print('OK')
"
```
Expected: `simswap val: 3200 frames from 100 videos`, `fomm val: 3200 frames from 100 videos`, `OK`

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/data/index_ffpp.py
git commit -m "feat: add index_val_method_frames for validation set indexing"
```

---

### Task 2: Add build_val_loader to builders.py and update config

**Files:**
- Modify: `src/deepfake_detection/data/builders.py`
- Modify: `configs/base.yaml`

- [ ] **Step 1: Add import for `index_val_method_frames`**

At line 9-10 in `builders.py`, add `index_val_method_frames` to the import:

```python
from deepfake_detection.data.index_ffpp import (
    index_train_method_frames,
    index_train_real_frames,
    index_train_aligned_triplets,
    index_test_method_frames,
    index_test_real_frames,
    index_val_method_frames,
)
```

- [ ] **Step 2: Add `build_val_loader` function**

Add after `build_eval_loader` (after line 88):

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

- [ ] **Step 3: Add `val_videos_per_method` to base.yaml**

In `configs/base.yaml`, add after `frames_per_video: 8` (line 5):

```yaml
  val_videos_per_method: 100
```

So the dataset section becomes:
```yaml
dataset:
  root: /Dataset/deepfake_detection/DF40_all
  methods: []
  train_videos_per_method: 200
  frames_per_video: 8
  val_videos_per_method: 100
  image_size: 224
  real_sampling: oversample_per_method
  use_mask: false
```

- [ ] **Step 4: Verify build_val_loader works**

Run:
```bash
PYTHONPATH=src /home/z/anaconda3/envs/deepfake-detection/bin/python -c "
import yaml
from deepfake_detection.data.builders import build_val_loader
with open('configs/exp1_efficientnet.yaml') as f:
    raw = yaml.safe_load(f)
with open('configs/base.yaml') as f:
    base = yaml.safe_load(f)
cfg = {**base, **raw}
for k in set(list(base.keys()) + list(raw.keys())):
    if k in base and k in raw and isinstance(base[k], dict) and isinstance(raw[k], dict):
        cfg[k] = {**base[k], **raw[k]}
loader = build_val_loader(cfg, distributed=False)
print(f'Val samples: {len(loader.dataset)}')
print('OK')
"
```
Expected: `Val samples: 25600` (16 methods × 100 videos × 8 frames × 2 balanced = 25600), then `OK`

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/builders.py configs/base.yaml
git commit -m "feat: add build_val_loader and val_videos_per_method config"
```

---

### Task 3: Modify train.py to use validation set for early stopping

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add import for build_val_loader**

At line 19, change:
```python
from deepfake_detection.data.builders import build_train_loader, build_eval_loader
```
to:
```python
from deepfake_detection.data.builders import build_train_loader, build_eval_loader, build_val_loader
```

- [ ] **Step 2: Replace training loop to use val_loader**

Replace the block from line 112 to line 151 (the data loader creation and training loop) with:

```python
    train_loader = build_train_loader(cfg, distributed=torch.distributed.is_initialized())
    val_loader = build_val_loader(cfg, distributed=torch.distributed.is_initialized())
    ffpp_eval_loader = build_eval_loader(cfg, domain="ffpp", distributed=torch.distributed.is_initialized())
    cdf_eval_loader = build_eval_loader(cfg, domain="cdf", distributed=torch.distributed.is_initialized())

    if logger:
        logger.log(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, "
                   f"FF++ test: {len(ffpp_eval_loader.dataset)}, CDF test: {len(cdf_eval_loader.dataset)}")

    best_auc = 0
    patience_counter = 0
    best_epoch = 0
    save_dir = os.path.join(output_dir, exp_name) if not logger else os.path.dirname(os.path.dirname(logger.log_path))

    for epoch in range(epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, optimizer, scaler, device, cfg)
        val_metrics = run_eval_epoch(model, val_loader, device, cfg)
        elapsed = time.time() - t0

        if logger:
            logger.log(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) loss={train_loss:.4f} "
                       f"val_auc={val_metrics['auc']:.4f} val_eer={val_metrics['eer']:.4f} val_acc={val_metrics['acc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            patience_counter = 0
            if logger:
                os.makedirs(save_dir, exist_ok=True)
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"epoch": epoch, "model_state_dict": state, "auc": best_auc,
                            "eer": val_metrics["eer"], "acc": val_metrics["acc"]},
                           os.path.join(save_dir, "best_model.pth"))
                logger.log(f"  -> Best model saved (val AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience and logger:
                logger.log(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        barrier()
```

- [ ] **Step 3: Replace final evaluation block**

Replace the block from line 153 to line 169 (the final eval section) with:

```python
    if logger:
        logger.log("Loading best model for final test evaluation...")
        ckpt_path = os.path.join(save_dir, "best_model.pth")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        state.load_state_dict(ckpt["model_state_dict"])

        val_final = run_eval_epoch(model, val_loader, device, cfg)
        logger.log(f"Val (best): auc={val_final['auc']:.4f} eer={val_final['eer']:.4f} acc={val_final['acc']:.4f}")

        ffpp_metrics = run_eval_epoch(model, ffpp_eval_loader, device, cfg)
        logger.log(f"FF++ test: auc={ffpp_metrics['auc']:.4f} eer={ffpp_metrics['eer']:.4f} acc={ffpp_metrics['acc']:.4f}")

        cdf_metrics = run_eval_epoch(model, cdf_eval_loader, device, cfg)
        logger.log(f"CDF test: auc={cdf_metrics['auc']:.4f} eer={cdf_metrics['eer']:.4f} acc={cdf_metrics['acc']:.4f}")

        logger.log(f"Best epoch: {best_epoch}, Best val AUC: {best_auc:.4f}")

        end_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = os.path.join(os.path.dirname(logger.log_path), "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["end_time"] = end_ts
        meta["best_epoch"] = best_epoch
        meta["best_val_auc"] = best_auc
        meta["final_val_auc"] = val_final["auc"]
        meta["final_val_eer"] = val_final["eer"]
        meta["final_val_acc"] = val_final["acc"]
        meta["ffpp_test_auc"] = ffpp_metrics["auc"]
        meta["ffpp_test_eer"] = ffpp_metrics["eer"]
        meta["ffpp_test_acc"] = ffpp_metrics["acc"]
        meta["cdf_test_auc"] = cdf_metrics["auc"]
        meta["cdf_test_eer"] = cdf_metrics["eer"]
        meta["cdf_test_acc"] = cdf_metrics["acc"]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.log(f"Meta saved to {meta_path}")

        results_path = os.path.join(save_dir, "results_summary.txt")
        with open(results_path, "w") as f:
            f.write(f"Exp: {exp_name}\n")
            f.write(f"Model: {cfg.get('model', {}).get('name', 'N/A')}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Per-GPU batch: {cfg['train'].get('per_gpu_batch')}\n")
            f.write(f"Val:  AUC={val_final['auc']:.4f} EER={val_final['eer']:.4f} ACC={val_final['acc']:.4f}\n")
            f.write(f"FF++: AUC={ffpp_metrics['auc']:.4f} EER={ffpp_metrics['eer']:.4f} ACC={ffpp_metrics['acc']:.4f}\n")
            f.write(f"CDF:  AUC={cdf_metrics['auc']:.4f} EER={cdf_metrics['eer']:.4f} ACC={cdf_metrics['acc']:.4f}\n")
        logger.log(f"Results saved to {results_path}")
        logger.close()
```

- [ ] **Step 4: Verify train.py syntax**

Run:
```bash
/home/z/anaconda3/envs/deepfake-detection/bin/python -c "import py_compile; py_compile.compile('train.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: use validation set for early stopping, independent FF++/CDF test eval"
```

---

### Task 4: Run exp1 (EfficientNet-B0) with validation set

**Files:**
- Run: `train.py --config configs/exp1_efficientnet.yaml`

- [ ] **Step 1: Clean old outputs and launch exp1**

Run:
```bash
cd /home/z/project/deepfake_detection
rm -f /tmp/exp1_train.log
PYTHONPATH=src /home/z/anaconda3/envs/deepfake-detection/bin/python -u -m torch.distributed.run --nproc_per_node=8 train.py --config configs/exp1_efficientnet.yaml > /tmp/exp1_train.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Verify first epoch completes with val metrics**

Wait until log shows epoch 1 result line. Run:
```bash
until grep -q "val_auc" /tmp/exp1_train.log 2>/dev/null; do sleep 5; done && head -30 /tmp/exp1_train.log | grep -E "(Train samples|Epoch 1)"
```
Expected: `Train samples: 7440, Val samples: 25600, FF++ test: ..., CDF test: ...` and `Epoch 1/30 (...) val_auc=X.XXXX`

- [ ] **Step 3: Wait for training to complete**

Run:
```bash
until grep -q "Results saved" /tmp/exp1_train.log 2>/dev/null; do sleep 15; done && echo "DONE" && tail -15 /tmp/exp1_train.log
```
Expected: Final lines showing val, FF++ test, and CDF test metrics, plus `Results saved`

- [ ] **Step 4: Verify results were saved**

Run:
```bash
cat outputs/exp1_efficientnet/results_summary.txt
```
Expected: File with Val/FF++/CDF metrics

- [ ] **Step 5: Commit results**

```bash
git add outputs/exp1_efficientnet/
git commit -m "feat: exp1 results with validation-based early stopping"
```

---

### Task 5: GPU memory profiling for exp2

**Files:**
- Modify: `configs/exp2_clip_ft.yaml` (temporarily)

- [ ] **Step 1: Create a temporary profiling config**

Run:
```bash
cat > /tmp/exp2_profile.yaml << 'EOF'
_base_: configs/base.yaml
experiment_name: exp2_clip_profile
model:
  name: clip_finetune
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy
train:
  per_gpu_batch: 16
  epochs: 1
EOF
```

- [ ] **Step 2: Run 1 epoch profiling**

Run:
```bash
cd /home/z/project/deepfake_detection
rm -f /tmp/exp2_profile.log
PYTHONPATH=src /home/z/anaconda3/envs/deepfake-detection/bin/python -u -m torch.distributed.run --nproc_per_node=8 train.py --config /tmp/exp2_profile.yaml > /tmp/exp2_profile.log 2>&1 &
PROFILE_PID=$!
echo "Profile PID=$PROFILE_PID"
```

- [ ] **Step 3: Wait for first epoch to start training, then check GPU memory**

Wait until training starts (log shows epoch line), then:
```bash
until grep -q "val_auc" /tmp/exp2_profile.log 2>/dev/null; do sleep 5; done && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```
Expected: Memory usage per GPU (e.g., `0, 12000 MiB, 32768 MiB`)

- [ ] **Step 4: Calculate optimal batch size**

Formula: `optimal_batch = floor((32000 - model_overhead_mb) / per_sample_mb * 0.90)`

If batch=16 uses X MiB per GPU:
- per_sample_mb = (X - model_overhead) / 16
- model_overhead for CLIP ViT-B/16 ≈ 1500 MiB (check actual from nvidia-smi with just model loaded)
- optimal = floor((32000 - model_overhead) / per_sample_mb * 0.90)

- [ ] **Step 5: Update exp2_clip_ft.yaml with optimal batch**

Edit `configs/exp2_clip_ft.yaml` to include the optimal `per_gpu_batch` in the train section:

```yaml
_base_: configs/base.yaml
experiment_name: exp2_clip_ft
model:
  name: clip_finetune
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy
train:
  per_gpu_batch: [CALCULATED_VALUE]
```

- [ ] **Step 6: Commit**

```bash
git add configs/exp2_clip_ft.yaml
git commit -m "feat: optimize exp2 per_gpu_batch for full V100 utilization"
```

---

### Task 6: Run exp2 (CLIP ViT-B/16 fine-tune) with validation set

**Files:**
- Run: `train.py --config configs/exp2_clip_ft.yaml`

- [ ] **Step 1: Launch exp2 training**

Run:
```bash
cd /home/z/project/deepfake_detection
rm -f /tmp/exp2_train.log
PYTHONPATH=src /home/z/anaconda3/envs/deepfake-detection/bin/python -u -m torch.distributed.run --nproc_per_node=8 train.py --config configs/exp2_clip_ft.yaml > /tmp/exp2_train.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Verify GPU utilization is high**

Run:
```bash
until grep -q "val_auc" /tmp/exp2_train.log 2>/dev/null; do sleep 5; done && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```
Expected: Each GPU showing 25-30GB usage (78-94% of 32GB)

- [ ] **Step 3: Monitor first 3 epochs**

Run:
```bash
until [ "$(grep -c 'val_auc' /tmp/exp2_train.log)" -ge 3 ]; do sleep 10; done && grep "Epoch" /tmp/exp2_train.log | head -5
```
Expected: Epoch results showing val_auc improving

- [ ] **Step 4: Wait for training to complete**

Run:
```bash
until grep -q "Results saved" /tmp/exp2_train.log 2>/dev/null; do sleep 15; done && echo "DONE" && tail -15 /tmp/exp2_train.log
```
Expected: Final lines showing val, FF++ test, and CDF test metrics

- [ ] **Step 5: Verify results**

Run:
```bash
cat outputs/exp2_clip_ft/results_summary.txt
```
Expected: File with Val/FF++/CDF metrics

- [ ] **Step 6: Commit results**

```bash
git add outputs/exp2_clip_ft/
git commit -m "feat: exp2 results with validation-based early stopping and optimized GPU"
```

---

## Self-Review

### Spec coverage
- Validation set from videos 201-300: Task 1 (indexing), Task 2 (loader), Task 3 (training loop)
- Val-based early stopping: Task 3 (train.py changes)
- Independent FF++/CDF test: Task 3 (final eval block)
- Exp1 re-run: Task 4
- GPU profiling for exp2: Task 5
- Exp2 with optimized GPU: Task 6
- Results saved: Tasks 4, 6

### Placeholder scan
- Task 5 Step 4 has `[CALCULATED_VALUE]` in Step 5 — this is a runtime value that can only be known after profiling. The formula is provided.
- No TBD, TODO, or vague steps.

### Type consistency
- `index_val_method_frames` returns `list[FrameRecord]` — matches existing `index_train_method_frames` signature
- `build_val_loader` returns `DataLoader` — matches `build_train_loader` and `build_eval_loader`
- `val_metrics` dict has `auc`, `eer`, `acc` keys — matches `run_eval_epoch` return type
- Checkpoint saves `model_state_dict` — matches `evaluate.py` load pattern
