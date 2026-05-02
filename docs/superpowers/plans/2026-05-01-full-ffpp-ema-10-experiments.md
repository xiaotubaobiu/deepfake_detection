# Full-FF++ EMA 10 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch 10 new 8-GPU experiments for exp2 and exp3 across seeds 42, 123, 7, 999, and 2048 using the full usable FF++ training split while keeping the existing validation split intact.

**Architecture:** Reuse the deterministic exp2/exp3 config pattern, but raise `train_videos_per_method` from 200 to the maximum value that still leaves the current 100-video validation window available for every method. Keep the current 8-GPU training stack, EMA, 15 epochs, warmup, and evaluation flow unchanged, then launch the 10 configs sequentially from one shell script.

**Tech Stack:** PyTorch DDP (`torchrun`), YAML configs, existing `train.py` training pipeline, FF++ JSON indexing

---

### Task 1: Determine the full usable FF++ training size

**Files:**
- Modify: `configs/base.yaml:1-37`
- Test: `src/deepfake_detection/data/index_ffpp.py:25-73`

- [ ] **Step 1: Verify available FF++ train videos per method**

```bash
python - <<'PY'
import json, pathlib
root=pathlib.Path('/Dataset/deepfake_detection/DF40_all/dataset_json')
methods=['simswap','inswap','blendface','faceswap','fsgan','mobileswap','e4s','facedancer','fomm','facevid2vid','wav2lip','sadtalker','MRAA','pirender','tpsm','lia']
counts=[]
for method in methods:
    data=json.load(open(root/f'{method}_ff.json'))
    counts.append(len(data[f'{method}_ff'][f'{method}_Fake']['train']))
print(min(counts), max(counts))
PY
```

- [ ] **Step 2: Confirm the target usable count**

```text
Use train_videos_per_method = 586.
Reason: the smallest method has 686 train videos total, and the current validation logic consumes videos [max_videos : max_videos + 100), so 586 leaves exactly 100 validation videos for every method.
```

- [ ] **Step 3: Keep the shared base config unchanged**

```yaml
# Do not change configs/base.yaml.
# Set train_videos_per_method per experiment config instead.
```

- [ ] **Step 4: Verify no code change is required**

Run: `python - <<'PY'
from deepfake_detection.data.index_ffpp import load_json_index
print('ok')
PY`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add configs
 git commit -m "feat: add full-ffpp experiment configs"
```

### Task 2: Create 10 full-FF++ EMA configs

**Files:**
- Create: `configs/exp2_fullffpp_ema_s42.yaml`
- Create: `configs/exp2_fullffpp_ema_s123.yaml`
- Create: `configs/exp2_fullffpp_ema_s7.yaml`
- Create: `configs/exp2_fullffpp_ema_s999.yaml`
- Create: `configs/exp2_fullffpp_ema_s2048.yaml`
- Create: `configs/exp3_fullffpp_ema_s42.yaml`
- Create: `configs/exp3_fullffpp_ema_s123.yaml`
- Create: `configs/exp3_fullffpp_ema_s7.yaml`
- Create: `configs/exp3_fullffpp_ema_s999.yaml`
- Create: `configs/exp3_fullffpp_ema_s2048.yaml`
- Test: `configs/exp2_det_s42.yaml`
- Test: `configs/exp3_det_s42.yaml`

- [ ] **Step 1: Use this template for exp2 full-data configs**

```yaml
_base_: configs/base.yaml
experiment_name: exp2_fullffpp_ema_s42
model:
  name: clip_finetune
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy
train:
  per_gpu_batch: 128
  lr: 0.00001
  weight_decay: 0.0005
  epochs: 15
  patience: 5
  warmup_epochs: 1
  ema_decay: 0.999
  seed: 42
dataset:
  train_videos_per_method: 586
```

- [ ] **Step 2: Use this template for exp3 full-data configs**

```yaml
_base_: configs/base.yaml
experiment_name: exp3_fullffpp_ema_s42
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy_plus_prompt
  beta: 0.1
  alpha: 0.3
  tau: 0.07
train:
  per_gpu_batch: 128
  lr: 0.00001
  weight_decay: 0.0005
  epochs: 15
  patience: 5
  warmup_epochs: 1
  ema_decay: 0.999
  seed: 42
dataset:
  train_videos_per_method: 586
```

- [ ] **Step 3: Duplicate per seed with exact experiment names**

```text
Seeds: 42, 123, 7, 999, 2048
Experiment names:
- exp2_fullffpp_ema_s42
- exp2_fullffpp_ema_s123
- exp2_fullffpp_ema_s7
- exp2_fullffpp_ema_s999
- exp2_fullffpp_ema_s2048
- exp3_fullffpp_ema_s42
- exp3_fullffpp_ema_s123
- exp3_fullffpp_ema_s7
- exp3_fullffpp_ema_s999
- exp3_fullffpp_ema_s2048
```

- [ ] **Step 4: Verify YAMLs load**

Run: `python - <<'PY'
import yaml
files=[
'configs/exp2_fullffpp_ema_s42.yaml','configs/exp2_fullffpp_ema_s123.yaml','configs/exp2_fullffpp_ema_s7.yaml','configs/exp2_fullffpp_ema_s999.yaml','configs/exp2_fullffpp_ema_s2048.yaml',
'configs/exp3_fullffpp_ema_s42.yaml','configs/exp3_fullffpp_ema_s123.yaml','configs/exp3_fullffpp_ema_s7.yaml','configs/exp3_fullffpp_ema_s999.yaml','configs/exp3_fullffpp_ema_s2048.yaml']
for f in files:
    yaml.safe_load(open(f))
print('loaded', len(files))
PY`
Expected: `loaded 10`

- [ ] **Step 5: Commit**

```bash
git add configs/exp2_fullffpp_ema_s*.yaml configs/exp3_fullffpp_ema_s*.yaml
git commit -m "feat: add full-ffpp ema experiment configs"
```

### Task 3: Create the sequential launcher and start the batch

**Files:**
- Create: `scripts/run_fullffpp_ema_10runs.sh`
- Test: `train.py:118-290`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
set -e
export PYTHONPATH=src

EXPERIMENTS=(
  "configs/exp2_fullffpp_ema_s42.yaml"
  "configs/exp2_fullffpp_ema_s123.yaml"
  "configs/exp2_fullffpp_ema_s7.yaml"
  "configs/exp2_fullffpp_ema_s999.yaml"
  "configs/exp2_fullffpp_ema_s2048.yaml"
  "configs/exp3_fullffpp_ema_s42.yaml"
  "configs/exp3_fullffpp_ema_s123.yaml"
  "configs/exp3_fullffpp_ema_s7.yaml"
  "configs/exp3_fullffpp_ema_s999.yaml"
  "configs/exp3_fullffpp_ema_s2048.yaml"
)

for i in "${!EXPERIMENTS[@]}"; do
  config="${EXPERIMENTS[$i]}"
  echo "============================================"
  echo "[$((i+1))/${#EXPERIMENTS[@]}] Running $config"
  echo "============================================"
  torchrun --nproc_per_node=8 train.py --config "$config"
  echo ""
done

echo "All 10 full-FF++ EMA runs complete!"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/run_fullffpp_ema_10runs.sh`
Expected: command succeeds silently

- [ ] **Step 3: Smoke-check the first config path**

Run: `python - <<'PY'
import os
print(os.path.exists('configs/exp2_fullffpp_ema_s42.yaml'))
print(os.path.exists('scripts/run_fullffpp_ema_10runs.sh'))
PY`
Expected:
```text
True
True
```

- [ ] **Step 4: Start the experiment batch**

Run: `bash scripts/run_fullffpp_ema_10runs.sh`
Expected: the first banner prints, then `train.py` logs the experiment name, batch size, epochs, LR, warmup, EMA, and dataset sizes.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_fullffpp_ema_10runs.sh
git commit -m "chore: add full-ffpp ema batch launcher"
```
