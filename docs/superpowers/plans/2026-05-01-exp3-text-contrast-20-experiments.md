# Exp3 Text Contrast 20 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch 20 full-FF++ experiments that start from the best exp2 checkpoint and compare four text/visual training strategies across five deterministic seeds.

**Architecture:** Add one reusable CLIP image-text model that can run either the current prompt-fusion behavior or image-text contrastive (ITC) training. Add checkpoint initialization from the best exp2 CDF model, configurable visual freezing modes, ITC loss support, and configs/scripts for 4 methods × 5 seeds while preserving the current full-FF++ validation/test pipeline.

**Tech Stack:** PyTorch, OpenAI CLIP ViT-B/16, DistributedDataParallel, YAML configs, existing `train.py` / `run_train_epoch()` / `run_eval_epoch()` pipeline

---

## Experiment Matrix

**Best exp2 starting point:** `outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth` because it has the best completed exp2 full-FF++ CDF AUC: `0.8397`.

**Seeds:** `42`, `123`, `7`, `999`, `2048`.

**Shared full-FF++ settings:**

```yaml
dataset:
  train_videos_per_method: 586
train:
  per_gpu_batch: 128
  lr: 0.00001
  weight_decay: 0.0005
  epochs: 15
  patience: 5
  warmup_epochs: 1
  ema_decay: 0.999
```

**Four methods:**

1. `exp3_itc_freeze_all` — initialize from best exp2, freeze all visual encoder parameters, train classifier + projection layers only, use `cls_loss + lambda_itc * image_text_contrastive_loss`, evaluate with classifier only.
2. `exp3_itc_freeze_partial` — initialize from best exp2, freeze early visual transformer blocks and train high visual blocks + classifier, use ITC, evaluate with classifier only.
3. `exp3_itc_train_all` — initialize from best exp2, train all visual encoder parameters + classifier, use ITC, evaluate with classifier only.
4. `exp3_prompt_init` — initialize from best exp2, keep the current prompt auxiliary CE and prompt-fusion inference as the baseline control.

### Task 1: Add a model that supports prompt fusion and ITC features

**Files:**
- Modify: `src/deepfake_detection/models/clip_prompt.py:1-55`
- Modify: `src/deepfake_detection/models/factory.py:1-22`
- Test: `src/deepfake_detection/models/clip_classifier.py:7-16`

- [ ] **Step 1: Replace `CLIPPromptBinaryClassifier` with a mode-capable implementation**

Use this full file content for `src/deepfake_detection/models/clip_prompt.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def build_fixed_prompt_texts():
    return REAL_PROMPTS, FAKE_PROMPTS


class CLIPPromptBinaryClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        tau: float = 0.07,
        freeze_visual: str = "none",
        freeze_visual_layers: int = 9,
        projection_dim: int = 512,
    ):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        self.visual = clip_model.visual
        self.classifier = nn.Linear(self.visual.output_dim, 2)
        self.image_projection = nn.Linear(self.visual.output_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(self.visual.output_dim, projection_dim, bias=False)
        self.register_buffer("tau", torch.tensor(tau))

        real_texts, fake_texts = build_fixed_prompt_texts()
        real_tokens = clip.tokenize(real_texts)
        fake_tokens = clip.tokenize(fake_texts)
        with torch.no_grad():
            real_features = clip_model.encode_text(real_tokens)
            fake_features = clip_model.encode_text(fake_tokens)
        self.register_buffer("_real_features", F.normalize(real_features, dim=-1))
        self.register_buffer("_fake_features", F.normalize(fake_features, dim=-1))

        self.apply_visual_freeze(freeze_visual, freeze_visual_layers)

    def apply_visual_freeze(self, mode: str, freeze_visual_layers: int):
        if mode == "none":
            return
        if mode == "all":
            for p in self.visual.parameters():
                p.requires_grad = False
            return
        if mode == "partial":
            for p in self.visual.parameters():
                p.requires_grad = False
            for p in self.visual.ln_post.parameters():
                p.requires_grad = True
            if self.visual.proj is not None:
                self.visual.proj.requires_grad = True
            blocks = self.visual.transformer.resblocks
            for block in blocks[freeze_visual_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            return
        raise ValueError(f"Unknown freeze_visual mode: {mode}")

    def encode_image_features(self, images):
        image_features = self.visual(images)
        return F.normalize(image_features, dim=-1)

    def prompt_logits_from_features(self, image_features):
        real_sim = (image_features @ self._real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ self._fake_features.T).mean(dim=1, keepdim=True)
        return torch.cat([real_sim, fake_sim], dim=1) / self.tau

    def forward(self, images):
        image_features = self.encode_image_features(images)
        cls_logits = self.classifier(image_features.float())
        prompt_logits = self.prompt_logits_from_features(image_features)
        return cls_logits, prompt_logits

    def forward_with_features(self, images):
        image_features = self.encode_image_features(images)
        cls_logits = self.classifier(image_features.float())
        prompt_logits = self.prompt_logits_from_features(image_features)
        text_features = torch.stack([
            self._real_features.mean(dim=0),
            self._fake_features.mean(dim=0),
        ], dim=0)
        image_features = self.image_projection(image_features.float())
        text_features = self.text_projection(text_features.float())
        return cls_logits, prompt_logits, image_features, text_features
```

- [ ] **Step 2: Update the model factory**

Change `src/deepfake_detection/models/factory.py:16-18` to:

```python
    if name == "clip_prompt":
        tau = model_cfg.get("tau", 0.07)
        freeze_visual = model_cfg.get("freeze_visual", "none")
        freeze_visual_layers = model_cfg.get("freeze_visual_layers", 9)
        return CLIPPromptBinaryClassifier(clip_name, tau, freeze_visual, freeze_visual_layers, model_cfg.get("projection_dim", 512))
```

- [ ] **Step 3: Run import smoke test**

Run:

```bash
PYTHONPATH=src python - <<'PY'
from deepfake_detection.models.factory import build_model
for mode in ['none', 'all', 'partial']:
    model = build_model({'name': 'clip_prompt', 'clip_model_name': 'ViT-B/16', 'freeze_visual': mode})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(mode, trainable > 0)
PY
```

Expected:

```text
none True
all True
partial True
```

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/models/clip_prompt.py src/deepfake_detection/models/factory.py
git commit -m "feat: add configurable prompt contrast model"
```

### Task 2: Add image-text contrastive loss and checkpoint initialization

**Files:**
- Modify: `src/deepfake_detection/losses/contrastive.py:1-58`
- Modify: `src/deepfake_detection/engine/trainers.py:1-135`
- Modify: `train.py:118-170`
- Test: `outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth`

- [ ] **Step 1: Add image-text InfoNCE loss**

Append this function to `src/deepfake_detection/losses/contrastive.py`:

```python

def image_text_contrastive_loss(image_features, text_features, labels, temperature=0.07):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits = image_features @ text_features.t() / temperature
    return F.cross_entropy(logits.float(), labels.long())
```

- [ ] **Step 2: Add ITC training branch**

Add this function after `prompt_contrast_step()` in `src/deepfake_detection/engine/trainers.py`:

```python

def prompt_itc_step(model, batch, device, lambda_itc=0.1, temperature=0.07):
    from deepfake_detection.losses.contrastive import image_text_contrastive_loss
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    cls_logits, _, image_features, text_features = model.forward_with_features(images)
    cls_loss = F.cross_entropy(cls_logits, labels)
    itc_loss = image_text_contrastive_loss(image_features, text_features, labels, temperature)
    total_loss = cls_loss + lambda_itc * itc_loss
    return total_loss, cls_logits
```

- [ ] **Step 3: Wire ITC into `run_train_epoch()`**

Change `src/deepfake_detection/engine/trainers.py:65-80` to:

```python
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    is_prompt_itc = loss_name == "cross_entropy_plus_prompt_itc"
    is_bgface = loss_name == "cross_entropy_plus_bgface_contrast"
    lambda_align = cfg.get("loss", {}).get("lambda_align", 0.1)
    lambda_itc = cfg.get("loss", {}).get("lambda_itc", 0.1)
    temperature = cfg.get("loss", {}).get("temperature", 0.07)
    beta = cfg.get("loss", {}).get("beta", 0.1)
    for batch in dataloader:
        optimizer.zero_grad()
        with autocast(enabled=True):
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            elif is_prompt_itc:
                loss, _ = prompt_itc_step(model, batch, device, lambda_itc, temperature)
            elif is_prompt:
                loss, _, _ = prompt_contrast_step(model, batch, device, beta)
            elif is_bgface:
                loss, _ = bgface_contrast_step(model, batch, device, lambda_align, temperature)
            else:
                loss, _ = classification_step(model, batch, device)
```

- [ ] **Step 4: Make ITC evaluation classifier-only**

Change `src/deepfake_detection/engine/trainers.py:99-114` to:

```python
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    is_prompt_itc = loss_name == "cross_entropy_plus_prompt_itc"
    alpha = cfg.get("loss", {}).get("alpha", 0.3)
    for batch in dataloader:
        images = batch["image"].to(device) if not is_contrastive else batch["background"].to(device)
        labels = batch["label"]
        video_ids = batch["video_id"]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            output = model(images.float())
        if is_prompt and not is_prompt_itc:
            cls_logits, prompt_logits = output
            cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]
            prompt_prob = torch.softmax(prompt_logits, dim=1)[:, 1]
            probs = ((1 - alpha) * cls_prob + alpha * prompt_prob).cpu().numpy()
        else:
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
```

- [ ] **Step 5: Add checkpoint initialization to `train.py`**

Insert after `model = build_model(cfg["model"]).to(device)` at `train.py:162`:

```python
    init_ckpt_path = cfg.get("train", {}).get("init_checkpoint")
    if init_ckpt_path:
        ckpt = torch.load(init_ckpt_path, map_location=device)
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main_process():
            print(f"Initialized from {init_ckpt_path}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
```

- [ ] **Step 6: Run ITC loss smoke test**

Run:

```bash
PYTHONPATH=src python - <<'PY'
import torch
from deepfake_detection.losses.contrastive import image_text_contrastive_loss
image = torch.randn(4, 512)
text = torch.randn(2, 512)
labels = torch.tensor([0, 1, 0, 1])
loss = image_text_contrastive_loss(image, text, labels)
print(loss.ndim, torch.isfinite(loss).item())
PY
```

Expected:

```text
0 True
```

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/losses/contrastive.py src/deepfake_detection/engine/trainers.py train.py
git commit -m "feat: add image-text contrastive training"
```

### Task 3: Create the 20 experiment configs

**Files:**
- Create: `configs/exp3_itc_freeze_all_s42.yaml`
- Create: `configs/exp3_itc_freeze_all_s123.yaml`
- Create: `configs/exp3_itc_freeze_all_s7.yaml`
- Create: `configs/exp3_itc_freeze_all_s999.yaml`
- Create: `configs/exp3_itc_freeze_all_s2048.yaml`
- Create: `configs/exp3_itc_freeze_partial_s42.yaml`
- Create: `configs/exp3_itc_freeze_partial_s123.yaml`
- Create: `configs/exp3_itc_freeze_partial_s7.yaml`
- Create: `configs/exp3_itc_freeze_partial_s999.yaml`
- Create: `configs/exp3_itc_freeze_partial_s2048.yaml`
- Create: `configs/exp3_itc_train_all_s42.yaml`
- Create: `configs/exp3_itc_train_all_s123.yaml`
- Create: `configs/exp3_itc_train_all_s7.yaml`
- Create: `configs/exp3_itc_train_all_s999.yaml`
- Create: `configs/exp3_itc_train_all_s2048.yaml`
- Create: `configs/exp3_prompt_init_s42.yaml`
- Create: `configs/exp3_prompt_init_s123.yaml`
- Create: `configs/exp3_prompt_init_s7.yaml`
- Create: `configs/exp3_prompt_init_s999.yaml`
- Create: `configs/exp3_prompt_init_s2048.yaml`
- Test: `configs/exp3_fullffpp_ema_s42.yaml`

- [ ] **Step 1: Use this shared checkpoint path in every config**

```yaml
  init_checkpoint: outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
```

- [ ] **Step 2: Template for `exp3_itc_freeze_all`**

```yaml
_base_: configs/base.yaml
experiment_name: exp3_itc_freeze_all_s42
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
  freeze_visual: all
  freeze_visual_layers: 9
  tau: 0.07
loss:
  name: cross_entropy_plus_prompt_itc
  lambda_itc: 0.1
  temperature: 0.07
train:
  init_checkpoint: outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
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

- [ ] **Step 3: Template for `exp3_itc_freeze_partial`**

```yaml
_base_: configs/base.yaml
experiment_name: exp3_itc_freeze_partial_s42
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
  freeze_visual: partial
  freeze_visual_layers: 9
  tau: 0.07
loss:
  name: cross_entropy_plus_prompt_itc
  lambda_itc: 0.1
  temperature: 0.07
train:
  init_checkpoint: outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
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

- [ ] **Step 4: Template for `exp3_itc_train_all`**

```yaml
_base_: configs/base.yaml
experiment_name: exp3_itc_train_all_s42
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
  freeze_visual: none
  freeze_visual_layers: 9
  tau: 0.07
loss:
  name: cross_entropy_plus_prompt_itc
  lambda_itc: 0.1
  temperature: 0.07
train:
  init_checkpoint: outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
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

- [ ] **Step 5: Template for `exp3_prompt_init` control**

```yaml
_base_: configs/base.yaml
experiment_name: exp3_prompt_init_s42
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
  freeze_visual: none
  freeze_visual_layers: 9
  tau: 0.07
loss:
  name: cross_entropy_plus_prompt
  beta: 0.1
  alpha: 0.3
  tau: 0.07
train:
  init_checkpoint: outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
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

- [ ] **Step 6: Duplicate templates for seeds 123, 7, 999, and 2048**

For every copied file, update both:

```text
experiment_name: <method>_s<seed>
seed: <seed>
```

- [ ] **Step 7: Verify all configs load**

Run:

```bash
python - <<'PY'
import glob, yaml
files = sorted(glob.glob('configs/exp3_itc_*_s*.yaml') + glob.glob('configs/exp3_prompt_init_s*.yaml'))
for f in files:
    yaml.safe_load(open(f))
print(len(files))
PY
```

Expected:

```text
20
```

- [ ] **Step 8: Commit**

```bash
git add configs/exp3_itc_*_s*.yaml configs/exp3_prompt_init_s*.yaml
git commit -m "feat: add exp3 text contrast experiment configs"
```

### Task 4: Add the sequential 20-run launcher

**Files:**
- Create: `scripts/run_exp3_text_contrast_20runs.sh`
- Test: `scripts/run_fullffpp_ema_10runs.sh`

- [ ] **Step 1: Write the launcher**

Use this full file content:

```bash
#!/bin/bash
set -e
export PYTHONPATH=src

START=${1:-1}

EXPERIMENTS=(
  "configs/exp3_itc_freeze_all_s42.yaml"
  "configs/exp3_itc_freeze_all_s123.yaml"
  "configs/exp3_itc_freeze_all_s7.yaml"
  "configs/exp3_itc_freeze_all_s999.yaml"
  "configs/exp3_itc_freeze_all_s2048.yaml"
  "configs/exp3_itc_freeze_partial_s42.yaml"
  "configs/exp3_itc_freeze_partial_s123.yaml"
  "configs/exp3_itc_freeze_partial_s7.yaml"
  "configs/exp3_itc_freeze_partial_s999.yaml"
  "configs/exp3_itc_freeze_partial_s2048.yaml"
  "configs/exp3_itc_train_all_s42.yaml"
  "configs/exp3_itc_train_all_s123.yaml"
  "configs/exp3_itc_train_all_s7.yaml"
  "configs/exp3_itc_train_all_s999.yaml"
  "configs/exp3_itc_train_all_s2048.yaml"
  "configs/exp3_prompt_init_s42.yaml"
  "configs/exp3_prompt_init_s123.yaml"
  "configs/exp3_prompt_init_s7.yaml"
  "configs/exp3_prompt_init_s999.yaml"
  "configs/exp3_prompt_init_s2048.yaml"
)

for i in "${!EXPERIMENTS[@]}"; do
  run_no=$((i+1))
  if [ "$run_no" -lt "$START" ]; then
    continue
  fi
  config="${EXPERIMENTS[$i]}"
  echo "============================================"
  echo "[$run_no/${#EXPERIMENTS[@]}] Running $config"
  echo "============================================"
  torchrun --nproc_per_node=8 train.py --config "$config"
  echo ""
done

echo "All 20 exp3 text contrast runs complete!"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/run_exp3_text_contrast_20runs.sh
```

Expected: command succeeds silently.

- [ ] **Step 3: Verify the run list count**

Run:

```bash
python - <<'PY'
from pathlib import Path
script = Path('scripts/run_exp3_text_contrast_20runs.sh').read_text()
print(script.count('configs/exp3_'))
PY
```

Expected:

```text
20
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_exp3_text_contrast_20runs.sh
git commit -m "chore: add exp3 text contrast batch launcher"
```

### Task 5: Run a one-experiment smoke test before the 20-run batch

**Files:**
- Test: `configs/exp3_itc_freeze_partial_s42.yaml`
- Test: `train.py`
- Test: `outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth`

- [ ] **Step 1: Start the freeze-partial smoke run**

Run:

```bash
torchrun --nproc_per_node=8 train.py --config configs/exp3_itc_freeze_partial_s42.yaml
```

Expected:

```text
Initialized from outputs/exp2_fullffpp_ema_s2048/20260501_163927/best_model.pth
Missing keys: 2, Unexpected keys: 0
Epoch 1/15 ... loss=...
```

- [ ] **Step 2: Confirm it writes results**

Run:

```bash
find outputs/exp3_itc_freeze_partial_s42 -name results_summary.txt | sort | tail -1
```

Expected: one path ending in `results_summary.txt`.

- [ ] **Step 3: Inspect the smoke result**

Run:

```bash
latest=$(find outputs/exp3_itc_freeze_partial_s42 -name results_summary.txt | sort | tail -1)
cat "$latest"
```

Expected: contains `Val:`, `FF++:`, and `CDF:` lines.

### Task 6: Launch the 20-run experiment batch

**Files:**
- Test: `scripts/run_exp3_text_contrast_20runs.sh`

- [ ] **Step 1: Start the 20-run batch**

Run:

```bash
bash scripts/run_exp3_text_contrast_20runs.sh
```

Expected: the script starts at `[1/20] Running configs/exp3_itc_freeze_all_s42.yaml`.

- [ ] **Step 2: Resume if a late run fails**

If run N fails after earlier runs completed, resume with:

```bash
bash scripts/run_exp3_text_contrast_20runs.sh N
```

Expected: the script skips runs before N and starts at run N.

### Self-Review

Spec coverage:
- Best exp2 CDF checkpoint as initialization: covered in Tasks 2 and 3.
- Four methods: covered in Task 3 configs and Task 4 launcher.
- Five seeds: covered in Task 3 configs.
- Full FF++ data with validation preserved: covered by `train_videos_per_method: 586` in every config.
- Current prompt-fusion baseline: covered by `exp3_prompt_init_*` configs using `cross_entropy_plus_prompt`.
- ITC classifier-only evaluation: covered in Task 2 Step 4.

Placeholder scan:
- No `TBD`, `TODO`, `implement later`, or missing code blocks.

Type consistency:
- The new loss name is consistently `cross_entropy_plus_prompt_itc`.
- The new model method is consistently `forward_with_features()`.
- The checkpoint config key is consistently `train.init_checkpoint`.
