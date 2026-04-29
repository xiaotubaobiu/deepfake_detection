# Exp3: CLIP + Prompt Contrast 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 exp3 双路径模型（分类头 + prompt 对比学习），训练时两组 logits 各自计算 CE loss，推理时加权概率融合。

**Architecture:** 修改现有 `CLIPPromptBinaryClassifier`，加入线性分类头，冻结 text encoder 并预计算 prompt 特征为 buffer。训练损失 `L = L_cls + β * L_prompt`，推理融合 `final = (1-α)*cls_prob + α*prompt_prob`。

**Tech Stack:** PyTorch, CLIP (ViT-B/16), DDP, AMP, Albumentations

---

### Task 1: 更新 Prompt 文本

**Files:**
- Modify: `src/deepfake_detection/data/constants.py:14-15`
- Modify: `tests/models/test_prompts.py`

- [ ] **Step 1: 更新 constants.py 中的 prompt 文本**

将 `REAL_PROMPTS` 和 `FAKE_PROMPTS` 从各 2 条扩展为各 3 条：

```python
REAL_PROMPTS = [
    "a real human face photo",
    "an authentic face image",
    "a natural face without manipulation",
]
FAKE_PROMPTS = [
    "a fake face photo",
    "a manipulated face image",
    "a deepfake face image",
]
```

- [ ] **Step 2: 更新 test_prompts.py 验证 prompt 数量**

```python
from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def test_build_fixed_prompt_texts_returns_real_and_fake_prompts():
    from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts
    real, fake = build_fixed_prompt_texts()
    assert real == REAL_PROMPTS
    assert fake == FAKE_PROMPTS


def test_prompt_counts():
    assert len(REAL_PROMPTS) == 3
    assert len(FAKE_PROMPTS) == 3
```

- [ ] **Step 3: 运行测试验证**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/models/test_prompts.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/data/constants.py tests/models/test_prompts.py
git commit -m "feat(exp3): update prompt texts to 3 per class"
```

---

### Task 2: 重写 CLIPPromptBinaryClassifier 模型

**Files:**
- Modify: `src/deepfake_detection/models/clip_prompt.py`
- Modify: `tests/models/test_prompts.py`（新增模型测试）

- [ ] **Step 1: 写模型前向传播的失败测试**

在 `tests/models/test_prompts.py` 末尾追加：

```python
import torch


def test_clip_prompt_forward_returns_tuple_of_logits():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    images = torch.randn(2, 3, 224, 224)
    cls_logits, prompt_logits = model(images)
    assert cls_logits.shape == (2, 2)
    assert prompt_logits.shape == (2, 2)


def test_clip_prompt_text_encoder_is_frozen():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    assert not hasattr(model, "clip_model"), "clip_model should not be kept"
    for name, buf in model.named_buffers():
        if "real_features" in name or "fake_features" in name:
            assert not buf.requires_grad


def test_clip_prompt_has_classifier_head():
    from deepfake_detection.models.clip_prompt import CLIPPromptBinaryClassifier
    model = CLIPPromptBinaryClassifier()
    assert hasattr(model, "classifier")
    assert model.classifier.out_features == 2
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/models/test_prompts.py::test_clip_prompt_forward_returns_tuple_of_logits -v`
Expected: FAIL（当前 forward 返回单个 tensor，不是 tuple）

- [ ] **Step 3: 重写 clip_prompt.py**

完整替换 `src/deepfake_detection/models/clip_prompt.py`：

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
    def __init__(self, clip_model_name: str = "ViT-B/16", tau: float = 0.07):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        # 只保留 visual encoder（可训练）
        self.visual = clip_model.visual

        # 预计算 prompt 特征（冻结 text encoder，用完即弃）
        real_texts, fake_texts = build_fixed_prompt_texts()
        real_tokens = clip.tokenize(real_texts)
        fake_tokens = clip.tokenize(fake_texts)
        with torch.no_grad():
            real_features = clip_model.encode_text(real_tokens)
            fake_features = clip_model.encode_text(fake_tokens)
        real_features = F.normalize(real_features, dim=-1)
        fake_features = F.normalize(fake_features, dim=-1)
        self.register_buffer("_real_features", real_features)
        self.register_buffer("_fake_features", fake_features)
        # text encoder 随 clip_model 一起释放，不保留引用

        # 分类头
        self.classifier = nn.Linear(self.visual.output_dim, 2)

        # 温度参数
        self.register_buffer("tau", torch.tensor(tau))

    def forward(self, images):
        # 1. 视觉特征
        image_features = self.visual(images)
        image_features = F.normalize(image_features, dim=-1)

        # 2. 分类头 logits
        cls_logits = self.classifier(image_features.float())

        # 3. Prompt logits（温度缩放）
        real_sim = (image_features @ self._real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (image_features @ self._fake_features.T).mean(dim=1, keepdim=True)
        prompt_logits = torch.cat([real_sim, fake_sim], dim=1) / self.tau

        return cls_logits, prompt_logits
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/models/test_prompts.py -v`
Expected: 所有测试 PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/models/clip_prompt.py tests/models/test_prompts.py
git commit -m "feat(exp3): rewrite clip_prompt with dual-path classifier + prompt logits"
```

---

### Task 3: 更新训练器 — prompt_contrast_step + eval 融合

**Files:**
- Modify: `src/deepfake_detection/engine/trainers.py`
- Modify: `tests/engine/test_metrics.py`（新增 trainer 测试）

- [ ] **Step 1: 写 prompt_contrast_step 失败测试**

在 `tests/engine/test_metrics.py` 末尾追加：

```python
import torch
import torch.nn as nn


class _FakePromptModel(nn.Module):
    """用于测试 prompt_contrast_step 的 mock 模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, images):
        cls_logits = self.linear(images.mean(dim=[1, 2, 3]).unsqueeze(1).expand(-1, 10))
        prompt_logits = cls_logits * 2.0
        return cls_logits, prompt_logits


def test_prompt_contrast_step_returns_total_loss_and_logits():
    from deepfake_detection.engine.trainers import prompt_contrast_step
    model = _FakePromptModel()
    batch = {
        "image": torch.randn(4, 3, 32, 32),
        "label": torch.tensor([0, 1, 0, 1]),
    }
    total_loss, cls_logits, prompt_logits = prompt_contrast_step(model, batch, "cpu", beta=0.1)
    assert total_loss.ndim == 0
    assert cls_logits.shape == (4, 2)
    assert prompt_logits.shape == (4, 2)
    assert total_loss.item() > 0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/engine/test_metrics.py::test_prompt_contrast_step_returns_total_loss_and_logits -v`
Expected: FAIL（`prompt_contrast_step` 函数不存在）

- [ ] **Step 3: 实现 prompt_contrast_step 函数**

在 `src/deepfake_detection/engine/trainers.py` 的 `contrastive_step` 函数之后（第 31 行后）添加：

```python
def prompt_contrast_step(model, batch, device, beta=0.1):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    cls_logits, prompt_logits = model(images)
    cls_loss = F.cross_entropy(cls_logits, labels)
    prompt_loss = F.cross_entropy(prompt_logits, labels)
    total_loss = cls_loss + beta * prompt_loss
    return total_loss, cls_logits, prompt_logits
```

- [ ] **Step 4: 修改 run_train_epoch 增加 prompt 分支**

在 `run_train_epoch` 函数中，在 `is_contrastive` 行之后添加 `is_prompt` 判断，并在循环内添加分支。

将第 33-54 行的 `run_train_epoch` 替换为：

```python
def run_train_epoch(model, dataloader, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0
    n = 0
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    lambda_align = cfg.get("loss", {}).get("lambda_align", 0.1)
    temperature = cfg.get("loss", {}).get("temperature", 0.07)
    beta = cfg.get("loss", {}).get("beta", 0.1)
    for batch in dataloader:
        optimizer.zero_grad()
        with autocast(enabled=True):
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            elif is_prompt:
                loss, _, _ = prompt_contrast_step(model, batch, device, beta)
            else:
                loss, _ = classification_step(model, batch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)
```

- [ ] **Step 5: 修改 run_eval_epoch 增加 prompt 融合**

将第 57-81 行的 `run_eval_epoch` 替换为：

```python
@torch.no_grad()
def run_eval_epoch(model, dataloader, device, cfg):
    model.eval()
    all_rows = []
    loss_name = cfg.get("loss", {}).get("name", "")
    is_contrastive = loss_name == "cross_entropy_plus_contrastive"
    is_prompt = loss_name == "cross_entropy_plus_prompt"
    alpha = cfg.get("loss", {}).get("alpha", 0.3)
    for batch in dataloader:
        images = batch["image"].to(device) if not is_contrastive else batch["background"].to(device)
        labels = batch["label"]
        video_ids = batch["video_id"]
        with autocast(enabled=True):
            output = model(images)
        if is_prompt:
            cls_logits, prompt_logits = output
            cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]
            prompt_prob = torch.softmax(prompt_logits, dim=1)[:, 1]
            probs = ((1 - alpha) * cls_prob + alpha * prompt_prob).cpu().numpy()
        else:
            logits = output
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for i in range(len(probs)):
            all_rows.append({
                "video_id": video_ids[i] if isinstance(video_ids, list) else video_ids[i],
                "score": float(probs[i]),
                "label": int(labels[i]),
            })
    labels, scores = aggregate_video_predictions(all_rows)
    if len(set(labels)) < 2:
        return {"auc": 0.0, "eer": 0.0, "acc": 0.0, "loss": 0.0}
    auc = compute_auc(labels, scores)
    eer = compute_eer(labels, scores)
    acc = compute_acc(labels, scores)
    return {"auc": auc, "eer": eer, "acc": acc}
```

- [ ] **Step 6: 运行测试验证通过**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/engine/test_metrics.py -v`
Expected: 所有测试 PASS

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/engine/trainers.py tests/engine/test_metrics.py
git commit -m "feat(exp3): add prompt_contrast_step and eval score fusion to trainer"
```

---

### Task 4: 更新配置文件

**Files:**
- Modify: `configs/exp3_clip_prompt.yaml`
- Modify: `tests/configs/test_config_load.py`

- [ ] **Step 1: 写配置加载失败测试**

在 `tests/configs/test_config_load.py` 末尾追加（如果文件已有 exp3 测试则更新）：

```python
def test_exp3_config_loads_prompt_loss():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from train import load_config
    cfg = load_config("configs/exp3_clip_prompt.yaml")
    assert cfg["loss"]["name"] == "cross_entropy_plus_prompt"
    assert cfg["loss"]["beta"] == 0.1
    assert cfg["loss"]["alpha"] == 0.3
    assert cfg["loss"]["tau"] == 0.07
    assert cfg["train"]["per_gpu_batch"] == 128
    assert cfg["train"]["lr"] == 0.00002
    assert cfg["train"]["weight_decay"] == 0.0005
    assert cfg["train"]["epochs"] == 5
    assert cfg["train"]["patience"] == 3
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/configs/test_config_load.py::test_exp3_config_loads_prompt_loss -v`
Expected: FAIL（当前 config 缺少 loss.beta/alpha/tau 和 train 参数）

- [ ] **Step 3: 更新 exp3 配置**

完整替换 `configs/exp3_clip_prompt.yaml`：

```yaml
_base_: configs/base.yaml
experiment_name: exp3_clip_prompt
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
  lr: 0.00002
  weight_decay: 0.0005
  epochs: 5
  patience: 3
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd /home/z/project/deepfake_detection && PYTHONPATH=src pytest tests/configs/test_config_load.py::test_exp3_config_loads_prompt_loss -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/exp3_clip_prompt.yaml tests/configs/test_config_load.py
git commit -m "feat(exp3): update config with prompt loss params and exp2-aligned hyperparams"
```

---

### Task 5: 运行 Exp3 训练

**Files:**
- None（运行训练，验证结果）

- [ ] **Step 1: 在 GPU 上启动训练**

Run:
```bash
cd /home/z/project/deepfake_detection && \
PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29501 \
    train.py --config configs/exp3_clip_prompt.yaml
```

Expected: 训练启动，日志显示 `loss.name: cross_entropy_plus_prompt, beta: 0.1, alpha: 0.3, tau: 0.07`，per_gpu_batch=128，lr=2e-5。

- [ ] **Step 2: 监控训练进度**

检查日志中 epoch 级别的 val_auc 是否持续上升。最终 val_auc 应接近或超过 exp2 的 0.968。

- [ ] **Step 3: 检查最终结果**

查看 `outputs/exp3_clip_prompt/results_summary.txt`：
- Val AUC ≥ 0.96
- FF++ AUC ≥ 0.97
- CDF AUC > 0.72（目标超越 exp2 的 0.7199）

- [ ] **Step 4: Commit 结果**

```bash
git add outputs/exp3_clip_prompt/
git commit -m "feat(exp3): CLIP prompt contrast training results"
```
