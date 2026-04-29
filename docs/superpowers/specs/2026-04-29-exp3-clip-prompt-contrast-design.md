# Exp3: CLIP + Prompt Contrast 设计文档

## 目标

在 exp2（CLIP visual encoder fine-tune）的基础上，引入类别级文本 prompt 作为辅助语义监督。训练时分类头和 prompt 通路各自产生 logits 并分别计算 CE loss，推理时融合两路概率。验证 prompt 语义先验能否提升跨域泛化能力。

## 架构

### 模型：CLIPPromptBinaryClassifier（修改 `clip_prompt.py`）

**组件：**

| 组件 | 来源 | 是否训练 |
|---|---|---|
| CLIP visual encoder (ViT-B/16) | clip.load() | 可训练（fine-tune） |
| 分类头 nn.Linear(visual_dim, 2) | 新增 | 可训练 |
| CLIP text encoder | clip.load() | **冻结** |
| Prompt 特征 (real/fake) | 预计算存为 buffer | 不可训练 |

**前向传播（训练）：**

```python
def forward(self, images):
    # 1. 视觉特征
    image_features = self.visual(images)           # [B, visual_dim]
    image_features = F.normalize(image_features, dim=-1)

    # 2. 分类头 logits
    cls_logits = self.classifier(image_features.float())  # [B, 2]

    # 3. Prompt logits（温度缩放）
    real_sim = (image_features @ self._real_features.T).mean(dim=1, keepdim=True)  # [B, 1]
    fake_sim = (image_features @ self._fake_features.T).mean(dim=1, keepdim=True)  # [B, 1]
    prompt_logits = torch.cat([real_sim, fake_sim], dim=1) / self.tau  # [B, 2]

    return cls_logits, prompt_logits
```

**初始化逻辑：**

1. 加载完整 CLIP 模型（visual + text encoder）
2. 只保留 `self.visual = clip_model.visual`（与 exp2 一致，不保留 text encoder 引用，节省显存）
3. 预计算 prompt 特征：用冻结的 text encoder 编码 prompt 文本，L2 归一化后存为 registered buffer
4. 新增分类头 `nn.Linear(visual.output_dim, 2)`
5. 注册 `self.tau` 为 buffer（值为 0.07）

### Prompt 文本（更新 `constants.py`）

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

每类 3 条 prompt，编码后得到 shape 为 [3, 768] 的特征矩阵。

### 训练损失（修改 `trainers.py`）

新增 `prompt_contrast_step()` 函数：

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

`run_train_epoch` 中增加分支：
- `loss.name == "cross_entropy_plus_prompt"` → `prompt_contrast_step`
- 其他走原有 `classification_step`

### 推理融合（修改 `run_eval_epoch`）

```python
classifier_prob = softmax(cls_logits, dim=1)[:, 1]   # fake 类概率
prompt_prob = softmax(prompt_logits, dim=1)[:, 1]     # prompt fake 类概率
final_score = (1 - alpha) * classifier_prob + alpha * prompt_prob
```

加权概率平均，`alpha=0.3`（分类头 70%，prompt 30%）。

需要在 `run_eval_epoch` 中判断 `loss.name`，对 prompt 模型使用融合逻辑，其他模型保持原有 softmax 行为。

## 配置

**`configs/exp3_clip_prompt.yaml`：**

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

与 exp2 对齐：batch=128、lr=2e-5、wd=5e-4、epochs=5、patience=3。

## 修改文件清单

| 文件 | 操作 | 内容 |
|---|---|---|
| `src/deepfake_detection/models/clip_prompt.py` | 修改 | 重写模型：dual-path forward，冻结 text encoder，预计算 prompt buffer |
| `src/deepfake_detection/engine/trainers.py` | 修改 | 新增 `prompt_contrast_step`，`run_train_epoch` 增加分支，`run_eval_epoch` 增加融合逻辑 |
| `src/deepfake_detection/data/constants.py` | 修改 | 更新 REAL_PROMPTS 和 FAKE_PROMPTS 为各 3 条 |
| `configs/exp3_clip_prompt.yaml` | 修改 | 新增 loss 配置和训练超参数 |

**不需要修改的文件：**
- `factory.py` — 已有 `"clip_prompt"` 映射，无需改动
- `builders.py` — 已自动检测 CLIP normalization
- `datasets.py` — 已支持 CLIP normalization 参数
- `transforms.py` — 已对齐 DF40 数据增强

## 成功标准

- 在 FF++ 训练集上收敛，val AUC 不低于 exp2 的 0.968
- CDF 跨域 AUC > exp2 的 0.7199（prompt 语义先验应提升泛化）
- 训练显存占用与 exp2 相当（text encoder 冻结且不参与前向传播）
