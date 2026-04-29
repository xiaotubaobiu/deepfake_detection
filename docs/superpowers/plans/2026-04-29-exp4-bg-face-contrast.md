# Exp4: Background-Face Contrast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exp4 — a CLIP-based deepfake detector that learns background-face consistency via InfoNCE contrastive loss with multiple background patches per frame.

**Architecture:** Shared CLIP ViT-B/16 visual encoder with three heads (classifier, bg_projection, face_projection). Training uses triplets of (bg_patches×4, real_face, fake_face) with shared augmentation. Inference fuses classifier probability with background-face consistency score.

**Tech Stack:** PyTorch, CLIP, Albumentations, DDP (torchrun)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/deepfake_detection/data/bg_patches.py` | Create | Crop background patches from anchor frames avoiding face region |
| `src/deepfake_detection/data/datasets.py` | Modify | Add `BgFaceTripletDataset` class |
| `src/deepfake_detection/data/builders.py` | Modify | Add `build_bgface_train_loader`, `build_bgface_eval_loader` |
| `src/deepfake_detection/losses/contrastive.py` | Modify | Add `infonce_bg_face_loss` with multi-patch support |
| `src/deepfake_detection/models/clip_bgcontrast.py` | Modify | Rewrite model with proper dual-projection + classifier architecture |
| `src/deepfake_detection/models/factory.py` | Modify | Wire new model config params |
| `src/deepfake_detection/engine/trainers.py` | Modify | Add `bgface_contrast_step` and inference fusion logic |
| `configs/exp4_clip_prompt_bgcontrast.yaml` | Modify | Update config with correct params |
| `train.py` | Modify | Add loss name handling for exp4 |

---

### Task 1: Background Patch Cropping Utility

**Files:**
- Create: `src/deepfake_detection/data/bg_patches.py`

- [ ] **Step 1: Create `bg_patches.py`**

```python
from __future__ import annotations

import random

import cv2
import numpy as np


def _detect_face_box_from_landmarks(landmarks: np.ndarray, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    margin_x = int((xs.max() - xs.min()) * 0.3)
    margin_y = int((ys.max() - ys.min()) * 0.3)
    x1 = max(0, int(xs.min()) - margin_x)
    y1 = max(0, int(ys.min()) - margin_y)
    x2 = min(img_w, int(xs.max()) + margin_x)
    y2 = min(img_h, int(ys.max()) + margin_y)
    return x1, y1, x2, y2


def _detect_face_box_fallback(img_h: int, img_w: int, margin_ratio: float = 0.2) -> tuple[int, int, int, int]:
    mx = int(img_w * margin_ratio)
    my = int(img_h * margin_ratio)
    return mx, my, img_w - mx, img_h - my


def sample_bg_patches(
    image: np.ndarray,
    num_patches: int = 4,
    patch_size: int = 224,
    face_box: tuple[int, int, int, int] | None = None,
    rng: random.Random | None = None,
) -> list[np.ndarray]:
    """Crop num_patches background patches from image, avoiding face_box region.

    Returns list of (patch_size, patch_size, 3) uint8 arrays.
    """
    if rng is None:
        rng = random.Random()

    h, w = image.shape[:2]

    if face_box is None:
        face_box = _detect_face_box_fallback(h, w)

    fx1, fy1, fx2, fy2 = face_box
    patches = []
    for _ in range(num_patches):
        best = None
        best_dist = -1
        for _ in range(20):  # 20 attempts to find non-overlapping patch
            px = rng.randint(0, max(0, w - patch_size))
            py = rng.randint(0, max(0, h - patch_size))
            # IoU with face box
            ix1 = max(px, fx1)
            iy1 = max(py, fy1)
            ix2 = min(px + patch_size, fx2)
            iy2 = min(py + patch_size, fy2)
            overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            patch_area = patch_size * patch_size
            iou = overlap / patch_area if patch_area > 0 else 0
            dist = 1.0 - iou
            if dist > best_dist:
                best_dist = dist
                best = (px, py)
            if iou == 0:
                break

        if best is not None:
            px, py = best
            patch = image[py:py + patch_size, px:px + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
        else:
            # Fallback: center crop of non-face region
            patches.append(cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR))

    return patches
```

- [ ] **Step 2: Commit**

```bash
git add src/deepfake_detection/data/bg_patches.py
git commit -m "feat: add background patch cropping utility for exp4"
```

---

### Task 2: BgFaceTripletDataset

**Files:**
- Modify: `src/deepfake_detection/data/datasets.py`

- [ ] **Step 1: Add `BgFaceTripletDataset` at the end of `datasets.py`**

```python
class BgFaceTripletDataset(Dataset):
    def __init__(self, triplets, augment=False, img_size=224, num_bg_patches=4,
                 normalize_mean=None, normalize_std=None):
        self.triplets = triplets
        self.augment = augment
        self.img_size = img_size
        self.num_bg_patches = num_bg_patches
        mean = normalize_mean or (0.485, 0.456, 0.406)
        std = normalize_std or (0.229, 0.224, 0.225)
        self.transform = build_rgb_augment(mean=mean, std=std) if augment else build_eval_transform(mean=mean, std=std)
        self.bg_transform = build_rgb_augment(mean=mean, std=std) if augment else build_eval_transform(mean=mean, std=std)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        import random as _random
        from deepfake_detection.data.bg_patches import sample_bg_patches
        from deepfake_detection.data.crops import crop_region, expand_box

        t = self.triplets[idx]

        # Load images
        anchor_img = _load_frame(t["real_frame_path"])  # anchor full frame
        fake_img = _load_frame(t["fake_frame_path"])

        # Resize face crops to img_size
        anchor_resized = cv2_resize(anchor_img, self.img_size)
        fake_resized = cv2_resize(fake_img, self.img_size)

        # Detect face box from landmarks for bg patch avoidance
        face_box = None
        if t.get("fake_landmark_path"):
            try:
                landmarks = np.load(t["fake_landmark_path"])
                h, w = anchor_img.shape[:2]
                face_box = _detect_face_box_from_landmarks(landmarks, h, w)
                face_box = expand_box(face_box, 1.3, h, w)
            except Exception:
                pass

        # Crop background patches from anchor
        rng = _random.Random()
        bg_patches_raw = sample_bg_patches(
            anchor_img, num_patches=self.num_bg_patches,
            patch_size=self.img_size, face_box=face_box, rng=rng,
        )

        # Apply transforms
        if self.augment:
            # Shared augmentation: apply same transform params to fake face and bg patches
            fake_transformed = self.transform(image=fake_resized)["image"]
            bg_tensors = []
            for bp in bg_patches_raw:
                bg_tensors.append(self.bg_transform(image=bp)["image"])
        else:
            fake_transformed = self.transform(image=fake_resized)["image"]
            bg_tensors = [self.bg_transform(image=bp)["image"] for bp in bg_patches_raw]

        # fake face tensor
        fake_tensor = torch.from_numpy(fake_transformed).permute(2, 0, 1).float() / 255.0
        # bg tensors
        bg_stacked = torch.stack([
            torch.from_numpy(bt).permute(2, 0, 1).float() / 255.0 for bt in bg_tensors
        ])

        return {
            "background": bg_stacked,       # (num_bg, 3, H, W)
            "fake_face": fake_tensor,        # (3, H, W)
            "label": torch.tensor(1, dtype=torch.long),
            "video_id": t["pair_id"],
        }
```

Note: Need to add `_detect_face_box_from_landmarks` as a module-level function in datasets.py (reuse from datasets.py existing `_detect_face_box`):

```python
def _detect_face_box_from_landmarks(landmarks, img_h, img_w):
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    margin_x = int((xs.max() - xs.min()) * 0.3)
    margin_y = int((ys.max() - ys.min()) * 0.3)
    x1 = max(0, int(xs.min()) - margin_x)
    y1 = max(0, int(ys.min()) - margin_y)
    x2 = min(img_w, int(xs.max()) + margin_x)
    y2 = min(img_h, int(ys.max()) + margin_y)
    return x1, y1, x2, y2
```

- [ ] **Step 2: Commit**

```bash
git add src/deepfake_detection/data/datasets.py
git commit -m "feat: add BgFaceTripletDataset for exp4"
```

---

### Task 3: InfoNCE Contrastive Loss

**Files:**
- Modify: `src/deepfake_detection/losses/contrastive.py`

- [ ] **Step 1: Add `infonce_bg_face_loss` function**

Append to `contrastive.py`:

```python
def infonce_bg_face_loss(bg_features, real_face_features, fake_face_features, temperature=0.07):
    """InfoNCE loss with multiple bg patches per sample.

    Args:
        bg_features: (B, K, D) — K bg patches per sample, L2-normalized
        real_face_features: (B, D) — real face features, L2-normalized
        fake_face_features: (B, D) — fake face features, L2-normalized
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    B, K, D = bg_features.shape

    # Reshape to (B*K, D)
    bg_flat = bg_features.reshape(B * K, D)

    # Expand face features to match: each bg patch pairs with its sample's face
    real_expanded = real_face_features.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
    fake_expanded = fake_face_features.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

    # Positive similarities: (bg, real_face) for same sample
    pos_sim = (bg_flat * real_expanded).sum(dim=1) / temperature  # (B*K,)

    # Negative similarities: (bg, fake_face) for same sample
    neg_same = (bg_flat * fake_expanded).sum(dim=1) / temperature  # (B*K,)

    # Cross-sample negatives: bg with all other real/fake faces
    # real_face_features: (B, D), bg_flat: (B*K, D)
    cross_real = torch.mm(bg_flat, real_face_features.t()) / temperature  # (B*K, B)
    cross_fake = torch.mm(bg_flat, fake_face_features.t()) / temperature  # (B*K, B)

    # Mask out self-pairs (bg_i with its own real_face)
    mask = torch.arange(B, device=bg_features.device).unsqueeze(1).expand(B, K).reshape(B * K)
    cross_real[mask, torch.arange(B, device=bg_features.device).unsqueeze(1).expand(B, K).reshape(B * K)] = float('-inf')

    # Concatenate all negatives: same-sample fake + cross-sample real + cross-sample fake
    all_neg = torch.cat([neg_same.unsqueeze(1), cross_real, cross_fake], dim=1)  # (B*K, 1+2B)

    # Logits: positive + all negatives
    logits = torch.cat([pos_sim.unsqueeze(1), all_neg], dim=1)  # (B*K, 2+2B)

    # Labels: positive is always index 0
    labels = torch.zeros(B * K, dtype=torch.long, device=bg_features.device)

    return F.cross_entropy(logits, labels)
```

- [ ] **Step 2: Commit**

```bash
git add src/deepfake_detection/losses/contrastive.py
git commit -m "feat: add InfoNCE bg-face contrastive loss for exp4"
```

---

### Task 4: Model — CLIP BgFace Contrast

**Files:**
- Modify: `src/deepfake_detection/models/clip_bgcontrast.py`

- [ ] **Step 1: Rewrite `clip_bgcontrast.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS
from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts


class CLIPBgFaceContrastModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", projection_dim: int = 256):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device="cpu")

        self.visual = clip_model.visual
        feat_dim = self.visual.output_dim

        self.classifier = nn.Linear(feat_dim, 2)
        self.bg_projection = nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.face_projection = nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # Pre-compute prompt features
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
        self.register_buffer("tau", torch.tensor(0.07))

    def encode_face(self, face_images):
        """Encode face crop(s). Accepts (B, 3, H, W)."""
        return F.normalize(self.visual(face_images), dim=-1)

    def encode_bg(self, bg_images):
        """Encode background patches. Accepts (B, 3, H, W) or (B, K, 3, H, W)."""
        if bg_images.dim() == 5:
            B, K, C, H, W = bg_images.shape
            flat = bg_images.reshape(B * K, C, H, W)
            feats = F.normalize(self.visual(flat), dim=-1)
            return feats.reshape(B, K, -1)
        return F.normalize(self.visual(bg_images), dim=-1)

    def forward(self, images):
        """Standard classification forward (for eval without bg)."""
        image_features = self.encode_face(images)
        return self.classifier(image_features.float())

    def forward_classification(self, face_images):
        """Classification logits from face crop."""
        face_feat = self.encode_face(face_images)
        return self.classifier(face_feat.float())

    def forward_contrastive(self, bg_images, face_images):
        """Project bg and face features for contrastive loss.

        Args:
            bg_images: (B, K, 3, H, W) background patches
            face_images: (B, 3, H, W) face crops (real or fake)

        Returns:
            bg_proj: (B, K, proj_dim) L2-normalized
            face_proj: (B, proj_dim) L2-normalized
        """
        bg_feat = self.encode_bg(bg_images)
        face_feat = self.encode_face(face_images)

        B, K, D = bg_feat.shape
        bg_proj = F.normalize(self.bg_projection(bg_feat.reshape(B * K, D).float()), dim=-1)
        bg_proj = bg_proj.reshape(B, K, -1)
        face_proj = F.normalize(self.face_projection(face_feat.float()), dim=-1)
        return bg_proj, face_proj

    def forward_prompt_logits(self, face_images):
        """Prompt-based logits from face crop."""
        face_feat = self.encode_face(face_images)
        real_sim = (face_feat @ self._real_features.T).mean(dim=1, keepdim=True)
        fake_sim = (face_feat @ self._fake_features.T).mean(dim=1, keepdim=True)
        return torch.cat([real_sim, fake_sim], dim=1) / self.tau

    def compute_consistency(self, bg_images, face_images):
        """Compute bg-face consistency score for inference.

        Returns:
            consistency: (B,) scores — higher means more consistent (likely real)
        """
        bg_proj, face_proj = self.forward_contrastive(bg_images, face_images)
        # (B, K, D) @ (B, D) -> (B, K)
        sims = (bg_proj * face_proj.unsqueeze(1)).sum(dim=-1)
        return sims.mean(dim=1)  # (B,)
```

- [ ] **Step 2: Commit**

```bash
git add src/deepfake_detection/models/clip_bgcontrast.py
git commit -m "feat: rewrite CLIP bg-face contrast model for exp4"
```

---

### Task 5: Factory and Config Wiring

**Files:**
- Modify: `src/deepfake_detection/models/factory.py`
- Modify: `configs/exp4_clip_prompt_bgcontrast.yaml`

- [ ] **Step 1: Update `factory.py` — change the `clip_prompt_bgcontrast` branch**

Replace the existing `clip_prompt_bgcontrast` case in `build_model`:

```python
    if name == "clip_prompt_bgcontrast":
        proj_dim = model_cfg.get("projection_dim", 256)
        return CLIPBgFaceContrastModel(clip_name, proj_dim)
```

Full updated `build_model`:

```python
def build_model(model_cfg: dict):
    name = model_cfg["name"]
    clip_name = model_cfg.get("clip_model_name", "ViT-B/16")
    if name == "efficientnet_b0":
        return EfficientNetBinaryClassifier()
    if name == "clip_finetune":
        return CLIPFineTuneBinaryClassifier(clip_name)
    if name == "clip_prompt":
        return CLIPPromptBinaryClassifier(clip_name)
    if name == "clip_prompt_bgcontrast":
        proj_dim = model_cfg.get("projection_dim", 256)
        return CLIPBgFaceContrastModel(clip_name, proj_dim)
    raise ValueError(f"Unknown model name: {name}")
```

- [ ] **Step 2: Update `configs/exp4_clip_prompt_bgcontrast.yaml`**

```yaml
_base_: configs/base.yaml
experiment_name: exp4_clip_prompt_bgcontrast
model:
  name: clip_prompt_bgcontrast
  clip_model_name: ViT-B/16
  projection_dim: 256
loss:
  name: cross_entropy_plus_bgface_contrast
  lambda_align: 0.1
  temperature: 0.07
  alpha: 0.3
  num_bg_patches: 4
data:
  require_aligned_triplets: true
train:
  per_gpu_batch: 128
  lr: 2.0e-05
  weight_decay: 0.0005
  epochs: 5
  patience: 3
  seed: 42
```

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/models/factory.py configs/exp4_clip_prompt_bgcontrast.yaml
git commit -m "feat: wire exp4 model factory and config"
```

---

### Task 6: Data Builders for BgFace Triplets

**Files:**
- Modify: `src/deepfake_detection/data/builders.py`

- [ ] **Step 1: Add `build_bgface_train_loader` and `build_bgface_eval_loader`**

Add after `build_val_loader`:

```python
def build_bgface_train_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    num_bg_patches = cfg.get("loss", {}).get("num_bg_patches", 4)
    from deepfake_detection.data.constants import CLIP_MEAN, CLIP_STD

    all_triplets = []
    for method in methods:
        triplets = index_train_aligned_triplets(root, method, max_videos)
        all_triplets.extend(triplets)

    dataset = BgFaceTripletDataset(
        all_triplets, augment=True,
        num_bg_patches=num_bg_patches,
        normalize_mean=CLIP_MEAN, normalize_std=CLIP_STD,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), num_workers=num_workers,
                      pin_memory=True, drop_last=True,
                      worker_init_fn=_worker_init_fn)


def build_bgface_eval_loader(cfg, domain="ffpp", distributed=True):
    """Eval loader for exp4: returns face crops (no bg patches)."""
    return build_eval_loader(cfg, domain=domain, distributed=distributed)
```

- [ ] **Step 2: Add import for `BgFaceTripletDataset` at the top of builders.py**

Add `BgFaceTripletDataset` to the import from datasets:

```python
from deepfake_detection.data.datasets import FrameClassificationDataset, AlignedTripletDataset, BgFaceTripletDataset
```

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/data/builders.py
git commit -m "feat: add bgface triplet data loaders for exp4"
```

---

### Task 7: Trainer — Training Step and Inference Fusion

**Files:**
- Modify: `src/deepfake_detection/engine/trainers.py`

- [ ] **Step 1: Add `bgface_contrast_step` function after `prompt_contrast_step`**

```python
def bgface_contrast_step(model, batch, device, lambda_align=0.1, temperature=0.07):
    bg_images = batch["background"].to(device)  # (B, K, 3, H, W)
    face_images = batch["fake_face"].to(device)  # (B, 3, H, W) — these are all fake
    labels = batch["label"].to(device)

    # Classification on face
    cls_logits = model.forward_classification(face_images)
    cls_loss = F.cross_entropy(cls_logits, labels)

    # Contrastive: bg vs (real, fake) face
    # We need real face features — but batch only has fake faces labeled as fake
    # Use the same face as "fake" in contrastive, and the model learns bg should not match fake faces
    from deepfake_detection.losses.contrastive import infonce_bg_face_loss
    bg_proj, fake_face_proj = model.forward_contrastive(bg_images, face_images)
    # For real face proxy: use the same bg images as "real" (bg should be consistent with itself)
    # Actually, we need to restructure: the contrastive loss needs explicit real and fake face features
    # The triplet dataset provides fake faces; real faces come from the anchor
    # We need to also pass real face images in the batch

    # Simplified: use same-frame bg as positive anchor (bg patches are from same frame)
    # This is a self-supervised signal: bg patches should be consistent with each other
    # and inconsistent with fake faces
    B, K, D = bg_proj.shape
    # Use mean of bg patches as "real face proxy"
    bg_mean = bg_proj.mean(dim=1)  # (B, D)

    contrastive_loss = infonce_bg_face_loss(bg_proj, bg_mean.detach(), fake_face_proj.unsqueeze(1).expand(B, K, D).reshape(B * K, D).reshape(B, K, -1) if False else fake_face_proj, temperature)

    total_loss = cls_loss + lambda_align * contrastive_loss
    return total_loss, cls_logits
```

Actually, this needs rethinking. Let me reconsider the approach:

**Revised `bgface_contrast_step`:**

The dataset returns `(bg_patches, fake_face, label)`. We need real face for the contrastive loss. Two options:
1. Also load real face in the dataset → larger batch
2. Use the anchor frame's face region as real face

The simplest correct approach: **the dataset should also return real_face.**

Going back to Task 2, the dataset should return both real and fake face. Let me revise the approach:

**Revised dataset return:**
```python
return {
    "background": bg_stacked,       # (K, 3, H, W)
    "real_face": real_face_tensor,  # (3, H, W) — from anchor, face crop using landmarks
    "fake_face": fake_tensor,       # (3, H, W)
    "label": torch.tensor(1, dtype=torch.long),
    "video_id": t["pair_id"],
}
```

Where `real_face` is cropped from the anchor frame using the same landmark-based face box used for fake_face.

**Revised `bgface_contrast_step`:**

```python
def bgface_contrast_step(model, batch, device, lambda_align=0.1, temperature=0.07):
    bg_images = batch["background"].to(device)       # (B, K, 3, H, W)
    real_face = batch["real_face"].to(device)          # (B, 3, H, W)
    fake_face = batch["fake_face"].to(device)          # (B, 3, H, W)
    labels = batch["label"].to(device)

    cls_logits = model.forward_classification(fake_face)
    cls_loss = F.cross_entropy(cls_logits, labels)

    from deepfake_detection.losses.contrastive import infonce_bg_face_loss
    bg_proj, real_face_proj = model.forward_contrastive(bg_images, real_face)
    _, fake_face_proj = model.forward_contrastive(bg_images, fake_face)
    contrastive_loss = infonce_bg_face_loss(bg_proj, real_face_proj, fake_face_proj, temperature)

    total_loss = cls_loss + lambda_align * contrastive_loss
    return total_loss, cls_logits
```

- [ ] **Step 2: Update `run_train_epoch` to handle `cross_entropy_plus_bgface_contrast` loss**

Add in the loss name check section:

```python
    is_bgface = loss_name == "cross_entropy_plus_bgface_contrast"
```

And in the loop:

```python
        with autocast(enabled=True):
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            elif is_prompt:
                loss, _, _ = prompt_contrast_step(model, batch, device, beta)
            elif is_bgface:
                loss, _ = bgface_contrast_step(model, batch, device, lambda_align, temperature)
            else:
                loss, _ = classification_step(model, batch, device)
```

- [ ] **Step 3: Update `run_eval_epoch` for exp4 inference fusion**

The eval loader returns standard face crops (no bg). For exp4 during eval, we use classifier + prompt fusion (same as exp3) since we don't have bg patches at eval time in the standard pipeline.

Add `is_bgface` check and use classifier-only or classifier+prompt inference:

```python
    is_bgface = loss_name == "cross_entropy_plus_bgface_contrast"
```

In the loop, bgface model's `forward()` already returns classification logits, so the existing `else` branch handles it:

```python
        with autocast(enabled=True):
            output = model(images)
        logits = output
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
```

This works because `CLIPBgFaceContrastModel.forward()` returns classification logits.

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/engine/trainers.py
git commit -m "feat: add bgface contrast training step for exp4"
```

---

### Task 8: Update train.py for exp4 Loss

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add exp4 train loader branch**

In `main()`, after the existing loader building, add conditional loader for bgface:

```python
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name == "cross_entropy_plus_bgface_contrast":
        from deepfake_detection.data.builders import build_bgface_train_loader
        train_loader = build_bgface_train_loader(cfg, distributed=torch.distributed.is_initialized())
```

Replace the existing `train_loader = build_train_loader(...)` line with:

```python
    loss_name = cfg.get("loss", {}).get("name", "")
    if loss_name == "cross_entropy_plus_bgface_contrast":
        from deepfake_detection.data.builders import build_bgface_train_loader
        train_loader = build_bgface_train_loader(cfg, distributed=torch.distributed.is_initialized())
    else:
        train_loader = build_train_loader(cfg, distributed=torch.distributed.is_initialized())
```

- [ ] **Step 2: Commit**

```bash
git add train.py
git commit -m "feat: wire exp4 bgface train loader in train.py"
```

---

### Task 9: Revise BgFaceTripletDataset to Include Real Face

**Files:**
- Modify: `src/deepfake_detection/data/datasets.py`

The `BgFaceTripletDataset.__getitem__` from Task 2 needs to also crop real_face from the anchor frame using the face landmarks. Update the return dict:

```python
        # Crop real face from anchor using same face box
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            real_face_crop = crop_region(anchor_img, (fx1, fy1, fx2, fy2), self.img_size)
        else:
            real_face_crop = cv2_resize(anchor_img, self.img_size)

        real_transformed = self.transform(image=real_face_crop)["image"]
        real_tensor = torch.from_numpy(real_transformed).permute(2, 0, 1).float() / 255.0

        return {
            "background": bg_stacked,
            "real_face": real_tensor,
            "fake_face": fake_tensor,
            "label": torch.tensor(1, dtype=torch.long),
            "video_id": t["pair_id"],
        }
```

- [ ] **Commit**

```bash
git add src/deepfake_detection/data/datasets.py
git commit -m "feat: add real face crop to BgFaceTripletDataset"
```

---

### Task 10: Run exp4 seed=42

**Files:** None (run only)

- [ ] **Step 1: Launch training**

```bash
PYTHONPATH=src torchrun --nproc_per_node=8 --master_port=29520 train.py --config configs/exp4_clip_prompt_bgcontrast.yaml
```

Expected: 5 epochs, ~5 min per epoch. Should output Val AUC, FF++ AUC, CDF AUC.

- [ ] **Step 2: Check results**

```bash
cat outputs/exp4_clip_prompt_bgcontrast/results_summary.txt
```

Expected: CDF AUC competitive with or better than exp2 baseline (~0.73+).

- [ ] **Step 3: Commit results**

```bash
git add outputs/exp4_clip_prompt_bgcontrast/
git commit -m "feat: exp4 seed=42 results"
```

---

### Task 11: Exp4 Seed Robustness (seed=7, seed=123)

**Files:**
- Create: `configs/exp4_clip_prompt_bgcontrast_s7.yaml`
- Create: `configs/exp4_clip_prompt_bgcontrast_s123.yaml`

- [ ] **Step 1: Create seed configs and run**

Same pattern as exp2/exp3 seed configs — copy exp4 config, change `experiment_name` and `train.seed`.

Run each with deterministic training pipeline, collect results.

- [ ] **Step 2: Compare exp4 vs exp2 vs exp3 across all seeds**

Target: exp4 CDF AUC stably above exp2 baseline across all 3 seeds.
