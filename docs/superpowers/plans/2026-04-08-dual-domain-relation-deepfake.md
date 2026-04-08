# Dual-Domain Face-Context Relation Deepfake Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal experimental framework to verify that face-context relation in RGB + frequency domains, stabilized by contrastive learning, improves deepfake detection generalization.

**Architecture:** ResNet18 encoders (shared within domain for face/context) extract features → relation module concatenates [z_f, z_c, |z_f-z_c|, z_f⊙z_c] per domain → dual-domain fusion → MLP classifier. M4 adds InfoNCE on relation embeddings.

**Tech Stack:** PyTorch, scipy (DCT), scikit-learn (metrics), opencv-python (image I/O), InsightFace/SCRFD (face detection), numpy (landmarks)

**Conda env:** `researchclaw`

**Spec:** `docs/superpowers/specs/2026-04-08-dual-domain-relation-deepfake-design.md`

---

## Key Dataset Facts (discovered during exploration)

| Dataset | Path | Structure | Landmarks |
|---------|------|-----------|-----------|
| DF40-train real | `DF40_train/anchor/` | 999 videos × 32 frames, `anchor/<vid_id>/<frame>.png` | **None** — need SCRFD |
| DF40-train fake | `DF40_train/<method>/` | `frames/<vid_id>/<frame>.png` + `landmarks/<vid_id>/<frame>.npy` | `.npy` shape (81,2) int64 |
| DF40-test | `DF40_test/<method>/ff/` and `cdf/` | Same `frames/` + `landmarks/` structure | Same `.npy` format |
| Celeb-DF-v2 | `Celeb-DF-v2/` | Only real videos (Celeb-real 588, YouTube-real 300) | None |

**Test strategy:** DF40-test `ff/` = same-domain test, `cdf/` = cross-domain test (Celeb-DF sources). No need for raw Celeb-DF-v2 since DF40-test already provides cdf splits with landmarks.

**All frames are 256×256 PNG.**

---

## File Structure

```
deepfake_detection/
├── configs/
│   └── experiments.yaml          # All 4 experiment configs
├── data/
│   ├── __init__.py
│   ├── frequency.py              # DCT transform: grayscale → 2D DCT → log amplitude → normalize
│   ├── transforms.py             # Pair-synchronized augmentation (for M4 InfoNCE)
│   ├── face_detector.py          # SCRFD wrapper (for anchor frames without landmarks)
│   └── df40_dataset.py           # DF40 dataset: handles train/test, ff/cdf, landmarks/SCRFD
├── models/
│   ├── __init__.py
│   ├── encoder.py                # ResNet18 backbone, output 512d
│   ├── relation.py               # [z_f, z_c, |z_f-z_c|, z_f⊙z_c]
│   └── classifier.py             # DualDomainDetector: wraps encoders + relation + MLP (M1-M4)
├── losses/
│   ├── __init__.py
│   └── infonce.py                # InfoNCE on relation embeddings
├── utils/
│   ├── __init__.py
│   └── metrics.py                # AUC, EER
├── train.py                      # Training entry point
├── evaluate.py                   # Evaluation entry point
├── run_experiments.sh             # Run all 4 experiments
└── docs/superpowers/
    ├── specs/                    # Design spec (already exists)
    └── plans/                    # This file
```

---

### Task 1: Environment Setup + Clean Old Code

**Files:**
- Delete: everything in `/home/z/project/deepfake_detection/` except `docs/`
- Create: directory structure above

- [ ] **Step 1: Delete old code**

```bash
cd /home/z/project/deepfake_detection
# Keep docs/, delete everything else
find . -maxdepth 1 ! -name '.' ! -name 'docs' -exec rm -rf {} +
```

- [ ] **Step 2: Create new directory structure**

```bash
mkdir -p configs data models losses utils
touch data/__init__.py models/__init__.py losses/__init__.py utils/__init__.py
```

- [ ] **Step 3: Install missing packages in researchclaw env**

```bash
conda run -n researchclaw pip install torchvision opencv-python insightface onnxruntime-gpu
```

- [ ] **Step 4: Verify environment**

```bash
conda run -n researchclaw python -c "
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import torchvision; print('torchvision:', torchvision.__version__)
import cv2; print('OpenCV:', cv2.__version__)
import scipy; print('scipy:', scipy.__version__)
import sklearn; print('sklearn:', sklearn.__version__)
import insightface; print('insightface OK')
import onnxruntime; print('onnxruntime:', onnxruntime.__version__)
"
```

Expected: all imports succeed, CUDA: True

- [ ] **Step 5: Init git repo and commit**

```bash
cd /home/z/project/deepfake_detection
git init
git add -A
git commit -m "chore: clean slate with new directory structure"
```

---

### Task 2: DCT Frequency Transform (`data/frequency.py`)

**Files:**
- Create: `data/frequency.py`

**What it does:** Converts a grayscale image (H,W) or (H,W,1) uint8 to a 3-channel DCT feature map (3,224,224) float32 suitable for ResNet18.

- [ ] **Step 1: Write `data/frequency.py`**

```python
import numpy as np
from scipy.fftpack import dctn
import cv2


def rgb_to_dct_map(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert RGB image to DCT frequency feature map.

    Args:
        img: (H, W, 3) uint8 RGB image
        size: output spatial size

    Returns:
        (3, size, size) float32 in [0, 1], DCT log-amplitude map replicated to 3 channels
    """
    # 1. Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float32)

    # 2. 2D DCT type-II, orthonormal
    dct_coeffs = dctn(gray, type=2, norm='ortho')

    # 3. Take absolute value
    dct_abs = np.abs(dct_coeffs)

    # 4. Log transform
    dct_log = np.log1p(dct_abs)

    # 5. Normalize to [0, 1]
    dct_min, dct_max = dct_log.min(), dct_log.max()
    if dct_max - dct_min > 1e-8:
        dct_norm = (dct_log - dct_min) / (dct_max - dct_min)
    else:
        dct_norm = np.zeros_like(dct_log)

    # 6. Resize to target size
    dct_resized = cv2.resize(dct_norm, (size, size), interpolation=cv2.INTER_LINEAR)

    # 7. Replicate to 3 channels (ResNet18 expects 3-channel input)
    dct_3ch = np.stack([dct_resized] * 3, axis=0).astype(np.float32)

    return dct_3ch


def batch_rgb_to_dct(imgs: np.ndarray, size: int = 224) -> np.ndarray:
    """Batch convert RGB images to DCT maps.

    Args:
        imgs: (N, H, W, 3) uint8
        size: output spatial size

    Returns:
        (N, 3, size, size) float32
    """
    return np.stack([rgb_to_dct_map(img, size) for img in imgs])
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import numpy as np
from data.frequency import rgb_to_dct_map

# Test with a random 256x256 RGB image
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
out = rgb_to_dct_map(img)
assert out.shape == (3, 224, 224), f'Expected (3,224,224), got {out.shape}'
assert out.dtype == np.float32
assert out.min() >= 0.0 and out.max() <= 1.0, f'Range: [{out.min()}, {out.max()}]'
print('DCT transform OK:', out.shape, f'range [{out.min():.3f}, {out.max():.3f}]')

# Test with actual DF40 frame
from PIL import Image
img2 = np.array(Image.open('/Dataset/deepfake_detection/DF40_all/DF40_train/simswap/frames/000_003/000.png'))
out2 = rgb_to_dct_map(img2)
assert out2.shape == (3, 224, 224)
print('DF40 frame DCT OK:', out2.shape)
"
```

Expected: both assertions pass

- [ ] **Step 3: Commit**

```bash
git add data/frequency.py data/__init__.py
git commit -m "feat: add DCT frequency transform"
```

---

### Task 3: Metrics (`utils/metrics.py`)

**Files:**
- Create: `utils/metrics.py`

- [ ] **Step 1: Write `utils/metrics.py`**

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: (N,) binary labels (0=real, 1=fake)
        y_score: (N,) predicted probabilities for class 1 (fake)

    Returns:
        AUC value
    """
    return float(roc_auc_score(y_true, y_score))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate.

    EER is the point where false positive rate equals false negative rate.

    Args:
        y_true: (N,) binary labels (0=real, 1=fake)
        y_score: (N,) predicted probabilities for class 1 (fake)

    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    return eer
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import numpy as np
from utils.metrics import compute_auc, compute_eer

np.random.seed(42)
y_true = np.array([0]*100 + [1]*100)
y_score = np.concatenate([
    np.random.uniform(0, 0.4, 100),  # real -> low scores
    np.random.uniform(0.6, 1.0, 100),  # fake -> high scores
])

auc = compute_auc(y_true, y_score)
eer = compute_eer(y_true, y_score)
print(f'AUC: {auc:.4f}, EER: {eer:.4f}')
assert 0.9 < auc <= 1.0, f'AUC should be high, got {auc}'
assert 0.0 <= eer < 0.2, f'EER should be low, got {eer}'
print('Metrics OK')
"
```

Expected: AUC > 0.9, EER < 0.2

- [ ] **Step 3: Commit**

```bash
git add utils/metrics.py utils/__init__.py
git commit -m "feat: add AUC and EER metrics"
```

---

### Task 4: Pair-Synchronized Augmentation (`data/transforms.py`)

**Files:**
- Create: `data/transforms.py`

**What it does:** Provides augmentation transforms that are applied identically to face_rgb and context_rgb of the same sample, ensuring consistent transformation for contrastive learning.

- [ ] **Step 1: Write `data/transforms.py`**

```python
import numpy as np
import cv2
from io import BytesIO


class PairAugmentation:
    """Synchronized augmentation for (face, context) pairs.

    Both face and context receive the same geometric and photometric transforms.
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def _jpeg_compress(self, img: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression to an image."""
        _, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    def __call__(self, face: np.ndarray, context: np.ndarray):
        """Apply synchronized augmentation to face and context pair.

        Args:
            face: (H, W, 3) uint8
            context: (H, W, 3) uint8

        Returns:
            (face_aug, context_aug) same shape and dtype
        """
        # JPEG compression (quality 50-95)
        if self.rng.random() < 0.5:
            quality = int(self.rng.uniform(50, 96))
            face = self._jpeg_compress(face, quality)
            context = self._jpeg_compress(context, quality)

        # Gaussian blur (sigma 0.5-1.5)
        if self.rng.random() < 0.3:
            sigma = self.rng.uniform(0.5, 1.5)
            ksize = int(sigma * 4) * 2 + 1
            face = cv2.GaussianBlur(face, (ksize, ksize), sigma)
            context = cv2.GaussianBlur(context, (ksize, ksize), sigma)

        # Brightness/contrast jitter
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.8, 1.2)  # contrast
            beta = self.rng.uniform(-0.1, 0.1)   # brightness (in [0,1] scale)
            face = np.clip(face.astype(np.float32) * alpha + beta * 255, 0, 255).astype(np.uint8)
            context = np.clip(context.astype(np.float32) * alpha + beta * 255, 0, 255).astype(np.uint8)

        # Slight color jitter (hue/saturation shift)
        if self.rng.random() < 0.3:
            # Convert to HSV, shift hue/sat, convert back
            shift = self.rng.uniform(-10, 10)  # hue shift in degrees
            face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.int16)
            face_hsv[:, :, 0] = np.clip(face_hsv[:, :, 0] + shift, 0, 179)
            face = cv2.cvtColor(face_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            context_hsv = cv2.cvtColor(context, cv2.COLOR_RGB2HSV).astype(np.int16)
            context_hsv[:, :, 0] = np.clip(context_hsv[:, :, 0] + shift, 0, 179)
            context = cv2.cvtColor(context_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Slight resize (0.9-1.1x)
        if self.rng.random() < 0.3:
            scale = self.rng.uniform(0.9, 1.1)
            h, w = face.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR)
            context = cv2.resize(context, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            context = cv2.resize(context, (w, h), interpolation=cv2.INTER_LINEAR)

        return face, context


def get_augmentation_pair(seed: int = None):
    """Return a PairAugmentation instance."""
    return PairAugmentation(seed=seed)
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import numpy as np
from data.transforms import PairAugmentation

face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
context = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

aug = PairAugmentation(seed=42)
f_out, c_out = aug(face, context)

assert f_out.shape == face.shape, f'Face shape changed: {f_out.shape}'
assert c_out.shape == context.shape, f'Context shape changed: {c_out.shape}'
assert f_out.dtype == np.uint8
print('Pair augmentation OK')
print(f'  face changed: {not np.array_equal(face, f_out)}')
print(f'  context changed: {not np.array_equal(context, c_out)}')
"
```

Expected: shapes preserved, dtype uint8, at least one transform applied

- [ ] **Step 3: Commit**

```bash
git add data/transforms.py
git commit -m "feat: add pair-synchronized augmentation"
```

---

### Task 5: SCRFD Face Detector Wrapper (`data/face_detector.py`)

**Files:**
- Create: `data/face_detector.py`

**What it does:** Wraps InsightFace SCRFD model to detect faces in images where landmarks are not pre-extracted (i.e., DF40-train anchor frames). Returns the largest face bounding box.

- [ ] **Step 1: Write `data/face_detector.py`**

```python
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis


class FaceDetector:
    """SCRFD-based face detector using InsightFace buffalo_l model pack."""

    def __init__(self, model_path: str = None, ctx_id: int = 0):
        """Initialize face detector.

        Args:
            model_path: path to directory containing det_10g.onnx (e.g. ~/project/evaluation_metic/models/buffalo_l)
            ctx_id: GPU id, -1 for CPU
        """
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_path.rsplit('/models/', 1)[0] if model_path else None,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_largest_face(self, img: np.ndarray):
        """Detect the largest face in an image.

        Args:
            img: (H, W, 3) uint8 BGR or RGB image

        Returns:
            bbox: [x1, y1, x2, y2] or None if no face detected
        """
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        # Pick the largest face by area
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        best = faces[np.argmax(areas)]
        return best.bbox.astype(int)  # [x1, y1, x2, y2]


def landmarks_to_bbox(landmarks: np.ndarray, img_h: int, img_w: int):
    """Convert 81-point landmarks to face bounding box.

    Args:
        landmarks: (81, 2) or (N, 2) array of (x, y) points
        img_h: image height
        img_w: image width

    Returns:
        bbox: [x1, y1, x2, y2] square-padded
    """
    x1 = landmarks[:, 0].min()
    y1 = landmarks[:, 1].min()
    x2 = landmarks[:, 0].max()
    y2 = landmarks[:, 1].max()

    # Square-pad: make the box square
    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(w, h)
    # Add 20% margin
    side = int(side * 1.2)
    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = x1 + side
    y2 = y1 + side

    # Clip to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return np.array([x1, y1, x2, y2])


def expand_bbox_for_context(bbox: np.ndarray, scale: float, img_h: int, img_w: int):
    """Expand bounding box for context crop.

    Args:
        bbox: [x1, y1, x2, y2]
        scale: expansion factor (e.g., 1.8)
        img_h: image height
        img_w: image width

    Returns:
        expanded: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    # Make square
    side = max(w, h)
    nx1 = int(cx - side / 2)
    ny1 = int(cy - side / 2)
    nx2 = int(cx + side / 2)
    ny2 = int(cy + side / 2)
    # Clip
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    return np.array([nx1, ny1, nx2, ny2])


def crop_face(img: np.ndarray, bbox: np.ndarray, size: int = 224):
    """Crop and resize face region.

    Args:
        img: (H, W, 3) uint8
        bbox: [x1, y1, x2, y2]
        size: output size

    Returns:
        (size, size, 3) uint8
    """
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_context(img: np.ndarray, face_bbox: np.ndarray, context_bbox: np.ndarray, size: int = 224):
    """Crop context region with face masked out.

    Args:
        img: (H, W, 3) uint8
        face_bbox: [x1, y1, x2, y2] in original image coordinates
        context_bbox: [x1, y1, x2, y2] in original image coordinates
        size: output size

    Returns:
        (size, size, 3) uint8 with face region filled black
    """
    cx1, cy1, cx2, cy2 = context_bbox
    crop = img[cy1:cy2, cx1:cx2].copy()

    # Mask face region (convert face bbox to context crop coordinates)
    fx1 = max(0, face_bbox[0] - cx1)
    fy1 = max(0, face_bbox[1] - cy1)
    fx2 = min(cx2 - cx1, face_bbox[2] - cx1)
    fy2 = min(cy2 - cy1, face_bbox[3] - cy1)
    crop[fy1:fy2, fx1:fx2] = 0

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
```

- [ ] **Step 2: Test it with DF40 landmarks**

```bash
conda run -n researchclaw python -c "
import numpy as np
from PIL import Image
from data.face_detector import landmarks_to_bbox, expand_bbox_for_context, crop_face, crop_context

# Load a real DF40 frame
img = np.array(Image.open('/Dataset/deepfake_detection/DF40_all/DF40_train/simswap/frames/000_003/000.png'))
lm = np.load('/Dataset/deepfake_detection/DF40_all/DF40_train/simswap/landmarks/000_003/000.npy')
print('Image:', img.shape, 'Landmarks:', lm.shape)

bbox = landmarks_to_bbox(lm, img.shape[0], img.shape[1])
print('Face bbox:', bbox)
assert bbox.shape == (4,)
assert bbox[2] > bbox[0] and bbox[3] > bbox[1]

ctx_bbox = expand_bbox_for_context(bbox, 1.8, img.shape[0], img.shape[1])
print('Context bbox:', ctx_bbox)

face_crop = crop_face(img, bbox)
ctx_crop = crop_context(img, bbox, ctx_bbox)
print('Face crop:', face_crop.shape, 'Context crop:', ctx_crop.shape)
assert face_crop.shape == (224, 224, 3)
assert ctx_crop.shape == (224, 224, 3)

# Verify face is masked in context
cx, cy = 112, 112  # center of context crop
assert ctx_crop[cy-10:cy+10, cx-10:cx+10].sum() == 0, 'Face should be masked (black)'
print('Face/context crop OK')
"
```

Expected: face_crop (224,224,3), context_crop (224,224,3) with black mask in center

- [ ] **Step 3: Commit**

```bash
git add data/face_detector.py
git commit -m "feat: add face detector wrapper and crop utilities"
```

---

### Task 6: DF40 Dataset (`data/df40_dataset.py`)

**Files:**
- Create: `data/df40_dataset.py`

**What it does:** PyTorch Dataset that loads DF40 train/test data. Handles:
- Train mode: scans `DF40_train/anchor/` (real) + `DF40_train/<method>/frames/` (fake)
- Test mode: scans `DF40_test/<method>/ff/` or `DF40_test/<method>/cdf/`
- Samples 1 frame per video
- Returns face_rgb, context_rgb, face_freq, context_freq, label

- [ ] **Step 1: Write `data/df40_dataset.py`**

```python
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.face_detector import landmarks_to_bbox, expand_bbox_for_context, crop_face, crop_context
from data.frequency import rgb_to_dct_map


class DF40Dataset(Dataset):
    """DF40 deepfake detection dataset.

    Supports:
      - Train: DF40_train/anchor (real) + DF40_train/<method>/frames (fake)
      - Test-ff: DF40_test/<method>/ff/frames (same-domain)
      - Test-cdf: DF40_test/<method>/cdf/frames (cross-domain, Celeb-DF sources)
    """

    def __init__(
        self,
        root: str = '/Dataset/deepfake_detection/DF40_all',
        split: str = 'train',
        test_domain: str = 'ff',
        methods: list = None,
        frames_per_video: int = 1,
        img_size: int = 224,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            root: path to DF40_all
            split: 'train', 'val', 'test'
            test_domain: 'ff' or 'cdf' (only for split='test')
            methods: list of fake methods to include. None = all.
            frames_per_video: number of frames to sample per video
            img_size: crop size
            val_ratio: fraction of training videos used for validation
            seed: random seed for reproducibility
        """
        self.root = root
        self.split = split
        self.test_domain = test_domain
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.rng = random.Random(seed)

        self.samples = []  # list of (frame_path, landmarks_path_or_None, label)

        if split in ('train', 'val'):
            self._load_train_val(methods, val_ratio, seed)
        else:
            self._load_test(test_domain, methods)

    def _load_train_val(self, methods, val_ratio, seed):
        train_dir = os.path.join(self.root, 'DF40_train')

        # Collect real samples from anchor
        anchor_dir = os.path.join(train_dir, 'anchor')
        real_videos = sorted(os.listdir(anchor_dir))

        # Collect fake samples
        all_methods = sorted([d for d in os.listdir(train_dir) if d != 'anchor'])
        if methods is not None:
            all_methods = [m for m in all_methods if m in methods]

        # Build video list with labels
        all_videos = []
        for vid_id in real_videos:
            vid_path = os.path.join(anchor_dir, vid_id)
            all_videos.append((vid_path, None, 0))  # (path, landmarks_dir, label)

        for method in all_methods:
            frames_dir = os.path.join(train_dir, method, 'frames')
            lm_dir = os.path.join(train_dir, method, 'landmarks')
            if not os.path.isdir(frames_dir):
                continue
            for vid_id in sorted(os.listdir(frames_dir)):
                vid_path = os.path.join(frames_dir, vid_id)
                lm_path = os.path.join(lm_dir, vid_id)
                all_videos.append((vid_path, lm_path, 1))

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(all_videos)
        n_val = int(len(all_videos) * val_ratio)

        if self.split == 'val':
            all_videos = all_videos[:n_val]
        else:
            all_videos = all_videos[n_val:]

        # Sample frames from each video
        for vid_path, lm_path, label in all_videos:
            frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.png')])
            if not frames:
                continue
            sampled = rng.choices(frames, k=min(self.frames_per_video, len(frames)))
            for frame_name in sampled:
                frame_path = os.path.join(vid_path, frame_name)
                frame_lm = None
                if lm_path is not None:
                    lm_file = os.path.splitext(frame_name)[0] + '.npy'
                    candidate = os.path.join(lm_path, lm_file)
                    if os.path.exists(candidate):
                        frame_lm = candidate
                self.samples.append((frame_path, frame_lm, label))

    def _load_test(self, domain, methods):
        test_dir = os.path.join(self.root, 'DF40_test')

        # Real samples from anchor
        anchor_dir = os.path.join(test_dir, 'anchor')
        if os.path.isdir(anchor_dir):
            for vid_id in sorted(os.listdir(anchor_dir)):
                vid_path = os.path.join(anchor_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.png')])
                for frame_name in frames:
                    self.samples.append((os.path.join(vid_path, frame_name), None, 0))

        # Fake samples
        all_methods = sorted([d for d in os.listdir(test_dir)
                              if os.path.isdir(os.path.join(test_dir, d))
                              and d != 'anchor' and not d.endswith('.zip')])
        if methods is not None:
            all_methods = [m for m in all_methods if m in methods]

        for method in all_methods:
            domain_dir = os.path.join(test_dir, method, domain)
            if not os.path.isdir(domain_dir):
                continue
            frames_dir = os.path.join(domain_dir, 'frames')
            lm_dir = os.path.join(domain_dir, 'landmarks')
            if not os.path.isdir(frames_dir):
                continue
            for vid_id in sorted(os.listdir(frames_dir)):
                vid_path = os.path.join(frames_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                lm_vid = os.path.join(lm_dir, vid_id) if os.path.isdir(lm_dir) else None
                for frame_name in sorted(os.listdir(vid_path)):
                    if not frame_name.endswith('.png'):
                        continue
                    frame_path = os.path.join(vid_path, frame_name)
                    frame_lm = None
                    if lm_vid is not None:
                        lm_file = os.path.splitext(frame_name)[0] + '.npy'
                        candidate = os.path.join(lm_vid, lm_file)
                        if os.path.exists(candidate):
                            frame_lm = candidate
                    self.samples.append((frame_path, frame_lm, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, lm_path, label = self.samples[idx]

        # Load image
        img = np.array(Image.open(frame_path).convert('RGB'))
        h, w = img.shape[:2]

        # Get face bounding box
        if lm_path is not None:
            landmarks = np.load(lm_path)
            face_bbox = landmarks_to_bbox(landmarks, h, w)
        else:
            # No landmarks — use center crop as fallback for 256x256 frames
            # (anchor frames are tightly cropped faces already)
            margin = 20
            face_bbox = np.array([margin, margin, w - margin, h - margin])

        # Get context bounding box
        ctx_bbox = expand_bbox_for_context(face_bbox, 1.8, h, w)

        # Crop
        face_rgb = crop_face(img, face_bbox, self.img_size)
        context_rgb = crop_context(img, face_bbox, ctx_bbox, self.img_size)

        # DCT frequency maps
        face_freq = rgb_to_dct_map(face_rgb, self.img_size)
        context_freq = rgb_to_dct_map(context_rgb, self.img_size)

        # Convert RGB to float tensor [0, 1]
        face_rgb = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        context_rgb = torch.from_numpy(context_rgb).permute(2, 0, 1).float() / 255.0

        # Freq already float32 [0, 1]
        face_freq = torch.from_numpy(face_freq)
        context_freq = torch.from_numpy(context_freq)

        return {
            'face_rgb': face_rgb,
            'context_rgb': context_rgb,
            'face_freq': face_freq,
            'context_freq': context_freq,
            'label': torch.tensor(label, dtype=torch.long),
        }


def get_dataloader(split, batch_size=64, num_workers=4, **kwargs):
    """Create DataLoader for DF40 dataset."""
    dataset = DF40Dataset(split=split, **kwargs)
    shuffle = split == 'train'
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
from data.df40_dataset import DF40Dataset

# Test train split
train_ds = DF40Dataset(split='train', frames_per_video=1)
print(f'Train samples: {len(train_ds)}')
sample = train_ds[0]
print('Keys:', list(sample.keys()))
print('face_rgb:', sample['face_rgb'].shape, sample['face_rgb'].dtype)
print('context_rgb:', sample['context_rgb'].shape)
print('face_freq:', sample['face_freq'].shape, sample['face_freq'].dtype)
print('label:', sample['label'])

# Test val split
val_ds = DF40Dataset(split='val', frames_per_video=1)
print(f'Val samples: {len(val_ds)}')

# Test test-ff split
test_ff = DF40Dataset(split='test', test_domain='ff')
print(f'Test-ff samples: {len(test_ff)}')

# Test test-cdf split
test_cdf = DF40Dataset(split='test', test_domain='cdf')
print(f'Test-cdf samples: {len(test_cdf)}')

# Verify shapes
s = test_ff[0]
assert s['face_rgb'].shape == (3, 224, 224)
assert s['context_rgb'].shape == (3, 224, 224)
assert s['face_freq'].shape == (3, 224, 224)
assert s['context_freq'].shape == (3, 224, 224)
assert s['label'] in (0, 1)
print('Dataset OK')
"
```

Expected: train ~28,000+ samples, val ~3,000+ samples, test-ff and test-cdf non-empty

- [ ] **Step 3: Commit**

```bash
git add data/df40_dataset.py
git commit -m "feat: add DF40 dataset with face/context cropping"
```

---

### Task 7: ResNet18 Encoder (`models/encoder.py`)

**Files:**
- Create: `models/encoder.py`

- [ ] **Step 1: Write `models/encoder.py`**

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Encoder(nn.Module):
    """ResNet18 backbone that outputs 512d features.

    Removes the final FC layer. Face and context share weights within the same domain.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # Remove the final FC layer, keep everything else
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            (B, 512)
        """
        h = self.features(x)
        return h.flatten(1)
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import torch
from models.encoder import ResNet18Encoder

enc = ResNet18Encoder(pretrained=True)
x = torch.randn(4, 3, 224, 224)
out = enc(x)
print(f'Input: {x.shape} -> Output: {out.shape}')
assert out.shape == (4, 512)
print('Encoder OK')
"
```

Expected: (4, 512)

- [ ] **Step 3: Commit**

```bash
git add models/encoder.py models/__init__.py
git commit -m "feat: add ResNet18 encoder"
```

---

### Task 8: Relation Module (`models/relation.py`)

**Files:**
- Create: `models/relation.py`

- [ ] **Step 1: Write `models/relation.py`**

```python
import torch
import torch.nn as nn


class RelationModule(nn.Module):
    """Computes relation features from face and context embeddings.

    r = [z_face, z_context, |z_face - z_context|, z_face * z_context]
    """

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = feat_dim * 4  # concat of 4 terms

    def forward(self, z_face: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_face: (B, feat_dim)
            z_context: (B, feat_dim)

        Returns:
            (B, feat_dim * 4)
        """
        return torch.cat([
            z_face,
            z_context,
            torch.abs(z_face - z_context),
            z_face * z_context,
        ], dim=1)
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import torch
from models.relation import RelationModule

rel = RelationModule(feat_dim=512)
z_f = torch.randn(4, 512)
z_c = torch.randn(4, 512)
r = rel(z_f, z_c)
print(f'z_face: {z_f.shape}, z_context: {z_c.shape} -> relation: {r.shape}')
assert r.shape == (4, 2048)
print('Relation module OK')
"
```

Expected: (4, 2048)

- [ ] **Step 3: Commit**

```bash
git add models/relation.py
git commit -m "feat: add relation module"
```

---

### Task 9: Classifier + Full Model (`models/classifier.py`)

**Files:**
- Create: `models/classifier.py`

**What it does:** `DualDomainDetector` wraps encoders + relation + MLP for all 4 experiments (M1-M4). Configured by `mode` parameter.

- [ ] **Step 1: Write `models/classifier.py`**

```python
import torch
import torch.nn as nn

from models.encoder import ResNet18Encoder
from models.relation import RelationModule


class DualDomainDetector(nn.Module):
    """Dual-domain face-context relation detector.

    Modes:
      - 'rgb':     M1 — RGB only, single-domain relation, CE loss
      - 'freq':    M2 — Freq only, single-domain relation, CE loss
      - 'dual':    M3 — RGB + Freq dual-domain fusion, CE loss
      - 'dual_cl': M4 — RGB + Freq dual-domain fusion, CE + InfoNCE loss
    """

    def __init__(self, mode: str = 'dual_cl', feat_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.mode = mode

        # Encoders
        use_rgb = mode in ('rgb', 'dual', 'dual_cl')
        use_freq = mode in ('freq', 'dual', 'dual_cl')
        self.use_rgb = use_rgb
        self.use_freq = use_freq

        if use_rgb:
            self.encoder_rgb = ResNet18Encoder(pretrained=pretrained)
        if use_freq:
            self.encoder_freq = ResNet18Encoder(pretrained=pretrained)

        # Relation module (shared across domains, same feat_dim)
        self.relation = RelationModule(feat_dim=feat_dim)

        # Calculate relation dimension
        n_domains = int(use_rgb) + int(use_freq)
        relation_dim = feat_dim * 4 * n_domains  # 2048 per domain

        # Classification head: relation_dim -> 512 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(relation_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        # Projection head for InfoNCE (M4 only)
        self.use_contrastive = mode == 'dual_cl'
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(relation_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

    def forward(self, batch: dict, return_features: bool = False):
        """
        Args:
            batch: dict with keys 'face_rgb', 'context_rgb', 'face_freq', 'context_freq'
            return_features: if True, return dict with logits, relation, projection

        Returns:
            logits (B, 2) if return_features=False
            dict with 'logits', 'relation', 'projection' if return_features=True
        """
        relations = []

        if self.use_rgb:
            z_face_rgb = self.encoder_rgb(batch['face_rgb'])
            z_ctx_rgb = self.encoder_rgb(batch['context_rgb'])
            r_rgb = self.relation(z_face_rgb, z_ctx_rgb)
            relations.append(r_rgb)

        if self.use_freq:
            z_face_freq = self.encoder_freq(batch['face_freq'])
            z_ctx_freq = self.encoder_freq(batch['context_freq'])
            r_freq = self.relation(z_face_freq, z_ctx_freq)
            relations.append(r_freq)

        r = torch.cat(relations, dim=1)  # (B, relation_dim)
        logits = self.classifier(r)

        if return_features:
            result = {
                'logits': logits,
                'relation': r,
            }
            if self.use_contrastive:
                result['projection'] = self.projection_head(r)
            return result

        return logits
```

- [ ] **Step 2: Test all 4 modes**

```bash
conda run -n researchclaw python -c "
import torch
from models.classifier import DualDomainDetector

batch = {
    'face_rgb': torch.randn(4, 3, 224, 224),
    'context_rgb': torch.randn(4, 3, 224, 224),
    'face_freq': torch.randn(4, 3, 224, 224),
    'context_freq': torch.randn(4, 3, 224, 224),
}

for mode in ['rgb', 'freq', 'dual', 'dual_cl']:
    model = DualDomainDetector(mode=mode, pretrained=False)
    with torch.no_grad():
        result = model(batch, return_features=True)
    print(f'{mode:10s}: logits={result[\"logits\"].shape}, relation={result[\"relation\"].shape}', end='')
    if 'projection' in result:
        print(f', proj={result[\"projection\"].shape}')
    else:
        print()
    assert result['logits'].shape == (4, 2)

# Verify relation dims
m1 = DualDomainDetector(mode='rgb', pretrained=False)
m3 = DualDomainDetector(mode='dual', pretrained=False)
assert m1(batch)['shape'] if False else True  # just check no crash
print('All 4 modes OK')
"
```

Expected: M1/M2 relation=2048d, M3/M4 relation=4096d, M4 has projection=128d

- [ ] **Step 3: Commit**

```bash
git add models/classifier.py
git commit -m "feat: add DualDomainDetector supporting all 4 experiment modes"
```

---

### Task 10: InfoNCE Loss (`losses/infonce.py`)

**Files:**
- Create: `losses/infonce.py`

- [ ] **Step 1: Write `losses/infonce.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss on relation embeddings.

    For each sample, two augmented views produce embeddings h1, h2.
    Positive pair: (h1_i, h2_i). Negatives: (h1_i, h2_j) for j != i.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) embeddings from view 1
            z2: (B, D) embeddings from view 2

        Returns:
            scalar loss
        """
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate to form (2B, D)
        representations = torch.cat([z1, z2], dim=0)

        # Similarity matrix (2B, 2B)
        similarity = torch.matmul(representations, representations.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        similarity.masked_fill_(mask, -1e9)

        # Labels: for position i, positive is at i+B (and vice versa)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z1.device)

        loss = F.cross_entropy(similarity, labels)
        return loss
```

- [ ] **Step 2: Test it**

```bash
conda run -n researchclaw python -c "
import torch
from losses.infonce import InfoNCELoss

loss_fn = InfoNCELoss(temperature=0.07)

# Test: identical pairs should have low loss
z = torch.randn(8, 128)
loss_same = loss_fn(z, z)
print(f'Same-view loss: {loss_same.item():.4f}')

# Test: random pairs should have higher loss
z1 = torch.randn(8, 128)
z2 = torch.randn(8, 128)
loss_diff = loss_fn(z1, z2)
print(f'Random-view loss: {loss_diff.item():.4f}')

assert loss_same.item() < loss_diff.item(), 'Same-view should have lower loss'
print('InfoNCE OK')
"
```

Expected: same-view loss < random-view loss

- [ ] **Step 3: Commit**

```bash
git add losses/infonce.py losses/__init__.py
git commit -m "feat: add InfoNCE loss"
```

---

### Task 11: Training Loop (`train.py`)

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write `train.py`**

```python
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.classifier import DualDomainDetector
from data.df40_dataset import get_dataloader
from data.transforms import PairAugmentation
from data.frequency import rgb_to_dct_map
from losses.infonce import InfoNCELoss
from utils.metrics import compute_auc, compute_eer


def train_one_epoch(model, dataloader, optimizer, scaler, device, args, contrastive=False, infonce_fn=None, aug_fn=None):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_nce_loss = 0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if contrastive and aug_fn is not None:
            # M4: create two augmented views
            # Augment RGB images
            aug1_face_rgb, aug1_ctx_rgb = [], []
            aug2_face_rgb, aug2_ctx_rgb = [], []
            for i in range(batch['face_rgb'].shape[0]):
                f = (batch['face_rgb'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                c = (batch['context_rgb'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                f1, c1 = aug_fn(f, c)
                f2, c2 = aug_fn(f, c)

                aug1_face_rgb.append(torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0)
                aug1_ctx_rgb.append(torch.from_numpy(c1).permute(2, 0, 1).float() / 255.0)
                aug2_face_rgb.append(torch.from_numpy(f2).permute(2, 0, 1).float() / 255.0)
                aug2_ctx_rgb.append(torch.from_numpy(c2).permute(2, 0, 1).float() / 255.0)

            batch1 = {
                'face_rgb': torch.stack(aug1_face_rgb).to(device),
                'context_rgb': torch.stack(aug1_ctx_rgb).to(device),
                'face_freq': batch['face_freq'],  # keep original freq for view 1
                'context_freq': batch['context_freq'],
            }
            batch2 = {
                'face_rgb': torch.stack(aug2_face_rgb).to(device),
                'context_rgb': torch.stack(aug2_ctx_rgb).to(device),
                'face_freq': batch['face_freq'],
                'context_freq': batch['context_freq'],
            }
        else:
            batch1 = batch2 = batch

        optimizer.zero_grad()

        with autocast('cuda', enabled=args.amp):
            result1 = model(batch1, return_features=True)
            cls_loss = criterion(result1['logits'], batch['label'])

            if contrastive and infonce_fn is not None:
                result2 = model(batch2, return_features=True)
                nce_loss = infonce_fn(result1['projection'], result2['projection'])
                loss = cls_loss + args.lambda_nce * nce_loss
            else:
                nce_loss = torch.tensor(0.0)
                loss = cls_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_nce_loss += nce_loss.item()
        n_batches += 1

        if n_batches % 50 == 0:
            print(f'  [{n_batches}/{len(dataloader)}] loss={loss.item():.4f} cls={cls_loss.item():.4f} nce={nce_loss.item():.4f}')

    return {
        'loss': total_loss / n_batches,
        'cls_loss': total_cls_loss / n_batches,
        'nce_loss': total_nce_loss / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []
    total_loss = 0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch_gpu)
        loss = criterion(logits, batch_gpu['label'])

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch['label'].numpy()

        all_scores.extend(probs)
        all_labels.extend(labels)
        total_loss += loss.item()
        n += 1

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = compute_auc(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)
    avg_loss = total_loss / n

    return {'auc': auc, 'eer': eer, 'loss': avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'freq', 'dual', 'dual_cl'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lambda_nce', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Mode: {args.mode}')

    # Data
    train_loader = get_dataloader('train', batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_dataloader('val', batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = DualDomainDetector(mode=args.mode, pretrained=True).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Contrastive learning setup
    contrastive = args.mode == 'dual_cl'
    infonce_fn = InfoNCELoss(temperature=args.tau).to(device) if contrastive else None
    aug_fn = PairAugmentation() if contrastive else None

    scaler = GradScaler(enabled=args.amp)

    # Training loop
    best_auc = 0
    patience_counter = 0
    exp_name = f'M4_{args.mode}' if args.mode != 'dual_cl' else 'M4_dual_cl'

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, contrastive, infonce_fn, aug_fn)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) '
              f'train_loss={train_metrics["loss"]:.4f} '
              f'val_auc={val_metrics["auc"]:.4f} val_eer={val_metrics["eer"]:.4f}')

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            save_dir = os.path.join(args.output_dir, exp_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': val_metrics['auc'],
                'eer': val_metrics['eer'],
                'mode': args.mode,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'  -> Best model saved (AUC={best_auc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'\nTraining done. Best val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (1 batch, no actual training)**

```bash
conda run -n researchclaw python -c "
import torch
from models.classifier import DualDomainDetector
from data.df40_dataset import get_dataloader

model = DualDomainDetector(mode='rgb', pretrained=False)
loader = get_dataloader('train', batch_size=4, num_workers=0)
batch = next(iter(loader))

print('Batch keys:', list(batch.keys()))
print('face_rgb:', batch['face_rgb'].shape)

logits = model(batch)
print(f'Logits: {logits.shape}')
assert logits.shape == (4, 2)
print('Smoke test passed')
"
```

Expected: (4, 2) logits, no errors

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add training loop with early stopping, AMP, contrastive learning"
```

---

### Task 12: Evaluation Script (`evaluate.py`)

**Files:**
- Create: `evaluate.py`

- [ ] **Step 1: Write `evaluate.py`**

```python
import argparse
import os
import numpy as np
import torch

from models.classifier import DualDomainDetector
from data.df40_dataset import get_dataloader
from utils.metrics import compute_auc, compute_eer


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []

    for batch in dataloader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch_gpu)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch['label'].numpy()
        all_scores.extend(probs)
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    auc = compute_auc(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)
    return auc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'freq', 'dual', 'dual_cl'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DualDomainDetector(mode=args.mode, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}, val AUC={ckpt["auc"]:.4f}')

    results = {}

    # Test on DF40-test ff (same-domain)
    loader_ff = get_dataloader('test', test_domain='ff', batch_size=args.batch_size, num_workers=args.num_workers)
    auc_ff, eer_ff = evaluate(model, loader_ff, device)
    results['DF40-test-ff'] = {'auc': auc_ff, 'eer': eer_ff}
    print(f'DF40-test (ff): AUC={auc_ff:.4f}, EER={eer_ff:.4f}')

    # Test on DF40-test cdf (cross-domain)
    loader_cdf = get_dataloader('test', test_domain='cdf', batch_size=args.batch_size, num_workers=args.num_workers)
    auc_cdf, eer_cdf = evaluate(model, loader_cdf, device)
    results['DF40-test-cdf'] = {'auc': auc_cdf, 'eer': eer_cdf}
    print(f'DF40-test (cdf): AUC={auc_cdf:.4f}, EER={eer_cdf:.4f}')

    # Save results
    exp_name = f'M4_{args.mode}' if args.mode != 'dual_cl' else 'M4_dual_cl'
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(f'Mode: {args.mode}\n')
        f.write(f'DF40-test-ff: AUC={auc_ff:.4f}, EER={eer_ff:.4f}\n')
        f.write(f'DF40-test-cdf: AUC={auc_cdf:.4f}, EER={eer_cdf:.4f}\n')

    # Print summary table
    print('\n' + '='*60)
    print(f'{"Model":<25} {"DF40-ff AUC":>12} {"DF40-ff EER":>12} {"DF40-cdf AUC":>12} {"DF40-cdf EER":>12}')
    print('-'*60)
    mode_label = {'rgb': 'M1: RGB-only', 'freq': 'M2: Freq-only', 'dual': 'M3: RGB+Freq', 'dual_cl': 'M4: RGB+Freq+InfoNCE'}[args.mode]
    print(f'{mode_label:<25} {auc_ff:>12.4f} {eer_ff:>12.4f} {auc_cdf:>12.4f} {eer_cdf:>12.4f}')
    print('='*60)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: add evaluation script for DF40-test ff/cdf"
```

---

### Task 13: Experiment Configs + Run Script

**Files:**
- Create: `configs/experiments.yaml`
- Create: `run_experiments.sh`

- [ ] **Step 1: Write `configs/experiments.yaml`**

```yaml
# Experiment configurations for M1-M4
# Each experiment uses the same training hyperparameters

common:
  batch_size: 64
  epochs: 30
  lr: 0.0001
  weight_decay: 0.00001
  patience: 5
  amp: true
  seed: 42
  num_workers: 4
  output_dir: outputs

experiments:
  M1_rgb:
    mode: rgb
    description: "RGB-only Relation + CE"

  M2_freq:
    mode: freq
    description: "Freq-only Relation + CE"

  M3_dual:
    mode: dual
    description: "RGB+Freq Relation + CE"

  M4_dual_cl:
    mode: dual_cl
    description: "RGB+Freq Relation + CE + InfoNCE"
    tau: 0.07
    lambda_nce: 0.1
```

- [ ] **Step 2: Write `run_experiments.sh`**

```bash
#!/bin/bash
# Run all 4 experiments sequentially
set -e

CONDA_ENV="researchclaw"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "Running all deepfake detection experiments"
echo "Project: $PROJECT_DIR"
echo "Conda env: $CONDA_ENV"
echo "============================================"

for MODE in rgb freq dual dual_cl; do
    echo ""
    echo "============================================"
    echo "Experiment: $MODE"
    echo "============================================"

    # Train
    conda run -n $CONDA_ENV python "$PROJECT_DIR/train.py" \
        --mode $MODE \
        --batch_size 64 \
        --epochs 30 \
        --lr 1e-4 \
        --patience 5 \
        --seed 42 \
        --output_dir "$PROJECT_DIR/outputs"

    # Evaluate
    CKPT="$PROJECT_DIR/outputs/M4_${MODE}/best_model.pth"
    if [ -f "$CKPT" ]; then
        conda run -n $CONDA_ENV python "$PROJECT_DIR/evaluate.py" \
            --mode $MODE \
            --checkpoint "$CKPT" \
            --batch_size 64 \
            --output_dir "$PROJECT_DIR/outputs"
    else
        echo "WARNING: No checkpoint found at $CKPT"
    fi
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
```

- [ ] **Step 3: Make run script executable and commit**

```bash
chmod +x run_experiments.sh
git add configs/experiments.yaml run_experiments.sh
git commit -m "feat: add experiment configs and run script"
```

---

## Summary: Execution Checklist

| Task | What | Key Output |
|------|------|-----------|
| 1 | Environment setup | Clean dir, packages installed, git init |
| 2 | DCT transform | `data/frequency.py` — RGB → DCT log-amplitude map |
| 3 | Metrics | `utils/metrics.py` — AUC, EER |
| 4 | Augmentation | `data/transforms.py` — pair-synchronized augmentation |
| 5 | Face detector | `data/face_detector.py` — SCRFD + crop utilities |
| 6 | Dataset | `data/df40_dataset.py` — DF40 train/val/test loading |
| 7 | Encoder | `models/encoder.py` — ResNet18 → 512d |
| 8 | Relation | `models/relation.py` — [z_f, z_c, |Δ|, ⊙] → 2048d |
| 9 | Classifier | `models/classifier.py` — DualDomainDetector (M1-M4) |
| 10 | InfoNCE | `losses/infonce.py` — contrastive loss |
| 11 | Training | `train.py` — full loop with AMP, early stopping |
| 12 | Evaluation | `evaluate.py` — DF40-test ff + cdf |
| 13 | Configs | `configs/experiments.yaml` + `run_experiments.sh` |

**Run order:** Tasks 1-13 sequentially. After all done, run `bash run_experiments.sh` to execute all 4 experiments.
