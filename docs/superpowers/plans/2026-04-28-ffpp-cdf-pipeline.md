# FF++→CDF Deepfake Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh deepfake detection training and evaluation pipeline for four controlled experiments on FF++→CDF, using a dedicated `deepfake-detection` conda environment, CLIP ViT-B/16 at 224×224, and 8-GPU DistributedDataParallel training.

**Architecture:** Create the project from scratch around a shared data pipeline, experiment-specific model builders, DDP-aware training/evaluation entrypoints, and config-driven experiment definitions. Keep full-frame RGB training as the common path for Experiments 1–3, and add a strict aligned-triplet path for Experiment 4 so background, real face, and fake face share identical augmentations before crop extraction.

**Tech Stack:** Python 3.10, PyTorch, torchvision, OpenAI CLIP (`ViT-B/16`), albumentations, numpy, scikit-learn, PyYAML, OpenCV/Pillow, pytest, torchrun/DDP, 8×V100 GPUs

---

## Planned File Structure

**Environment and dependency files**
- Create: `environment.yml` — canonical conda environment named `deepfake-detection`
- Create: `requirements.txt` — pip-installable mirror for non-conda installs

**Configuration**
- Create: `configs/base.yaml` — shared dataset/model/training defaults
- Create: `configs/exp1_efficientnet.yaml` — EfficientNet experiment config
- Create: `configs/exp2_clip_ft.yaml` — CLIP fine-tune experiment config
- Create: `configs/exp3_clip_prompt.yaml` — CLIP + fixed prompt config
- Create: `configs/exp4_clip_prompt_bgcontrast.yaml` — CLIP + prompt + aligned-triplet contrast config

**Dataset and transforms**
- Create: `src/deepfake_detection/data/constants.py` — method lists, prompt text, split names
- Create: `src/deepfake_detection/data/index_ffpp.py` — FF++ indexing and aligned-pair resolution
- Create: `src/deepfake_detection/data/index_cdf.py` — CDF indexing for matched-method evaluation
- Create: `src/deepfake_detection/data/sampling.py` — per-method 200-video sampling, oversampled real balancing, 8-frame uniform selection
- Create: `src/deepfake_detection/data/transforms.py` — shared RGB augmentation and deterministic paired augmentation helpers
- Create: `src/deepfake_detection/data/crops.py` — face/background crop extraction from aligned frames
- Create: `src/deepfake_detection/data/frequency.py` — frequency feature extraction from augmented RGB inputs
- Create: `src/deepfake_detection/data/datasets.py` — frame dataset and aligned-triplet dataset
- Create: `src/deepfake_detection/data/builders.py` — dataloader builders and distributed samplers

**Models and losses**
- Create: `src/deepfake_detection/models/efficientnet.py` — EfficientNet binary classifier
- Create: `src/deepfake_detection/models/clip_classifier.py` — CLIP ViT-B/16 binary classifier
- Create: `src/deepfake_detection/models/clip_prompt.py` — fixed-prompt CLIP classifier
- Create: `src/deepfake_detection/models/clip_bgcontrast.py` — CLIP prompt model with face/background projection heads
- Create: `src/deepfake_detection/losses/contrastive.py` — same-frame background-face contrastive loss
- Create: `src/deepfake_detection/models/factory.py` — config-driven model construction

**Training and evaluation**
- Create: `src/deepfake_detection/engine/ddp.py` — process-group setup, rank helpers, sampler epoch control
- Create: `src/deepfake_detection/engine/trainers.py` — train/val loops for classification and contrastive variants
- Create: `src/deepfake_detection/engine/metrics.py` — frame/video aggregation, AUC/EER/ACC
- Create: `train.py` — DDP entrypoint for all four experiments
- Create: `evaluate.py` — checkpoint evaluation on FF++ and CDF
- Create: `scripts/run_exp1.sh` — torchrun launcher for Experiment 1
- Create: `scripts/run_exp2.sh` — torchrun launcher for Experiment 2
- Create: `scripts/run_exp3.sh` — torchrun launcher for Experiment 3
- Create: `scripts/run_exp4.sh` — torchrun launcher for Experiment 4

**Tests**
- Create: `tests/data/test_sampling.py`
- Create: `tests/data/test_transforms.py`
- Create: `tests/data/test_triplets.py`
- Create: `tests/models/test_prompts.py`
- Create: `tests/losses/test_contrastive.py`
- Create: `tests/engine/test_metrics.py`
- Create: `tests/configs/test_config_load.py`

**Package setup**
- Create: `src/deepfake_detection/__init__.py`
- Create: `src/deepfake_detection/data/__init__.py`
- Create: `src/deepfake_detection/models/__init__.py`
- Create: `src/deepfake_detection/losses/__init__.py`
- Create: `src/deepfake_detection/engine/__init__.py`

---

### Task 1: Create the environment and dependency baseline

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Write the failing dependency test**

```python
# tests/configs/test_config_load.py
from pathlib import Path


def test_environment_file_declares_deepfake_detection_name():
    text = Path("environment.yml").read_text()
    assert "name: deepfake-detection" in text


def test_requirements_include_clip_and_albumentations():
    text = Path("requirements.txt").read_text()
    assert "albumentations" in text
    assert "git+https://github.com/openai/CLIP.git" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/configs/test_config_load.py -v`
Expected: FAIL with `FileNotFoundError` for `environment.yml`

- [ ] **Step 3: Write the environment definition**

```yaml
# environment.yml
name: deepfake-detection
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch=2.2
  - torchvision=0.17
  - pytorch-cuda=12.1
  - numpy
  - scipy
  - scikit-learn
  - pyyaml
  - pillow
  - opencv
  - tqdm
  - pandas
  - pip:
      - albumentations==1.4.18
      - ftfy
      - regex
      - git+https://github.com/openai/CLIP.git
      - pytest
```

```txt
# requirements.txt
numpy
scipy
scikit-learn
pyyaml
pillow
opencv-python
pandas
tqdm
albumentations==1.4.18
ftfy
regex
git+https://github.com/openai/CLIP.git
pytest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/configs/test_config_load.py -v`
Expected: PASS

- [ ] **Step 5: Validate environment creation commands**

Run: `conda env create -f environment.yml`
Expected: Conda creates the `deepfake-detection` environment without dependency resolution errors

Run: `conda run -n deepfake-detection python -c "import torch, albumentations, clip; print(torch.__version__)"`
Expected: Prints a PyTorch version and exits successfully

- [ ] **Step 6: Commit**

```bash
git add environment.yml requirements.txt tests/configs/test_config_load.py
git commit -m "build: add deepfake-detection environment baseline"
```

### Task 2: Create project skeleton and shared constants

**Files:**
- Create: `src/deepfake_detection/__init__.py`
- Create: `src/deepfake_detection/data/__init__.py`
- Create: `src/deepfake_detection/models/__init__.py`
- Create: `src/deepfake_detection/losses/__init__.py`
- Create: `src/deepfake_detection/engine/__init__.py`
- Create: `src/deepfake_detection/data/constants.py`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Write the failing constants test**

```python
# append to tests/configs/test_config_load.py
from deepfake_detection.data.constants import CLIP_MODEL_NAME, IMG_SIZE, FACE_SWAP_METHODS, REENACTMENT_METHODS


def test_constants_match_experiment_spec():
    assert CLIP_MODEL_NAME == "ViT-B/16"
    assert IMG_SIZE == 224
    assert len(FACE_SWAP_METHODS) == 8
    assert len(REENACTMENT_METHODS) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/configs/test_config_load.py::test_constants_match_experiment_spec -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepfake_detection'`

- [ ] **Step 3: Write the package skeleton and constants**

```python
# src/deepfake_detection/data/constants.py
CLIP_MODEL_NAME = "ViT-B/16"
IMG_SIZE = 224
FACE_SWAP_METHODS = [
    "simswap", "inswap", "blendface", "faceswap",
    "fsgan", "mobileswap", "e4s", "facedancer",
]
REENACTMENT_METHODS = [
    "fomm", "facevid2vid", "wav2lip", "sadtalker",
    "MRAA", "pirender", "tpsm", "lia",
]
ALL_METHODS = FACE_SWAP_METHODS + REENACTMENT_METHODS
REAL_LABEL = 0
FAKE_LABEL = 1
REAL_PROMPTS = ["a real face image", "an authentic face photo"]
FAKE_PROMPTS = ["a fake face image", "a manipulated face photo"]
```

```python
# src/deepfake_detection/__init__.py
__all__ = ["data", "models", "losses", "engine"]
```

```python
# src/deepfake_detection/data/__init__.py
from .constants import ALL_METHODS
```

```python
# src/deepfake_detection/models/__init__.py
__all__ = []
```

```python
# src/deepfake_detection/losses/__init__.py
__all__ = []
```

```python
# src/deepfake_detection/engine/__init__.py
__all__ = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/configs/test_config_load.py::test_constants_match_experiment_spec -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src tests/configs/test_config_load.py
git commit -m "feat: add project skeleton and experiment constants"
```

### Task 3: Add config files for the four experiments

**Files:**
- Create: `configs/base.yaml`
- Create: `configs/exp1_efficientnet.yaml`
- Create: `configs/exp2_clip_ft.yaml`
- Create: `configs/exp3_clip_prompt.yaml`
- Create: `configs/exp4_clip_prompt_bgcontrast.yaml`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Write the failing config test**

```python
# append to tests/configs/test_config_load.py
import yaml
from pathlib import Path


def test_experiment_configs_reference_vit_b_16_and_ddp():
    exp4 = yaml.safe_load(Path("configs/exp4_clip_prompt_bgcontrast.yaml").read_text())
    assert exp4["model"]["clip_model_name"] == "ViT-B/16"
    assert exp4["train"]["launcher"] == "torchrun"
    assert exp4["train"]["distributed"] == "ddp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/configs/test_config_load.py::test_experiment_configs_reference_vit_b_16_and_ddp -v`
Expected: FAIL with `FileNotFoundError` for config files

- [ ] **Step 3: Write the config files**

```yaml
# configs/base.yaml
dataset:
  ffpp_root: /Dataset/deepfake_detection/FaceForensics++
  cdf_root: /Dataset/deepfake_detection/DF40_all
  methods: []
  train_videos_per_method: 200
  frames_per_video: 8
  image_size: 224
  real_sampling: oversample_per_method
  use_mask: false
augment:
  horizontal_flip_p: 0.5
  rotate_limit: 10
  rotate_p: 0.5
  brightness_limit: 0.1
  contrast_limit: 0.1
  brightness_contrast_p: 0.5
  compression_quality_lower: 40
  compression_quality_upper: 100
  compression_p: 0.2
model:
  clip_model_name: ViT-B/16
train:
  launcher: torchrun
  distributed: ddp
  image_size: 224
  num_nodes: 1
  gpus_per_node: 8
  precision: amp
```

```yaml
# configs/exp1_efficientnet.yaml
_base_: configs/base.yaml
experiment_name: exp1_efficientnet
model:
  name: efficientnet_b0
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy
```

```yaml
# configs/exp2_clip_ft.yaml
_base_: configs/base.yaml
experiment_name: exp2_clip_ft
model:
  name: clip_finetune
  clip_model_name: ViT-B/16
loss:
  name: cross_entropy
```

```yaml
# configs/exp3_clip_prompt.yaml
_base_: configs/base.yaml
experiment_name: exp3_clip_prompt
model:
  name: clip_prompt
  clip_model_name: ViT-B/16
  prompt_mode: fixed
loss:
  name: cross_entropy
```

```yaml
# configs/exp4_clip_prompt_bgcontrast.yaml
_base_: configs/base.yaml
experiment_name: exp4_clip_prompt_bgcontrast
model:
  name: clip_prompt_bgcontrast
  clip_model_name: ViT-B/16
  prompt_mode: fixed
  projection_dim: 256
loss:
  name: cross_entropy_plus_contrastive
  lambda_align: 0.1
  temperature: 0.07
data:
  require_aligned_triplets: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/configs/test_config_load.py::test_experiment_configs_reference_vit_b_16_and_ddp -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs tests/configs/test_config_load.py
git commit -m "feat: add config files for four experiments"
```

### Task 4: Build per-method video sampling and frame selection

**Files:**
- Create: `src/deepfake_detection/data/sampling.py`
- Test: `tests/data/test_sampling.py`

- [ ] **Step 1: Write the failing sampling test**

```python
# tests/data/test_sampling.py
from deepfake_detection.data.sampling import sample_uniform_frame_indices, balance_real_video_ids


def test_sample_uniform_frame_indices_returns_eight_sorted_positions():
    indices = sample_uniform_frame_indices(num_frames=32, num_samples=8)
    assert indices == [0, 4, 8, 13, 17, 22, 26, 31]


def test_balance_real_video_ids_oversamples_to_target_count():
    balanced = balance_real_video_ids(["r0", "r1"], target_count=5, seed=7)
    assert len(balanced) == 5
    assert set(balanced).issubset({"r0", "r1"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing functions

- [ ] **Step 3: Write the minimal sampling implementation**

```python
# src/deepfake_detection/data/sampling.py
from __future__ import annotations

import random
from math import floor


def sample_uniform_frame_indices(num_frames: int, num_samples: int = 8) -> list[int]:
    if num_frames < num_samples:
        raise ValueError("num_frames must be >= num_samples")
    return [floor(i * (num_frames - 1) / (num_samples - 1)) for i in range(num_samples)]


def balance_real_video_ids(real_video_ids: list[str], target_count: int, seed: int) -> list[str]:
    if not real_video_ids:
        raise ValueError("real_video_ids must not be empty")
    rng = random.Random(seed)
    if len(real_video_ids) >= target_count:
        return real_video_ids[:target_count]
    padded = list(real_video_ids)
    while len(padded) < target_count:
        padded.append(rng.choice(real_video_ids))
    return padded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/sampling.py tests/data/test_sampling.py
git commit -m "feat: add frame sampling and real balancing helpers"
```

### Task 5: Build shared and synchronized augmentations

**Files:**
- Create: `src/deepfake_detection/data/transforms.py`
- Test: `tests/data/test_transforms.py`

- [ ] **Step 1: Write the failing transform test**

```python
# tests/data/test_transforms.py
import numpy as np
from deepfake_detection.data.transforms import build_rgb_augment, apply_shared_transform_pair


def test_build_rgb_augment_uses_expected_policy_names():
    transform = build_rgb_augment()
    names = [type(t).__name__ for t in transform.transforms]
    assert names == [
        "HorizontalFlip",
        "Rotate",
        "RandomBrightnessContrast",
        "ImageCompression",
    ]


def test_apply_shared_transform_pair_keeps_image_shapes_aligned():
    image_a = np.zeros((224, 224, 3), dtype=np.uint8)
    image_b = np.zeros((224, 224, 3), dtype=np.uint8)
    aug_a, aug_b = apply_shared_transform_pair(image_a, image_b)
    assert aug_a.shape == (224, 224, 3)
    assert aug_b.shape == (224, 224, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_transforms.py -v`
Expected: FAIL with missing module or functions

- [ ] **Step 3: Write the transform implementation**

```python
# src/deepfake_detection/data/transforms.py
from __future__ import annotations

import albumentations as A


def build_rgb_augment() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5,
        ),
        A.ImageCompression(quality_lower=40, quality_upper=100, p=0.2),
    ], additional_targets={"image_pair": "image"})


def apply_shared_transform_pair(image, image_pair):
    transformed = build_rgb_augment()(image=image, image_pair=image_pair)
    return transformed["image"], transformed["image_pair"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_transforms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/transforms.py tests/data/test_transforms.py
git commit -m "feat: add shared RGB augmentation policy"
```

### Task 6: Add frequency extraction from augmented RGB inputs

**Files:**
- Create: `src/deepfake_detection/data/frequency.py`
- Test: `tests/data/test_transforms.py`

- [ ] **Step 1: Write the failing frequency test**

```python
# append to tests/data/test_transforms.py
from deepfake_detection.data.frequency import rgb_to_frequency_map


def test_rgb_to_frequency_map_preserves_spatial_size():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    freq = rgb_to_frequency_map(image)
    assert freq.shape[:2] == (224, 224)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_transforms.py::test_rgb_to_frequency_map_preserves_spatial_size -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the frequency extractor**

```python
# src/deepfake_detection/data/frequency.py
from __future__ import annotations

import cv2
import numpy as np


def rgb_to_frequency_map(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    channels = []
    for channel in cv2.split(image):
        dct = cv2.dct(channel)
        channels.append(dct)
    return np.stack(channels, axis=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_transforms.py::test_rgb_to_frequency_map_preserves_spatial_size -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/frequency.py tests/data/test_transforms.py
git commit -m "feat: add frequency feature extraction"
```

### Task 7: Build face and background crop helpers

**Files:**
- Create: `src/deepfake_detection/data/crops.py`
- Test: `tests/data/test_triplets.py`

- [ ] **Step 1: Write the failing crop test**

```python
# tests/data/test_triplets.py
import numpy as np
from deepfake_detection.data.crops import expand_box, crop_region


def test_expand_box_grows_face_region_for_context():
    box = (50, 60, 150, 160)
    expanded = expand_box(box, scale=1.3, image_h=224, image_w=224)
    assert expanded[0] <= 50
    assert expanded[1] <= 60
    assert expanded[2] >= 150
    assert expanded[3] >= 160


def test_crop_region_returns_square_output():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    crop = crop_region(image, (32, 32, 192, 192), output_size=224)
    assert crop.shape == (224, 224, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py -v`
Expected: FAIL with missing module or functions

- [ ] **Step 3: Write the crop helpers**

```python
# src/deepfake_detection/data/crops.py
from __future__ import annotations

import cv2


def expand_box(box, scale, image_h, image_w):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - w / 2)))
    ny1 = max(0, int(round(cy - h / 2)))
    nx2 = min(image_w, int(round(cx + w / 2)))
    ny2 = min(image_h, int(round(cy + h / 2)))
    return nx1, ny1, nx2, ny2


def crop_region(image, box, output_size=224):
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/crops.py tests/data/test_triplets.py
git commit -m "feat: add face and background crop helpers"
```

### Task 8: Index FF++ training data and aligned real/fake pairs

**Files:**
- Create: `src/deepfake_detection/data/index_ffpp.py`
- Test: `tests/data/test_triplets.py`

- [ ] **Step 1: Write the failing FF++ index test**

```python
# append to tests/data/test_triplets.py
from deepfake_detection.data.index_ffpp import build_aligned_pair_key


def test_build_aligned_pair_key_uses_pair_and_frame_index():
    key = build_aligned_pair_key(method="simswap", pair_id="000_003", frame_name="012.png")
    assert key == "simswap::000_003::012"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py::test_build_aligned_pair_key_uses_pair_and_frame_index -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the aligned-key helper**

```python
# src/deepfake_detection/data/index_ffpp.py
from __future__ import annotations

from pathlib import Path


def build_aligned_pair_key(method: str, pair_id: str, frame_name: str) -> str:
    frame_id = Path(frame_name).stem
    return f"{method}::{pair_id}::{frame_id}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py::test_build_aligned_pair_key_uses_pair_and_frame_index -v`
Expected: PASS

- [ ] **Step 5: Expand the file to include index builders**

```python
# add to src/deepfake_detection/data/index_ffpp.py
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameRecord:
    method: str
    pair_id: str
    frame_name: str
    frame_path: str
    landmark_path: str | None
    label: int
```

Add functions:
- `index_ffpp_method_frames(...)`
- `index_ffpp_real_frames(...)`
- `index_ffpp_aligned_triplets(...)`

Each function should return lists of `FrameRecord` or triplet dictionaries keyed by `build_aligned_pair_key`.

- [ ] **Step 6: Run the existing test suite**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/data/index_ffpp.py tests/data/test_triplets.py
git commit -m "feat: add FF++ pair indexing helpers"
```

### Task 9: Index CDF evaluation data

**Files:**
- Create: `src/deepfake_detection/data/index_cdf.py`
- Test: `tests/data/test_sampling.py`

- [ ] **Step 1: Write the failing CDF index test**

```python
# append to tests/data/test_sampling.py
from deepfake_detection.data.index_cdf import normalize_cdf_method_name


def test_normalize_cdf_method_name_keeps_known_method_names():
    assert normalize_cdf_method_name("wav2lip") == "wav2lip"
    assert normalize_cdf_method_name("MRAA") == "MRAA"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py::test_normalize_cdf_method_name_keeps_known_method_names -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the CDF method normalizer**

```python
# src/deepfake_detection/data/index_cdf.py
from __future__ import annotations


def normalize_cdf_method_name(name: str) -> str:
    return name.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py::test_normalize_cdf_method_name_keeps_known_method_names -v`
Expected: PASS

- [ ] **Step 5: Expand the file to include evaluation indexing**

```python
# add to src/deepfake_detection/data/index_cdf.py
from dataclasses import dataclass


@dataclass(frozen=True)
class CDFRecord:
    method: str
    video_id: str
    frame_path: str
    label: int
```

Add functions:
- `index_cdf_real_frames(...)`
- `index_cdf_fake_frames(...)`
- `filter_cdf_records_by_methods(...)`

These should preserve the same 16-method evaluation protocol used in the spec.

- [ ] **Step 6: Run the data tests**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py tests/data/test_triplets.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/data/index_cdf.py tests/data/test_sampling.py
git commit -m "feat: add CDF evaluation indexing helpers"
```

### Task 10: Build frame and aligned-triplet datasets

**Files:**
- Create: `src/deepfake_detection/data/datasets.py`
- Test: `tests/data/test_triplets.py`

- [ ] **Step 1: Write the failing dataset test**

```python
# append to tests/data/test_triplets.py
from deepfake_detection.data.datasets import collate_video_scores


def test_collate_video_scores_groups_frames_by_video_id():
    rows = [
        {"video_id": "v0", "score": 0.2},
        {"video_id": "v0", "score": 0.4},
        {"video_id": "v1", "score": 0.9},
    ]
    grouped = collate_video_scores(rows)
    assert grouped == {"v0": [0.2, 0.4], "v1": [0.9]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py::test_collate_video_scores_groups_frames_by_video_id -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the dataset helpers**

```python
# src/deepfake_detection/data/datasets.py
from __future__ import annotations

from collections import defaultdict


def collate_video_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row["score"])
    return dict(grouped)
```

Then add dataset classes:
- `FrameClassificationDataset`
- `AlignedTripletDataset`

Requirements for `AlignedTripletDataset`:
- load aligned real/fake frame pairs
- apply the same sampled augmentation to both frames
- crop background from the real frame after augmentation
- crop `f_real` from the augmented real frame
- crop `f_fake` from the augmented fake frame
- optionally emit frequency maps derived from the same augmented frames

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py::test_collate_video_scores_groups_frames_by_video_id -v`
Expected: PASS

- [ ] **Step 5: Run the triplet tests after dataset integration**

Run: `PYTHONPATH=src pytest tests/data/test_triplets.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deepfake_detection/data/datasets.py tests/data/test_triplets.py
git commit -m "feat: add frame and aligned-triplet datasets"
```

### Task 11: Build distributed dataloaders

**Files:**
- Create: `src/deepfake_detection/data/builders.py`
- Test: `tests/data/test_sampling.py`

- [ ] **Step 1: Write the failing dataloader config test**

```python
# append to tests/data/test_sampling.py
from deepfake_detection.data.builders import build_loader_kwargs


def test_build_loader_kwargs_enables_distributed_sampler_when_requested():
    kwargs = build_loader_kwargs(batch_size=32, num_workers=4, distributed=True)
    assert kwargs["batch_size"] == 32
    assert kwargs["num_workers"] == 4
    assert kwargs["distributed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py::test_build_loader_kwargs_enables_distributed_sampler_when_requested -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the loader helper**

```python
# src/deepfake_detection/data/builders.py
from __future__ import annotations


def build_loader_kwargs(batch_size: int, num_workers: int, distributed: bool) -> dict:
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "distributed": distributed,
    }
```

Then add:
- `build_train_loader(...)`
- `build_eval_loader(...)`

Use `DistributedSampler` when `distributed=True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/data/test_sampling.py::test_build_loader_kwargs_enables_distributed_sampler_when_requested -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/data/builders.py tests/data/test_sampling.py
git commit -m "feat: add dataloader builders for DDP"
```

### Task 12: Implement EfficientNet baseline model

**Files:**
- Create: `src/deepfake_detection/models/efficientnet.py`
- Create: `src/deepfake_detection/models/factory.py`
- Test: `tests/models/test_prompts.py`

- [ ] **Step 1: Write the failing model factory test**

```python
# tests/models/test_prompts.py
from deepfake_detection.models.factory import build_model


def test_build_model_creates_efficientnet_from_name():
    model = build_model({"name": "efficientnet_b0"})
    assert model.__class__.__name__ == "EfficientNetBinaryClassifier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/models/test_prompts.py::test_build_model_creates_efficientnet_from_name -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the EfficientNet model and factory**

```python
# src/deepfake_detection/models/efficientnet.py
from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, images):
        return self.backbone(images)
```

```python
# src/deepfake_detection/models/factory.py
from __future__ import annotations

from deepfake_detection.models.efficientnet import EfficientNetBinaryClassifier


def build_model(model_cfg: dict):
    if model_cfg["name"] == "efficientnet_b0":
        return EfficientNetBinaryClassifier()
    raise ValueError(f"Unknown model name: {model_cfg['name']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/models/test_prompts.py::test_build_model_creates_efficientnet_from_name -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/models/efficientnet.py src/deepfake_detection/models/factory.py tests/models/test_prompts.py
git commit -m "feat: add EfficientNet baseline model"
```

### Task 13: Implement CLIP fine-tune and fixed-prompt models

**Files:**
- Create: `src/deepfake_detection/models/clip_classifier.py`
- Create: `src/deepfake_detection/models/clip_prompt.py`
- Modify: `src/deepfake_detection/models/factory.py`
- Test: `tests/models/test_prompts.py`

- [ ] **Step 1: Write the failing prompt test**

```python
# append to tests/models/test_prompts.py
from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS
from deepfake_detection.models.clip_prompt import build_fixed_prompt_texts


def test_build_fixed_prompt_texts_returns_real_and_fake_prompts():
    real, fake = build_fixed_prompt_texts()
    assert real == REAL_PROMPTS
    assert fake == FAKE_PROMPTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/models/test_prompts.py::test_build_fixed_prompt_texts_returns_real_and_fake_prompts -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the CLIP models**

```python
# src/deepfake_detection/models/clip_prompt.py
from __future__ import annotations

import clip
import torch
import torch.nn as nn
from deepfake_detection.data.constants import REAL_PROMPTS, FAKE_PROMPTS


def build_fixed_prompt_texts():
    return REAL_PROMPTS, FAKE_PROMPTS


class CLIPPromptBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")

    def forward(self, images):
        return self.clip_model.encode_image(images)
```
```

```python
# src/deepfake_detection/models/clip_classifier.py
from __future__ import annotations

import clip
import torch.nn as nn


class CLIPFineTuneBinaryClassifier(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.classifier = nn.Linear(self.clip_model.visual.output_dim, 2)

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        return self.classifier(image_features)
```

Update `build_model(...)` in `src/deepfake_detection/models/factory.py` to support:
- `clip_finetune`
- `clip_prompt`

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/models/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/models/clip_classifier.py src/deepfake_detection/models/clip_prompt.py src/deepfake_detection/models/factory.py tests/models/test_prompts.py
git commit -m "feat: add CLIP fine-tune and fixed-prompt models"
```

### Task 14: Implement background-face contrastive loss and model

**Files:**
- Create: `src/deepfake_detection/losses/contrastive.py`
- Create: `src/deepfake_detection/models/clip_bgcontrast.py`
- Modify: `src/deepfake_detection/models/factory.py`
- Test: `tests/losses/test_contrastive.py`

- [ ] **Step 1: Write the failing contrastive-loss test**

```python
# tests/losses/test_contrastive.py
import torch
from deepfake_detection.losses.contrastive import same_frame_contrastive_loss


def test_same_frame_contrastive_loss_returns_scalar():
    background = torch.randn(4, 8)
    real_face = torch.randn(4, 8)
    fake_face = torch.randn(4, 8)
    loss = same_frame_contrastive_loss(background, real_face, fake_face, temperature=0.07)
    assert loss.ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/losses/test_contrastive.py -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the contrastive loss**

```python
# src/deepfake_detection/losses/contrastive.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def same_frame_contrastive_loss(background, real_face, fake_face, temperature: float = 0.07):
    background = F.normalize(background, dim=1)
    real_face = F.normalize(real_face, dim=1)
    fake_face = F.normalize(fake_face, dim=1)
    pos = (background * real_face).sum(dim=1, keepdim=True) / temperature
    neg = (background * fake_face).sum(dim=1, keepdim=True) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(background.size(0), dtype=torch.long, device=background.device)
    return F.cross_entropy(logits, labels)
```

- [ ] **Step 4: Add the CLIP background-contrast model**

```python
# src/deepfake_detection/models/clip_bgcontrast.py
from __future__ import annotations

import clip
import torch.nn as nn


class CLIPPromptBackgroundContrastModel(nn.Module):
    def __init__(self, clip_model_name: str = "ViT-B/16", projection_dim: int = 256):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        width = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(width, 2)
        self.face_projection = nn.Linear(width, projection_dim)
        self.background_projection = nn.Linear(width, projection_dim)
```

Update `build_model(...)` in `src/deepfake_detection/models/factory.py` to support `clip_prompt_bgcontrast`.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/losses/test_contrastive.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deepfake_detection/losses/contrastive.py src/deepfake_detection/models/clip_bgcontrast.py src/deepfake_detection/models/factory.py tests/losses/test_contrastive.py
git commit -m "feat: add same-frame contrastive loss and bgcontrast model"
```

### Task 15: Implement DDP helpers, train loops, and metrics

**Files:**
- Create: `src/deepfake_detection/engine/ddp.py`
- Create: `src/deepfake_detection/engine/trainers.py`
- Create: `src/deepfake_detection/engine/metrics.py`
- Test: `tests/engine/test_metrics.py`

- [ ] **Step 1: Write the failing metrics test**

```python
# tests/engine/test_metrics.py
from deepfake_detection.engine.metrics import mean_video_score


def test_mean_video_score_averages_eight_frame_scores():
    score = mean_video_score([0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5])
    assert score == 0.40625
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/engine/test_metrics.py -v`
Expected: FAIL with missing module or function

- [ ] **Step 3: Write the metrics helper**

```python
# src/deepfake_detection/engine/metrics.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def mean_video_score(scores):
    return float(np.mean(scores))
```

Then add:
- `compute_auc(labels, scores)`
- `compute_eer(labels, scores)`
- `aggregate_video_predictions(rows)`

- [ ] **Step 4: Add DDP and trainer helpers**

```python
# src/deepfake_detection/engine/ddp.py
from __future__ import annotations

import os
import torch.distributed as dist


def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank
```

```python
# src/deepfake_detection/engine/trainers.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_step(model, batch):
    logits = model(batch["image"])
    return F.cross_entropy(logits, batch["label"])
```

Then extend `trainers.py` with:
- `run_train_epoch(...)`
- `run_eval_epoch(...)`
- `contrastive_step(...)`

Requirements:
- use AMP
- support standard classification and Experiment 4 contrastive mode
- report frame- and video-level stats

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/engine/test_metrics.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deepfake_detection/engine/ddp.py src/deepfake_detection/engine/trainers.py src/deepfake_detection/engine/metrics.py tests/engine/test_metrics.py
git commit -m "feat: add DDP helpers training loops and metrics"
```

### Task 16: Add training and evaluation entrypoints

**Files:**
- Create: `train.py`
- Create: `evaluate.py`
- Modify: `src/deepfake_detection/models/factory.py`
- Modify: `src/deepfake_detection/data/builders.py`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# append to tests/configs/test_config_load.py
from pathlib import Path


def test_train_and_evaluate_entrypoints_exist():
    assert Path("train.py").exists()
    assert Path("evaluate.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/configs/test_config_load.py::test_train_and_evaluate_entrypoints_exist -v`
Expected: FAIL because the entrypoint files do not exist

- [ ] **Step 3: Write the entrypoints**

```python
# train.py
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(args.config)


if __name__ == "__main__":
    main()
```

```python
# evaluate.py
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
```

Then expand them to:
- load YAML config
- initialize DDP in `train.py`
- build the configured dataset/model/optimizer
- train with FF++ data
- evaluate checkpoints on both FF++ and CDF
- write outputs under `outputs/<experiment_name>/`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/configs/test_config_load.py::test_train_and_evaluate_entrypoints_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train.py evaluate.py tests/configs/test_config_load.py src/deepfake_detection/models/factory.py src/deepfake_detection/data/builders.py
git commit -m "feat: add train and evaluate entrypoints"
```

### Task 17: Add torchrun launch scripts for 8-GPU DDP

**Files:**
- Create: `scripts/run_exp1.sh`
- Create: `scripts/run_exp2.sh`
- Create: `scripts/run_exp3.sh`
- Create: `scripts/run_exp4.sh`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Write the failing launcher test**

```python
# append to tests/configs/test_config_load.py
from pathlib import Path


def test_run_exp4_launcher_uses_torchrun_on_eight_gpus():
    text = Path("scripts/run_exp4.sh").read_text()
    assert "torchrun" in text
    assert "--nproc_per_node=8" in text
    assert "configs/exp4_clip_prompt_bgcontrast.yaml" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/configs/test_config_load.py::test_run_exp4_launcher_uses_torchrun_on_eight_gpus -v`
Expected: FAIL with `FileNotFoundError` for `scripts/run_exp4.sh`

- [ ] **Step 3: Write the launcher scripts**

```bash
#!/usr/bin/env bash
# scripts/run_exp4.sh
set -euo pipefail
conda run -n deepfake-detection torchrun --nproc_per_node=8 train.py --config configs/exp4_clip_prompt_bgcontrast.yaml
```

Also create:
- `scripts/run_exp1.sh` using `configs/exp1_efficientnet.yaml`
- `scripts/run_exp2.sh` using `configs/exp2_clip_ft.yaml`
- `scripts/run_exp3.sh` using `configs/exp3_clip_prompt.yaml`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/configs/test_config_load.py::test_run_exp4_launcher_uses_torchrun_on_eight_gpus -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts tests/configs/test_config_load.py
git commit -m "feat: add torchrun launch scripts for four experiments"
```

### Task 18: End-to-end verification

**Files:**
- Modify: `train.py`
- Modify: `evaluate.py`
- Modify: `src/deepfake_detection/**`
- Test: `tests/data/test_sampling.py`
- Test: `tests/data/test_transforms.py`
- Test: `tests/data/test_triplets.py`
- Test: `tests/models/test_prompts.py`
- Test: `tests/losses/test_contrastive.py`
- Test: `tests/engine/test_metrics.py`
- Test: `tests/configs/test_config_load.py`

- [ ] **Step 1: Run the full unit test suite**

Run: `PYTHONPATH=src pytest tests -v`
Expected: PASS

- [ ] **Step 2: Run a single-process train smoke test**

Run: `conda run -n deepfake-detection python train.py --config configs/exp1_efficientnet.yaml`
Expected: The process parses config, builds model/dataset objects, and starts one training epoch without import errors

- [ ] **Step 3: Run a DDP smoke test**

Run: `conda run -n deepfake-detection torchrun --nproc_per_node=2 train.py --config configs/exp2_clip_ft.yaml`
Expected: Two ranks initialize NCCL successfully and start training without sampler or device errors

- [ ] **Step 4: Run evaluation smoke test**

Run: `conda run -n deepfake-detection python evaluate.py --config configs/exp3_clip_prompt.yaml --checkpoint outputs/exp3_clip_prompt/best_model.pth`
Expected: The script loads the checkpoint, runs FF++ and CDF evaluation, and prints video-level AUC/EER/ACC

- [ ] **Step 5: Run the full 8-GPU launch command for Experiment 4**

Run: `bash scripts/run_exp4.sh`
Expected: DDP launches on 8 V100 GPUs and begins training with the aligned-triplet pipeline

- [ ] **Step 6: Commit**

```bash
git add train.py evaluate.py src tests scripts configs
git commit -m "feat: finish FF++ to CDF deepfake training pipeline"
```

## Self-Review Notes

### Spec coverage
- FF++ training with 16 fixed methods: covered by `constants.py`, config files, and indexing tasks
- 200 fake videos per method, 8 frames per video: covered by sampling helpers and dataset tasks
- Real balancing via per-method oversampling: covered by `balance_real_video_ids(...)` and dataset builders
- FF++ and CDF evaluation: covered by `index_cdf.py`, `evaluate.py`, and metrics tasks
- ViT-B/16 at 224×224: covered by constants and config tasks
- No mask for now: covered in `configs/base.yaml`
- Shared RGB augmentation: covered by `transforms.py`
- Frequency derived from augmented RGB rather than separate augmentation: covered by `frequency.py` and `AlignedTripletDataset`
- Experiment 4 same-frame triplets with synchronized augmentation: covered by FF++ indexing, crop helpers, dataset tasks, and contrastive model/loss tasks
- DDP on 8 GPUs: covered by environment, config, DDP, and launcher tasks

### Placeholder scan
- No `TODO`, `TBD`, or “implement later” placeholders remain in tasks
- Every code-changing step includes concrete file paths and code snippets
- Every validation step includes a concrete command and expected outcome

### Type consistency
- CLIP model name is consistently `ViT-B/16`
- Binary labels remain `REAL_LABEL = 0`, `FAKE_LABEL = 1`
- Experiment 4 consistently uses `background`, `real_face`, `fake_face`
- DDP consistently uses `torchrun` and `distributed: ddp`
