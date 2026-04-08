# Dual-Domain Face-Context Relation for Deepfake Detection

**Date**: 2026-04-08
**Status**: Draft
**Conda env**: `researchclaw`

## Core Hypothesis

Face-context relation (inconsistency between face crop and surrounding context) is a useful deepfake signal. This relation is complementary in RGB and frequency domains, and contrastive learning (InfoNCE on relation embeddings) makes it more robust across unseen forgery methods.

## Experiments

4 image-level models, no video/temporal experiments (DF40 frames are uniformly sampled, not consecutive).

| Model | Encoders | Relation dim | Loss |
|-------|----------|-------------|------|
| M1: RGB-only Relation | E_rgb (shared for face+context) | 2048 | CE |
| M2: Freq-only Relation | E_freq (shared for face+context) | 2048 | CE |
| M3: RGB+Freq Relation | E_rgb + E_freq (2 encoders) | 4096 | CE |
| M4: RGB+Freq+InfoNCE | same as M3 | 4096 | CE + lambda * InfoNCE |

### Success Criteria

- M3 > M1 and M3 > M2: dual-domain relation is more comprehensive than single-domain
- M4 > M3 on Celeb-DF cross-domain test: contrastive learning improves generalization

## Dataset

### Training: DF40

- Path: `/Dataset/deepfake_detection/DF40_all/DF40_train/`
- Real: `anchor/` (999 videos, 32 frames each)
- Fake: 31 forgery methods (each ~988 videos)
- Landmarks: pre-extracted in `<method>/landmarks/` directories
- Frame format: PNG, named `000.png` ... (every ~10th frame, not consecutive)
- Train/val split: 90/10 by video ID (not by frame)
- Sampling: 1 frame per video for image-level experiments
- Label balance: real=999 videos, fake=31×988≈30,628 videos. Use balanced sampling or weighted CE to handle imbalance.

### Testing

- **DF40-test**: `/Dataset/deepfake_detection/DF40_all/DF40_test/` (same-domain generalization)
- **Celeb-DF-v2**: `/Dataset/deepfake_detection/DF40_all/Celeb-DF-v2/` (cross-domain generalization)
  - Face detection: SCRFD via `det_10g.onnx` from `~/project/evaluation_metic/models/buffalo_l/`

## Input Processing

### Face Crop

1. From landmarks (DF40) or SCRFD (Celeb-DF), get face bounding box
2. Square-pad the box
3. Resize to 224x224

### Context Crop

1. Use face box center, expand to 1.8x area
2. Crop from original image
3. Mask out the face region (fill with black)
4. Resize to 224x224

Each sample produces 4 inputs: `face_rgb`, `context_rgb`, `face_freq`, `context_freq`

### Frequency Transform (DCT)

```python
def to_dct_map(img_gray):
    # 1. Ensure grayscale
    # 2. 2D DCT via scipy.fftpack.dctn(img, type=2, norm='ortho')
    # 3. Take absolute value
    # 4. log(1 + |DCT|)
    # 5. Normalize to [0, 1]
    # 6. Resize to 224x224
    # Output: single channel, replicated to 3 channels for ResNet18
```

## Model Architecture

### Encoders (`models/encoder.py`)

- ResNet18 (torchvision pretrained, ImageNet)
- Remove final FC layer, output 512d
- Two independent encoders: E_rgb, E_freq
- Within each domain, face and context share the same encoder

### Relation (`models/relation.py`)

```python
r = concat([z_face, z_context, |z_face - z_context|, z_face * z_context])
# single domain: [B, 2048]
# dual domain: concat(r_rgb, r_freq) -> [B, 4096]
```

### Classifier (`models/classifier.py`)

- Classification head: relation_dim -> 512 (ReLU, Dropout 0.5) -> 256 (ReLU) -> 2
- Projection head (M4 only): relation_dim -> 256 -> 128 (for InfoNCE)

## Contrastive Learning (M4)

### Data Augmentation (pair-synchronized)

Applied to both face_rgb and context_rgb together, then re-generate freq maps:

- JPEG compression (quality 50-95)
- Gaussian blur (sigma 0.5-1.5)
- Brightness/contrast jitter (±0.2)
- Slight color jitter
- Slight resize (0.9-1.1x)

Two views per sample, different random params.

### InfoNCE Loss

Applied on relation embedding (not on individual face/context features):

```
h = ProjectionHead(r)
L_nce = -log( exp(sim(h_i^1, h_i^2) / tau) / sum_j exp(sim(h_i^1, h_j^2) / tau) )
```

- tau = 0.07
- lambda = 0.1
- Total loss: L = L_cls + lambda * L_nce

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 30 |
| Early stopping | patience=5, monitor val AUC |
| AMP | enabled |
| InfoNCE tau | 0.07 |
| InfoNCE lambda | 0.1 |

## Evaluation

### Metrics

- AUC (Area Under ROC Curve)
- EER (Equal Error Rate)

### Results Table

```
Model                  | DF40-test AUC | DF40-test EER | Celeb-DF AUC | Celeb-DF EER
M1: RGB-only Relation  |               |               |              |
M2: Freq-only Relation |               |               |              |
M3: RGB+Freq Relation  |               |               |              |
M4: RGB+Freq+InfoNCE   |               |               |              |
```

## Project Structure

```
deepfake_detection/
├── configs/
│   └── experiments.yaml
├── data/
│   ├── __init__.py
│   ├── df40_dataset.py
│   ├── celebdf_dataset.py
│   ├── transforms.py
│   └── frequency.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── relation.py
│   └── classifier.py
├── losses/
│   ├── __init__.py
│   └── infonce.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── run_experiments.sh
└── docs/superpowers/specs/
```

## Execution Order

1. Implement data loading (DF40 + Celeb-DF), DCT transform, face/context cropping
2. Implement models (encoder, relation, classifier)
3. Implement losses (CE + InfoNCE)
4. Implement training loop + evaluation
5. Run M1 (RGB-only) -> verify pipeline works end-to-end
6. Run M2 (Freq-only)
7. Run M3 (RGB+Freq)
8. Run M4 (RGB+Freq+InfoNCE)
9. Evaluate all on DF40-test + Celeb-DF, fill results table

## Dependencies

Using conda env `researchclaw` (PyTorch 2.10, CUDA 12, scipy, scikit-learn).

Additional installs needed:
- `torchvision` (for ResNet18)
- `opencv-python` (for image I/O)
- `insightface` + `onnxruntime-gpu` (for SCRFD face detection on Celeb-DF)

Face detector: `~/project/evaluation_metic/models/buffalo_l/det_10g.onnx` (SCRFD-10G)
