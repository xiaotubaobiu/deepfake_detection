# Exp4: CLIP + Background-Face Contrast

## Goal

Learn cross-domain deepfake detection by contrasting background patches against real/fake faces, exploiting the inconsistency between forged faces and their surrounding context.

## Architecture

Shared CLIP ViT-B/16 visual encoder (trainable) with three heads:

- **classifier**: Linear(512, 2) — standard real/fake classification on face crops
- **bg_projection**: Linear(512, proj_dim) — projects background patch features
- **face_projection**: Linear(512, proj_dim) — projects face crop features

## Data: Triplet Construction

Each training sample is a `(bg_patches, real_face, fake_face)` triplet aligned by `(pair_id, frame_name)`:

| Component | Source | Path pattern |
|-----------|--------|-------------|
| bg patches ×4 | Anchor full frame, crop 224×224 patches avoiding face region | `DF40_train/anchor/<vid>/<frame>.png` |
| real face | FF++ original sequence face frame | `FaceForensics++/original_sequences/.../<vid>/<frame>.png` |
| fake face | Forgery method face frame | `DF40_train/<method>/frames/<pair_id>/<frame>.png` |

Face region detected via landmarks (`DF40_train/<method>/landmarks/<pair_id>/<frame>.npy`). Background patches sampled randomly from non-face regions of the anchor frame.

All three components share the same data augmentation (applied identically to bg patches, real face, fake face).

## Training Loss

```
L = L_cls + lambda * L_InfoNCE
```

**L_cls**: Cross-entropy on classifier output using face crop input.

**L_InfoNCE**:
- Positive pairs: 4 × (bg_patch_i, real_face) — same-frame background and real face are consistent
- Negative pairs: 4 × (bg_patch_i, fake_face) + cross-sample negatives within batch — fake face is inconsistent with background

Temperature-scaled cosine similarity in projection space.

## Inference Fusion

```
cls_prob = softmax(classifier(visual(face)))
face_feat = face_projection(visual(face))
bg_feat_i = bg_projection(visual(bg_patch_i))
consistency = mean(cosine_sim(bg_feat_i, face_feat))  # high = real, low = fake

final_prob = (1 - alpha) * cls_prob + alpha * consistency_prob
```

During inference, bg patches are cropped from the anchor frame using face detection to locate and avoid the face region.

## Hyperparameters

| Param | Value |
|-------|-------|
| clip_model | ViT-B/16 |
| projection_dim | 256 |
| lambda (contrastive weight) | 0.1 |
| temperature (InfoNCE) | 0.07 |
| alpha (inference fusion) | 0.3 |
| lr | 2e-5 |
| weight_decay | 5e-4 |
| per_gpu_batch | 128 |
| epochs | 5 |
| patience | 3 |
| bg_patches_per_frame | 4 |
| patch_size | 224×224 |

## Evaluation

Same as exp1–exp3: Val AUC, FF++ test AUC, CDF test AUC. Cross-seed robustness testing (seed 42, 7, 123) with full deterministic training pipeline.

## Expected Outcome

Background-face consistency provides a domain-invariant signal: real faces naturally belong to their background, fake faces do not. This should improve CDF cross-domain generalization over exp2/exp3.
