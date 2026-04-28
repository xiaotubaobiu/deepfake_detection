# FF++→CDF Deepfake Detection Experiment Design

## Goal

Design a controlled deepfake detection study under a **cross-domain but not cross-method** setting.

The study trains on **16 selected forgery methods from FF++** and evaluates on **FF++ and CDF** while keeping the manipulation method set fixed between training and testing. The main question is whether stronger visual priors, language priors, and explicit background-face consistency constraints improve cross-domain generalization.

The task is defined as:

- **Binary classification:** real vs fake

## Forgery Methods

### Face swap methods
- simswap
- inswap
- blendface
- faceswap
- fsgan
- mobileswap
- e4s
- facedancer

### Face reenactment methods
- fomm
- facevid2vid
- wav2lip
- sadtalker
- MRAA
- pirender
- tpsm
- lia

Total: **16 methods**.

## Data Protocol

### Training set
Training uses only **FF++ source-domain data**.

For each fake method:
- sample **200 fake videos**
- sample **8 frames uniformly** from each video
- use all resulting samples for **real/fake binary classification**

### Real-sample construction
Real samples are balanced at the **per-method** level.

For each fake method, construct a matched real subset such that:
- the number of real videos matches the number of fake videos for that method
- if the real pool is insufficient, use **video-level oversampling**
- each real video also contributes **8 uniformly sampled frames**

This yields a strictly balanced real/fake training subset inside each method.

### Test sets
Testing uses the official splits without artificially capping the number of test videos:
- **FF++ test set**
- **CDF test set**

The evaluation protocol is:
- **cross-domain**
- **not cross-method**

The same 16 forgery methods are used in training and testing. The only shift of interest is the domain shift from FF++ to CDF.

## Sample Organization

Although each video contributes 8 frames, this study does **not** introduce temporal modeling. All four experiments use a frame-based pipeline with video-level aggregation.

### Training
- sample 8 frames from each video
- expand them into 8 independent frame-level training samples
- keep the same frame-level training protocol across all methods for fairness

### Testing
- predict a fake probability for each of the 8 sampled frames
- average the 8 frame-level scores
- use the mean as the final video-level score

This keeps the comparison clean by avoiding extra temporal variables.

## Four Experiments

### Experiment 1: EfficientNet baseline
- input: single-frame RGB image
- model: EfficientNet with a binary classification head
- role: pure visual classification baseline

### Experiment 2: CLIP fine-tuning
- input: single-frame image
- model: CLIP image encoder with a binary classification head
- training: fine-tune the visual encoder
- role: test whether generic pretrained visual representations transfer well to deepfake detection

### Experiment 3: CLIP + language prompt
- input: image plus text prompts
- task: real/fake binary classification
- prompt type: fixed prompts, not learnable prompts

Suggested prompt templates:
- real:
  - `a real face image`
  - `an authentic face photo`
- fake:
  - `a fake face image`
  - `a manipulated face photo`

The purpose is to test whether language priors improve cross-domain robustness.

### Experiment 4: CLIP + prompt + background-face contrast
Experiment 4 extends Experiment 3 with an explicit **background-face consistency constraint**.

The main hypothesis is:
- the background is real
- a real face is more consistent with its background
- a fake face is less consistent with the same background

## Experiment 4: Aligned Triplet Design

### Strict frame-level aligned triplets
Experiment 4 uses **strictly aligned frame-level triplets** rather than ordinary frame samples.

For each manipulated sample, build:
- `b`: the real background from a frame
- `f_real`: the real face from the same frame
- `f_fake`: the fake face from the aligned fake frame

Requirements:
- all three elements must come from the **same source-target pair**
- all three elements must come from the **same frame index**
- pairing must rely only on **officially alignable paired data**
- no approximate matching
- no cross-frame pairing

So each usable training item for Experiment 4 is derived from:
- a real frame
- its aligned fake frame
- a background crop from the real frame
- a real-face crop from the real frame
- a fake-face crop from the fake frame

### Positive and negative pairs
Use the background as the anchor:
- **positive pair:** `(b, f_real)`
- **negative pair:** `(b, f_fake)`

This forces the model to learn that the same background should be more compatible with the true face than with the manipulated face.

### Loss
The total loss is:

`L_total = L_cls + λ * L_align`

where:
- `L_cls` is the real/fake classification loss
- `L_align` is the background-face consistency contrastive loss
- `λ` is the auxiliary loss weight

This remains an **InfoNCE-style** contrastive objective:
- anchor = background embedding
- positive = same-frame real-face embedding
- negative = same-frame fake-face embedding

Optional batch negatives may be added later, but the core structure must remain the same-frame one-positive one-negative design.

### Representation level
Apply the contrastive objective directly between:
- **face embeddings**
- **background embeddings**

Do not place it only on fused features. This keeps the learning objective aligned with the core hypothesis and makes the result easier to interpret.

## Input Views

To support all four experiments, each frame can provide:
- **full image**
- **face crop**
- **background crop**

Usage:
- Experiments 1, 2, and 3 mainly use the full image
- Experiment 4 additionally uses the face crop and background crop

This preserves a shared data basis while only Experiment 4 explicitly models face-background relations.

## Prompt Design Principle

Prompts are **not** the main research object in this study, so prompt design should stay minimal and controlled.

Use:
- **fixed prompts**
- no learnable prompt tuning
- no heavy prompt engineering

This keeps the focus on:
- CNN vs CLIP
- no language prior vs language prior
- no structural constraint vs background-face consistency constraint

## Evaluation Protocol

### Primary evaluation level
Report **video-level** results as the main outcome. Frame-level numbers can be used only for auxiliary analysis.

### Main metrics
Use:
- **AUC**
- **EER**
- **ACC**

Recommended emphasis:
- AUC as the main metric
- EER because it is standard in deepfake detection
- ACC for intuitive comparison

## Result Tables

### Main result table
Rows:
- EfficientNet
- CLIP fine-tuning
- CLIP + prompt
- CLIP + prompt + background-face contrast

Columns:
- FF++ AUC / EER / ACC
- CDF AUC / EER / ACC

This table answers the main question:
- given FF++ training, which method is most stable both in-domain and cross-domain

### Per-method result table
Add a breakdown table for the 16 forgery methods:
- one row per fake method
- columns report model performance for that method, especially **CDF AUC**

This table helps analyze:
- which methods suffer most from domain shift
- which methods benefit most from prompts or background-face contrast

### Ablation table for Experiment 4
Add a focused ablation table with:
- CLIP + prompt
- CLIP + prompt + contrast
- different `λ` values
- same-frame single negative only vs adding batch negatives

This verifies whether the gain from Experiment 4 is stable and whether it truly comes from the consistency constraint.

## Final Summary

The final study is defined as:
- **task:** real/fake binary classification
- **training:** 16 selected methods from FF++, 200 videos per method, 8 frames per video
- **balancing:** real samples balanced per method with oversampling when needed
- **testing:** FF++ and CDF, cross-domain but not cross-method
- **models:**
  1. EfficientNet
  2. CLIP fine-tuning
  3. CLIP + fixed prompts
  4. CLIP + fixed prompts + same-frame background/real-face/fake-face contrast

The key contribution of the fourth setting is to explicitly model the fact that the same real background should be more consistent with the true face than with the manipulated face, using strictly aligned frame-level triplets to strengthen cross-domain generalization.
