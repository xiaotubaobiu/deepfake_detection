# Deepfake Detection Feature Geometry Experiments

This repository contains the code and retained artifacts for studying whether CLIP-based deepfake detection transfers across datasets through feature direction, feature length, or both.

## Main result

The paper compares three evaluation modes on the same raw checkpoints:

- `raw`: classifier uses both feature direction and feature length.
- `feature_norm`: features are L2-normalized before classification, so the classifier mainly uses direction.
- `feature_length`: the feature norm is used as the score, isolating length alone.

Across FF++, CDF, and DFD, the observed pattern is:

- FF++: raw and direction-only are both near-perfect.
- CDF: direction-only > raw > length-only.
- DFD: raw > direction-only >> length-only.

## Repository layout

```text
train.py                              # training entrypoint
evaluate.py                           # standard FF++ / CDF / DFD evaluation entrypoint
scripts/evaluate_norm_correction.py   # paper raw / direction / length evaluation
scripts/prepare_dfd_df40.py           # prepare DFD in DF40-style JSON/frame layout
scripts/summarize_paper_results.py    # print paper AUC tables
src/deepfake_detection/               # package code
experiments/                          # retained paper artifacts only
```

## Installation

```bash
conda env create -f environment.yml
conda activate deepfake-detection
pip install -r requirements.txt
```

If you use an existing environment, make sure PyTorch, torchvision, OpenCV, scikit-learn, pandas, PyYAML, and CLIP dependencies are installed.

## Dataset layout

The code expects DF40/DeepfakeBench-style JSON indexes under the configured dataset root. DFD is loaded from:

```text
/Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json
```

with frame paths under the FaceForensics++-style DFD folders.

## Prepare DFD

```bash
PYTHONPATH=src python scripts/prepare_dfd_df40.py \
  --root /Dataset/deepfake_detection \
  --output-json /Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json \
  --frames-per-video 32
```

## Train

```bash
PYTHONPATH=src python train.py --config experiments/20260505_130308/config/config.yaml
```

Use the config from the timestamp folder closest to the experiment you want to reproduce.

## Standard evaluation

```bash
PYTHONPATH=src python evaluate.py \
  --config experiments/20260505_130308/config/config.yaml \
  --checkpoint $(cat experiments/20260505_130308/output/checkpoint_path.txt) \
  --domain all
```

The retained artifact folders usually store the original checkpoint path in `output/checkpoint_path.txt` instead of copying large checkpoint files into the repository.

## Paper feature-geometry evaluation

```bash
PYTHONPATH=src python scripts/evaluate_norm_correction.py \
  --config experiments/20260505_130308/config/config.yaml \
  --checkpoint $(cat experiments/20260505_130308/output/checkpoint_path.txt) \
  --splits ffpp,cdf,dfd \
  --mode raw,feature_norm,feature_length \
  --output-dir experiments/20260505_130308/output
```

## Summarize paper results

```bash
PYTHONPATH=src python scripts/summarize_paper_results.py
```

This prints the AUC table used in the paper.

## Retained experiment artifacts

The retained experiment folders are:

```text
experiments/20260505_130308   raw seed 42
experiments/20260505_163304   raw seed 123
experiments/20260505_200221   raw seed 7
experiments/20260505_234842   raw seed 999
experiments/20260506_032147   raw seed 2048
experiments/20260505_145908   normtrain seed 42
experiments/20260505_181828   normtrain seed 123
experiments/20260505_215543   normtrain seed 7
experiments/20260506_013459   normtrain seed 999
experiments/20260506_050719   normtrain seed 2048
experiments/dfd_threshold_analysis
```

Each timestamp folder uses:

```text
config/              # training config and resolved config
logs/                # retained train/evaluation logs
output/              # summaries, metadata, paper CSVs, checkpoint_path.txt
experiment_name.txt  # original experiment name
```

## Metrics

AUC is the main ranking metric and is independent of any fixed threshold. EER also scans thresholds. ACC, balanced ACC, real recall, and fake recall depend on the chosen threshold.

DFD is highly imbalanced, with fake samples roughly ten times more frequent than real samples. Ordinary accuracy can therefore be misleading: an almost-always-fake threshold can reach high ordinary ACC while giving real recall near zero. Use AUC and balanced accuracy for DFD analysis.
