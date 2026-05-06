# Paper Artifact Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the repository into a clean paper reproduction artifact that keeps only the code and experiment data needed for the feature direction / feature length / raw deepfake detection paper.

**Architecture:** The root keeps `train.py` and `evaluate.py` as the only public entrypoints. Auxiliary scripts move under `scripts/`; paper-used experiment artifacts live directly under `experiments/<timestamp>/`; threshold CSVs live under `experiments/dfd_threshold_analysis/`; a root `README.md` documents how to use the cleaned repository.

**Tech Stack:** Python, PyTorch, CLIP-based deepfake detection code under `src/deepfake_detection`, pytest, shell utilities for file moves and verification.

---

## File Structure

### Root files

- Keep: `train.py` — training entrypoint.
- Keep: `evaluate.py` — standard evaluation entrypoint.
- Move: `evaluate_norm_correction.py` → `scripts/evaluate_norm_correction.py` — paper-specific raw / feature_norm / feature_length evaluation.
- Move: `evaluate_norm_probe.py` → `scripts/evaluate_norm_probe.py` if still useful; otherwise delete if no README command references it.
- Create: `README.md` — paper reproduction instructions.

### Scripts

- Keep: `scripts/prepare_dfd_df40.py` — DFD frame/JSON preparation helper.
- Keep: `scripts/evaluate_norm_correction.py` — paper decomposition evaluation helper.
- Create: `scripts/summarize_paper_results.py` — summarize the retained CSVs into paper tables if not already present.
- Delete obsolete shell experiment scripts not referenced by README.

### Experiments

Keep only:

```text
experiments/20260505_130308/
experiments/20260505_145908/
experiments/20260505_163304/
experiments/20260505_181828/
experiments/20260505_200221/
experiments/20260505_215543/
experiments/20260505_234842/
experiments/20260506_013459/
experiments/20260506_032147/
experiments/20260506_050719/
experiments/dfd_threshold_analysis/
```

Delete all other `experiments/20*` directories.

### Tests

Keep tests that validate retained code paths:

- `tests/configs/`
- `tests/data/`
- `tests/engine/`
- `tests/evaluation/` if it still passes with retained entrypoints.
- `tests/models/` and `tests/losses/` if they cover retained training code.

Delete only tests that import removed scripts or obsolete functionality.

---

### Task 1: Inventory and protect paper artifacts

**Files:**
- Inspect: `experiments/`
- Create if needed: `experiments/dfd_threshold_analysis/`

- [ ] **Step 1: List experiment directories before deletion**

Run:

```bash
find experiments -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
```

Expected: output includes the ten paper timestamp folders plus many temporary `20260504*` and `20260506*` directories.

- [ ] **Step 2: Verify raw paper folders contain required outputs**

Run:

```bash
for d in 20260505_130308 20260505_163304 20260505_200221 20260505_234842 20260506_032147; do
  test -f "experiments/$d/output/results_summary.txt" && \
  test -f "experiments/$d/output/angle_length_metrics.csv" && \
  test -f "experiments/$d/output/length_only_metrics.csv" && \
  echo "ok raw $d"
done
```

Expected:

```text
ok raw 20260505_130308
ok raw 20260505_163304
ok raw 20260505_200221
ok raw 20260505_234842
ok raw 20260506_032147
```

- [ ] **Step 3: Verify normtrain paper folders contain required summaries**

Run:

```bash
for d in 20260505_145908 20260505_181828 20260505_215543 20260506_013459 20260506_050719; do
  test -f "experiments/$d/output/results_summary.txt" && \
  test -f "experiments/$d/output/meta.json" && \
  echo "ok normtrain $d"
done
```

Expected:

```text
ok normtrain 20260505_145908
ok normtrain 20260505_181828
ok normtrain 20260505_215543
ok normtrain 20260506_013459
ok normtrain 20260506_050719
```

- [ ] **Step 4: Preserve DFD threshold CSVs**

Run:

```bash
mkdir -p experiments/dfd_threshold_analysis
find experiments -path '*dfd_best_thresholds.csv' -o -path '*dfd_best_balanced_thresholds.csv'
```

Expected: find prints the current locations of `dfd_best_thresholds.csv` and `dfd_best_balanced_thresholds.csv`, or they are already in `experiments/dfd_threshold_analysis/`.

- [ ] **Step 5: Move threshold CSVs into the retained folder**

Run:

```bash
for f in $(find experiments -path '*dfd_best_thresholds.csv' -o -path '*dfd_best_balanced_thresholds.csv'); do
  cp "$f" "experiments/dfd_threshold_analysis/$(basename "$f")"
done
ls experiments/dfd_threshold_analysis
```

Expected:

```text
dfd_best_balanced_thresholds.csv
dfd_best_thresholds.csv
```

---

### Task 2: Delete non-paper experiment artifacts

**Files:**
- Delete: non-retained `experiments/20*` directories.
- Delete: leftover aggregate folders `experiments/configs`, `experiments/logs`, `experiments/outputs`, `experiments/timeline` if present.

- [ ] **Step 1: Print deletion candidates**

Run:

```bash
python - <<'PY'
from pathlib import Path
keep = {
    '20260505_130308', '20260505_145908', '20260505_163304', '20260505_181828',
    '20260505_200221', '20260505_215543', '20260505_234842', '20260506_013459',
    '20260506_032147', '20260506_050719', 'dfd_threshold_analysis'
}
root = Path('experiments')
for p in sorted(root.iterdir()):
    if p.is_dir() and p.name not in keep:
        print(p)
PY
```

Expected: output lists temporary experiment directories such as `experiments/20260504_*`, `experiments/20260506_093629`, and no retained paper timestamp folder.

- [ ] **Step 2: Delete deletion candidates**

Run:

```bash
python - <<'PY'
from pathlib import Path
import shutil
keep = {
    '20260505_130308', '20260505_145908', '20260505_163304', '20260505_181828',
    '20260505_200221', '20260505_215543', '20260505_234842', '20260506_013459',
    '20260506_032147', '20260506_050719', 'dfd_threshold_analysis'
}
root = Path('experiments')
for p in sorted(root.iterdir()):
    if p.is_dir() and p.name not in keep:
        shutil.rmtree(p)
PY
```

Expected: command exits successfully.

- [ ] **Step 3: Verify only retained experiment folders remain**

Run:

```bash
find experiments -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
```

Expected exactly:

```text
20260505_130308
20260505_145908
20260505_163304
20260505_181828
20260505_200221
20260505_215543
20260505_234842
20260506_013459
20260506_032147
20260506_050719
dfd_threshold_analysis
```

---

### Task 3: Move auxiliary evaluation scripts under `scripts/`

**Files:**
- Move: `evaluate_norm_correction.py` → `scripts/evaluate_norm_correction.py`
- Move/delete: `evaluate_norm_probe.py`
- Modify references in README later.

- [ ] **Step 1: Move root helper scripts**

Run:

```bash
mv evaluate_norm_correction.py scripts/evaluate_norm_correction.py
if [ -f evaluate_norm_probe.py ]; then mv evaluate_norm_probe.py scripts/evaluate_norm_probe.py; fi
```

Expected: root no longer contains `evaluate_norm_correction.py` or `evaluate_norm_probe.py`.

- [ ] **Step 2: Verify root entrypoints**

Run:

```bash
find . -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: includes `train.py` and `evaluate.py`; does not include `evaluate_norm_correction.py` or `evaluate_norm_probe.py`.

- [ ] **Step 3: Compile moved script**

Run:

```bash
PYTHONPATH=src python -m py_compile scripts/evaluate_norm_correction.py
```

Expected: no output and exit code 0.

---

### Task 4: Remove obsolete scripts

**Files:**
- Inspect/delete files in `scripts/`.
- Keep: `scripts/prepare_dfd_df40.py`, `scripts/evaluate_norm_correction.py`, `scripts/summarize_norm_shortcut_results.py` if useful.
- Create later: `scripts/summarize_paper_results.py`.

- [ ] **Step 1: List scripts**

Run:

```bash
find scripts -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: output shows current helper scripts.

- [ ] **Step 2: Delete obsolete shell batch scripts**

Run:

```bash
find scripts -maxdepth 1 -type f \( -name 'run_*.sh' -o -name '*experiment*.sh' \) -delete
```

Expected: no output.

- [ ] **Step 3: Verify retained scripts**

Run:

```bash
find scripts -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: includes `prepare_dfd_df40.py` and `evaluate_norm_correction.py`; obsolete batch shell scripts are absent.

---

### Task 5: Add paper result summarizer

**Files:**
- Create: `scripts/summarize_paper_results.py`

- [ ] **Step 1: Write summarizer script**

Create `scripts/summarize_paper_results.py` with this content:

```python
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RAW_RUNS = {
    "s42": "20260505_130308",
    "s123": "20260505_163304",
    "s7": "20260505_200221",
    "s999": "20260505_234842",
    "s2048": "20260506_032147",
}


def read_auc_by_variant(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    result: dict[tuple[str, str], float] = {}
    for row in rows:
        result[(row["split"], row["variant"])] = float(row["auc"])
    return result


def main() -> None:
    rows = []
    for seed, timestamp in RAW_RUNS.items():
        output = EXPERIMENTS / timestamp / "output"
        angle = read_auc_by_variant(output / "angle_length_metrics.csv")
        length = read_auc_by_variant(output / "length_only_metrics.csv")
        for split in ["ffpp", "cdf", "dfd"]:
            rows.append({
                "dataset": split.upper(),
                "seed": seed,
                "length": length[(split, "feature_length")],
                "angle": angle[(split, "feature_norm")],
                "raw": angle[(split, "raw")],
            })

    print("dataset,seed,length,angle,raw")
    for row in rows:
        print(f"{row['dataset']},{row['seed']},{row['length']:.4f},{row['angle']:.4f},{row['raw']:.4f}")

    print("\nmean_auc")
    print("dataset,length,angle,raw")
    for dataset in ["FFPP", "CDF", "DFD"]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        print(
            f"{dataset},"
            f"{mean(row['length'] for row in dataset_rows):.4f},"
            f"{mean(row['angle'] for row in dataset_rows):.4f},"
            f"{mean(row['raw'] for row in dataset_rows):.4f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run summarizer**

Run:

```bash
PYTHONPATH=src python scripts/summarize_paper_results.py
```

Expected: prints per-seed table and mean AUC table matching the paper values.

---

### Task 6: Write root README

**Files:**
- Create/overwrite: `README.md`

- [ ] **Step 1: Write README**

Write `README.md` with these sections:

```markdown
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
train.py                         # training entrypoint
evaluate.py                      # standard FF++ / CDF / DFD evaluation entrypoint
scripts/evaluate_norm_correction.py  # paper raw / direction / length evaluation
scripts/prepare_dfd_df40.py       # prepare DFD in DF40-style JSON/frame layout
scripts/summarize_paper_results.py # print paper AUC tables
src/deepfake_detection/           # package code
experiments/                      # retained paper artifacts only
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
  --checkpoint experiments/20260505_130308/output/best_model.pth \
  --domain all
```

If `best_model.pth` is not stored in the artifact folder, read `output/checkpoint_path.txt` for the original checkpoint path.

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

## Metrics

AUC is the main ranking metric and is independent of any fixed threshold. EER also scans thresholds. ACC, balanced ACC, real recall, and fake recall depend on the chosen threshold.

DFD is highly imbalanced, with fake samples roughly ten times more frequent than real samples. Ordinary accuracy can therefore be misleading: an almost-always-fake threshold can reach high ordinary ACC while giving real recall near zero. Use AUC and balanced accuracy for DFD analysis.
```

- [ ] **Step 2: Check README mentions retained commands only**

Run:

```bash
grep -nE 'evaluate_norm_probe|outputs/|timeline/|experiments/configs' README.md || true
```

Expected: no output.

---

### Task 7: Update references to moved helper script

**Files:**
- Search/modify: project files referencing `evaluate_norm_correction.py` or old experiment paths.

- [ ] **Step 1: Search old script references**

Run:

```bash
grep -RIn "evaluate_norm_correction.py\|evaluate_norm_probe.py\|experiments/outputs\|experiments/timeline\|experiments/configs" . --exclude-dir=.git --exclude-dir=.pytest_cache
```

Expected: references appear only in docs/plans or files that should be updated.

- [ ] **Step 2: Update active references**

Use `Edit` to change active README/script references to:

```text
scripts/evaluate_norm_correction.py
experiments/<timestamp>/config/config.yaml
experiments/<timestamp>/output/...
```

Expected: active code and README no longer point to deleted paths.

---

### Task 8: Verify core entrypoints and retained tests

**Files:**
- Verify: `train.py`, `evaluate.py`, `scripts/evaluate_norm_correction.py`

- [ ] **Step 1: Compile entrypoints**

Run:

```bash
PYTHONPATH=src python -m py_compile train.py evaluate.py scripts/evaluate_norm_correction.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Check evaluation help**

Run:

```bash
PYTHONPATH=src python evaluate.py --help
```

Expected: help shows domain choices including `ffpp`, `cdf`, `dfd`, `both`, and `all`.

- [ ] **Step 3: Run retained fast tests**

Run:

```bash
PYTHONPATH=src pytest tests/configs tests/data tests/engine -q
```

Expected: tests pass. If failures import deleted scripts only, delete or update those tests; if failures affect retained loaders, fix the retained code.

---

### Task 9: Review git status and report cleanup

**Files:**
- Inspect: git status.

- [ ] **Step 1: Show status**

Run:

```bash
git status --short
```

Expected: shows deleted obsolete artifacts, moved scripts, new README, retained experiment folders, and no accidental deletion of retained paper folders.

- [ ] **Step 2: Verify retained paper folders still exist**

Run:

```bash
for d in 20260505_130308 20260505_145908 20260505_163304 20260505_181828 20260505_200221 20260505_215543 20260505_234842 20260506_013459 20260506_032147 20260506_050719 dfd_threshold_analysis; do
  test -d "experiments/$d" && echo "kept $d"
done
```

Expected: prints all eleven retained entries.

- [ ] **Step 3: Summarize changed layout**

Report to the user:

```text
Root entrypoints retained: train.py, evaluate.py
Auxiliary scripts moved under scripts/
Experiments retained: ten timestamp folders + dfd_threshold_analysis
Deleted: smoke/probe/cache/old aggregate experiment artifacts
Verification: py_compile/help/tests status
```

Do not commit unless the user explicitly asks for a commit.
