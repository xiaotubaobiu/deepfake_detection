# Deepfake Detection Paper Artifact Cleanup Design

## Goal

Restructure this repository into a clean paper reproduction artifact for the feature direction / feature length / raw deepfake detection experiments. The final repository should keep only the code and experiment artifacts needed to train, evaluate, and reproduce the paper tables.

## Approved scope

The repository should preserve:

- Core training entrypoint: `train.py` at repository root.
- Core evaluation entrypoint: `evaluate.py` at repository root.
- Python package code under `src/deepfake_detection/` needed by training and evaluation.
- Auxiliary scripts under `scripts/`, not at repository root.
- Paper-used experiment artifacts only.
- A new root `README.md` explaining installation, data preparation, training, evaluation, and paper result reproduction.

The repository may delete:

- Old smoke/probe/cache experiment outputs not used in the paper.
- Old aggregate experiment folders such as `configs/`, `logs/`, `outputs/`, and `timeline/` after movable content has been folded into timestamp folders.
- Obsolete experiment shell scripts and helper scripts unrelated to the paper reproduction workflow.
- Root-level helper evaluation scripts after moving them into `scripts/`.
- Tests that only cover removed functionality, while preserving tests for retained loaders/configs/metrics where useful.

## Experiment artifact design

`experiments/` should directly contain timestamp folders and one threshold-analysis folder:

```text
experiments/
  20260505_130308/   # raw s42
  20260505_163304/   # raw s123
  20260505_200221/   # raw s7
  20260505_234842/   # raw s999
  20260506_032147/   # raw s2048
  20260505_145908/   # normtrain s42
  20260505_181828/   # normtrain s123
  20260505_215543/   # normtrain s7
  20260506_013459/   # normtrain s999
  20260506_050719/   # normtrain s2048
  dfd_threshold_analysis/
```

Each timestamp folder should use this structure:

```text
config/
logs/
output/
experiment_name.txt
```

For the five raw checkpoints, `output/` should preserve:

```text
results_summary.txt
meta.json
checkpoint_path.txt
angle_length_metrics.csv
length_only_metrics.csv
```

For the five normtrain checkpoints, `output/` should preserve:

```text
results_summary.txt
meta.json
checkpoint_path.txt
```

The DFD threshold analysis folder should preserve:

```text
dfd_best_thresholds.csv
dfd_best_balanced_thresholds.csv
```

and any sample rows needed to recompute those threshold tables if they are still present and reasonably sized.

## Code design

The root should keep only:

```text
README.md
train.py
evaluate.py
requirements.txt
environment.yml
src/
scripts/
experiments/
tests/
docs/
```

Root-level helper scripts should be moved into `scripts/`:

```text
evaluate_norm_correction.py -> scripts/evaluate_norm_correction.py
evaluate_norm_probe.py -> scripts/evaluate_norm_probe.py
```

`evaluate.py` should remain the public evaluation entrypoint. If needed, it should either expose the raw / feature_norm / feature_length modes directly or clearly document when to use `scripts/evaluate_norm_correction.py` for paper-specific decomposition analysis.

`train.py` should remain the public training entrypoint.

`scripts/` should keep only useful auxiliary workflows, including:

```text
prepare_dfd_df40.py
evaluate_norm_correction.py
summarize_paper_results.py
```

Obsolete shell scripts from old experiment batches should be removed unless they are directly referenced by the README.

## README design

The root `README.md` should explain:

1. Project purpose and paper question.
2. Environment installation.
3. Dataset layout expected by the code.
4. DFD preparation.
5. Training command.
6. Standard evaluation command for FF++ / CDF / DFD.
7. Paper reproduction commands for raw, direction-only, and length-only evaluation.
8. Experiment artifact layout.
9. Where to find paper tables and threshold-analysis CSVs.
10. Metric notes: AUC, EER, ACC, balanced ACC, and why DFD fixed-threshold ACC is misleading.

## Verification plan

After implementation, run:

```bash
PYTHONPATH=src python -m py_compile train.py evaluate.py scripts/evaluate_norm_correction.py
PYTHONPATH=src python evaluate.py --help
find experiments -maxdepth 1 -type d | sort
```

Run retained tests if they still apply:

```bash
PYTHONPATH=src pytest tests/configs tests/data tests/engine -q
```

## Risks

This cleanup intentionally deletes old artifacts. Before deletion, list the candidate paths and preserve a backup if needed. The paper-used timestamp folders and DFD threshold CSVs must not be deleted.
