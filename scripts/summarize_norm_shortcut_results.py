from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


TRAIN_METHOD_ALIASES = {
    "norm_shortcut_raw": "direction_plus_magnitude",
    "norm_shortcut_normtrain": "direction",
}

VARIANT_METHOD_ALIASES = {
    "raw": "direction_plus_magnitude",
    "feature_norm": "direction",
    "partial_norm": "partial_norm",
    "mean_scale_norm": "length_mean_scale",
    "raw_norm_score_fusion": "score_fusion",
}


def infer_seed_from_name(name: str) -> int | None:
    parts = name.split("_s")
    if len(parts) < 2:
        return None
    tail = parts[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def latest_run_dirs(root: Path, pattern: str) -> list[Path]:
    latest = []
    for exp_dir in sorted(root.glob(pattern)):
        run_dirs = sorted(path for path in exp_dir.iterdir() if path.is_dir())
        if run_dirs:
            latest.append(run_dirs[-1])
    return latest


def read_training_results(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in latest_run_dirs(root, "norm_shortcut_*_s*"):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = pd.read_json(meta_path, typ="series")
        exp_name = run_dir.parent.name
        train_key = "norm_shortcut_normtrain" if "normtrain" in exp_name else "norm_shortcut_raw"
        seed = infer_seed_from_name(exp_name)
        for split, prefix in (("ffpp", "ffpp_test"), ("cdf", "cdf_test"), ("dfd", "dfd_test")):
            auc_key = f"{prefix}_auc"
            if auc_key not in meta:
                continue
            rows.append({
                "source": "train",
                "experiment_name": exp_name,
                "seed": seed,
                "split": split,
                "variant": "feature_norm" if train_key.endswith("normtrain") else "raw",
                "method": TRAIN_METHOD_ALIASES[train_key],
                "param": 1.0 if train_key.endswith("normtrain") else 0.0,
                "auc": float(meta[auc_key]),
                "eer": float(meta[f"{prefix}_eer"]),
                "acc_0.5": float(meta[f"{prefix}_acc"]),
                "val_threshold": None,
                "val_acc_at_threshold": None,
                "acc_val_threshold": None,
                "rows_before_dedup": None,
                "rows_after_dedup": None,
                "unique_sample_ids_before": None,
                "unique_image_paths_before": None,
                "unique_videos_after": None,
                "source_file": str(meta_path),
            })
    return pd.DataFrame(rows)


def read_eval_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for path in root.glob("**/metrics.csv"):
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_eval_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["seed"] = df["source_file"].map(lambda p: infer_seed_from_name(Path(p).parts[-3]))
    df["experiment_name"] = df["source_file"].map(lambda p: Path(p).parts[-3])
    df["method"] = df["variant"].map(VARIANT_METHOD_ALIASES).fillna(df["variant"])
    df["source"] = "eval"
    return df


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols)
    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped


def paired_direction_gain(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["method"].isin(["direction", "direction_plus_magnitude"])]
    if main.empty:
        return pd.DataFrame(columns=["seed", "split", "direction", "direction_plus_magnitude", "auc_gain_direction_minus_raw"])
    pivot = main.pivot_table(index=["seed", "split"], columns="method", values="auc", aggfunc="first").reset_index()
    if "direction" in pivot and "direction_plus_magnitude" in pivot:
        pivot["auc_gain_direction_minus_raw"] = pivot["direction"] - pivot["direction_plus_magnitude"]
    return pivot


def split_eval_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    angle = df[df["method"].isin(["direction", "direction_plus_magnitude", "partial_norm", "score_fusion"])].copy()
    length = df[df["method"].isin(["length_mean_scale"])].copy()
    return angle, length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="experiments")
    parser.add_argument("--eval-root", default="experiments/paper_eval")
    parser.add_argument("--output-dir", default="experiments/norm_shortcut_tables")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_training_results(train_root)
    eval_df = add_eval_metadata(read_eval_metrics(eval_root))

    all_df = pd.concat([df for df in (train_df, eval_df) if not df.empty], ignore_index=True)
    if all_df.empty:
        raise SystemExit("No training or evaluation metrics found.")
    all_df.to_csv(output_dir / "all_metrics.csv", index=False)

    metric_cols = ["auc", "eer", "acc_0.5"]
    if "acc_val_threshold" in all_df.columns:
        metric_cols.append("acc_val_threshold")

    main_df = all_df[all_df["method"].isin(["direction", "direction_plus_magnitude"])].copy()
    normtrain_df = train_df[train_df["method"] == "direction"].copy()
    if not normtrain_df.empty:
        normtrain_df["method"] = "norm_train_norm_test"
        main_df = pd.concat([main_df, normtrain_df], ignore_index=True)
    summarize_mean_std(main_df, ["split", "method"], metric_cols).to_csv(output_dir / "main_results.csv", index=False)

    ablation_df = all_df[all_df["method"].isin(["partial_norm", "score_fusion", "length_mean_scale"])].copy()
    summarize_mean_std(ablation_df, ["split", "method", "param"], metric_cols).to_csv(output_dir / "ablation_results.csv", index=False)

    probe_files = sorted(train_root.glob("norm_only_probe/*/metrics.csv"))
    if probe_files:
        pd.concat([pd.read_csv(path) for path in probe_files], ignore_index=True).to_csv(output_dir / "probe_results.csv", index=False)

    paired_direction_gain(main_df).to_csv(output_dir / "paired_comparison.csv", index=False)

    angle_df, length_df = split_eval_tables(all_df)
    summarize_mean_std(angle_df, ["split", "method", "param"], metric_cols).to_csv(output_dir / "angle_results.csv", index=False)
    summarize_mean_std(length_df, ["split", "method", "param"], metric_cols).to_csv(output_dir / "length_results.csv", index=False)

    print(output_dir)


if __name__ == "__main__":
    main()
