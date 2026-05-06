from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepfake_detection.data.builders import build_eval_loader, build_external_eval_loader, build_val_loader
from deepfake_detection.data.index_external import (
    index_binary_external_dataset,
    load_external_index_cache,
    write_external_index_cache,
)
from deepfake_detection.engine.ddp import init_ddp, is_main_process
from deepfake_detection.engine.metrics import (
    compute_acc,
    compute_auc,
    compute_eer,
)
from deepfake_detection.evaluation.sample_eval import (
    build_sample_rows,
    deduplicate_sample_rows,
    gather_rows_across_ranks,
    sample_rows_to_video_predictions,
    summarize_sample_rows,
    write_sample_rows_csv,
)
from deepfake_detection.evaluation.norm_correction import (
    apply_mean_scale_norm,
    apply_partial_norm,
    best_accuracy_threshold,
    fuse_scores,
)
from deepfake_detection.models.factory import build_model
from train import load_config


@torch.no_grad()
def mean_feature_norm(model, loader, device, log_prefix: str, log_every: int):
    model.eval()
    norm_sum = torch.tensor(0.0, dtype=torch.float64)
    count = torch.tensor(0.0, dtype=torch.float64)
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        features = model.visual(images.float()).float()
        norm_sum += features.norm(dim=-1).double().sum().cpu()
        count += features.shape[0]
        if log_every > 0 and step % log_every == 0 and is_main_process():
            elapsed = max(time.time() - start, 1e-6)
            print(f"[{log_prefix}] step={step} rows_rank0={int(count.item())} rows_per_sec_rank0={count.item() / elapsed:.1f}", flush=True)
    if torch.distributed.is_initialized():
        packed = torch.stack([norm_sum, count]).to(device)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        norm_sum, count = packed.cpu()
    return float((norm_sum / count.clamp_min(1)).item())


def score_from_features(features, weight, bias, mode: str, alpha: float = 1.0, mean_scale: float = 1.0):
    if mode == "partial":
        corrected = apply_partial_norm(features, alpha)
    elif mode == "mean_scale":
        corrected = apply_mean_scale_norm(features, mean_scale)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    logits = corrected @ weight.t().to(features.device) + bias.to(features.device)
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def collect_sample_score_rows(model, loader, device, weight, bias, scorers: dict[str, dict], log_prefix: str, log_every: int):
    model.eval()
    rows_by_scorer = {name: [] for name in scorers}
    start = time.time()
    rows_seen = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        features = model.visual(images.float()).float()
        raw_scores = None
        norm_scores = None
        for name, scorer in scorers.items():
            if scorer["mode"] == "fusion":
                if raw_scores is None:
                    raw_scores = score_from_features(features, weight, bias, "partial", alpha=0.0)
                if norm_scores is None:
                    norm_scores = score_from_features(features, weight, bias, "partial", alpha=1.0)
                scores = fuse_scores(raw_scores, norm_scores, scorer["beta"])
                clamped_scores = scores.clamp(1e-12, 1 - 1e-12)
                logits = torch.stack([1 - clamped_scores, clamped_scores], dim=1).log()
            elif scorer["mode"] == "partial":
                corrected = apply_partial_norm(features, scorer["alpha"])
                logits = corrected @ weight.t().to(features.device) + bias.to(features.device)
                scores = torch.softmax(logits, dim=1)[:, 1]
            elif scorer["mode"] == "mean_scale":
                corrected = apply_mean_scale_norm(features, scorer["mean_scale"])
                logits = corrected @ weight.t().to(features.device) + bias.to(features.device)
                scores = torch.softmax(logits, dim=1)[:, 1]
            elif scorer["mode"] == "feature_length":
                scores = features.norm(dim=-1).float()
                logits = torch.stack([-scores, scores], dim=1)
            else:
                raise ValueError(f"Unknown scorer mode: {scorer['mode']}")
            rows_by_scorer[name].extend(build_sample_rows(batch, logits, scores, features.norm(dim=-1)))
        rows_seen += features.shape[0]
        if log_every > 0 and step % log_every == 0 and is_main_process():
            elapsed = max(time.time() - start, 1e-6)
            print(f"[{log_prefix}] step={step} rows_rank0={rows_seen} rows_per_sec_rank0={rows_seen / elapsed:.1f}", flush=True)

    if torch.distributed.is_initialized():
        rows_by_scorer = {name: gather_rows_across_ranks(rows) for name, rows in rows_by_scorer.items()}
    return rows_by_scorer


def video_metrics(video_labels, video_scores, threshold=0.5):
    if len(set(video_labels)) < 2:
        return {"auc": 0.0, "eer": 0.0, "acc": compute_acc(video_labels, video_scores, threshold)}, torch.tensor(video_labels, dtype=torch.long), torch.tensor(video_scores, dtype=torch.float32)
    return {
        "auc": compute_auc(video_labels, video_scores),
        "eer": compute_eer(video_labels, video_scores),
        "acc": compute_acc(video_labels, video_scores, threshold),
    }, torch.tensor(video_labels, dtype=torch.long), torch.tensor(video_scores, dtype=torch.float32)


def parse_floats(value: str):
    return [float(item) for item in value.split(",") if item]


def parse_names(value: str):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def external_dataset_specs(args):
    return {
        "faceshifter": {
            "real_root": args.faceshifter_real_root,
            "fake_root": args.faceshifter_fake_root,
        },
        "dfd": {
            "real_root": args.dfd_real_root,
            "fake_root": args.dfd_fake_root,
        },
    }


def external_cache_path(args, dataset_name: str) -> str | None:
    if not args.external_index_cache_dir:
        return None
    filename = f"{dataset_name}_frames{args.external_frames_per_video}_seed{args.external_seed}.json"
    return os.path.join(args.external_index_cache_dir, filename)


def load_or_build_external_records(args, dataset_name: str, spec: dict) -> list:
    cache_path = external_cache_path(args, dataset_name)
    if cache_path and os.path.exists(cache_path):
        return load_external_index_cache(cache_path)
    records = index_binary_external_dataset(
        dataset_name,
        spec["real_root"],
        spec["fake_root"],
        args.external_frames_per_video,
        args.external_seed,
    )
    if cache_path:
        write_external_index_cache(records, cache_path)
    return records


def variant_name_for_partial_alpha(alpha: float) -> str:
    if alpha == 0:
        return "raw"
    if alpha == 1:
        return "feature_norm"
    return "partial_norm"


def variant_name(scorer_name: str) -> str:
    if scorer_name == "feature_length":
        return "feature_length"
    if scorer_name.startswith("partial:"):
        return variant_name_for_partial_alpha(float(scorer_name.split(":", 1)[1]))
    return scorer_name


def maybe_write_rows(args, rows: list[dict], split: str, scorer_name: str) -> None:
    if args.write_sample_rows and is_main_process():
        write_sample_rows_csv(deduplicate_sample_rows(rows), f"{args.run_dir}/sample_rows/{split}_{scorer_name}_sample_rows.csv")


def write_results(args, result_rows, meta):
    os.makedirs(args.run_dir, exist_ok=True)
    csv_path = os.path.join(args.run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary_path = os.path.join(args.run_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        for line in meta["summary_lines"]:
            f.write(line + "\n")
    with open(os.path.join(args.run_dir, "run_meta.json"), "w") as f:
        json.dump(meta["json"], f, indent=2)
    print(f"Results written to {summary_path}", flush=True)
    return summary_path


def maybe_destroy_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def run_external_evaluation(args, cfg, model, device, weight, bias, distributed):
    scorers = {
        "raw": {"mode": "partial", "alpha": 0.0},
        "feature_norm": {"mode": "partial", "alpha": 1.0},
    }
    specs = external_dataset_specs(args)
    result_rows = []
    selected = parse_names(args.external_datasets)
    for dataset_name in selected:
        if dataset_name not in specs:
            raise ValueError(f"Unknown external dataset: {dataset_name}")
        spec = specs[dataset_name]
        records = load_or_build_external_records(args, dataset_name, spec)
        if is_main_process():
            real_count = sum(1 for record in records if record.label == 0)
            fake_count = sum(1 for record in records if record.label == 1)
            print(f"[{dataset_name}] records real={real_count} fake={fake_count} frames_per_video={args.external_frames_per_video}", flush=True)
        rows_by_scorer = collect_sample_score_rows(
            model,
            build_external_eval_loader(cfg, records, distributed=distributed, batch_size=args.batch_size, num_workers=args.num_workers),
            device,
            weight,
            bias,
            scorers,
            dataset_name,
            args.log_every,
        )
        predictions = {name: sample_rows_to_video_predictions(deduplicate_sample_rows(rows)) for name, rows in rows_by_scorer.items()}
        if is_main_process():
            for scorer_name in ("raw", "feature_norm"):
                labels, scores = predictions[scorer_name]
                metrics_05, _, _ = video_metrics(labels, scores, threshold=0.5)
                summary = summarize_sample_rows(rows_by_scorer[scorer_name])
                result_rows.append({
                    "variant": scorer_name,
                    "param": scorers[scorer_name]["alpha"],
                    "split": dataset_name,
                    "auc": metrics_05["auc"],
                    "eer": metrics_05["eer"],
                    "acc_0.5": metrics_05["acc"],
                    "val_threshold": "",
                    "val_acc_at_threshold": "",
                    "acc_val_threshold": "",
                    **summary,
                })
                maybe_write_rows(args, rows_by_scorer[scorer_name], dataset_name, scorer_name)
        del predictions

    if not is_main_process():
        maybe_destroy_process_group()
        return

    meta = {
        "summary_lines": [
            "Exp: external_norm_correction",
            f"Checkpoint: {args.checkpoint}",
            f"Batch size per GPU: {args.batch_size}",
            f"Num workers per rank: {args.num_workers}",
            f"External frames per video: {args.external_frames_per_video}",
            f"External seed: {args.external_seed}",
            *[
                f"{row['split']} {row['variant']}: AUC={row['auc']:.4f} EER={row['eer']:.4f} ACC@0.5={row['acc_0.5']:.4f}"
                for row in result_rows
            ],
        ],
        "json": {
            "checkpoint": args.checkpoint,
            "external_frames_per_video": args.external_frames_per_video,
            "external_seed": args.external_seed,
            "results": result_rows,
        },
    }
    write_results(args, result_rows, meta)
    print("External results:", flush=True)
    for row in result_rows:
        print(row, flush=True)
    maybe_destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output-dir", default="experiments/paper_eval")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--external-datasets", default="dfd")
    parser.add_argument("--external-frames-per-video", type=int, default=32)
    parser.add_argument("--external-seed", type=int, default=42)
    parser.add_argument("--external-index-cache-dir", default="experiments/external_index_cache")
    parser.add_argument("--write-sample-rows", action="store_true")
    parser.add_argument("--splits", default="ffpp,cdf")
    parser.add_argument("--mode", default="raw,feature_norm,partial,mean_scale,fusion")
    parser.add_argument("--ffpp-val-threshold", type=float, default=None)
    parser.add_argument("--faceshifter-real-root", default="/Dataset/deepfake_detection/FaceForensics++/c23/test/original")
    parser.add_argument("--faceshifter-fake-root", default="/Dataset/deepfake_detection/FaceForensics++/c23/test/FaceShifter")
    parser.add_argument("--dfd-real-root", default="/Dataset/deepfake_detection/FaceForensics++/original_sequences/actors/c23/frames")
    parser.add_argument("--dfd-fake-root", default="/Dataset/deepfake_detection/FaceForensics++/manipulated_sequences/DeepFakeDetection/c23/frames")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg["experiment_name"] = "exp3_norm_correction"
    cfg["train"]["per_gpu_batch"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers
    cfg["train"]["prefetch_factor"] = args.prefetch_factor

    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if is_main_process():
        print(f"DataLoader: batch_size_per_gpu={args.batch_size} num_workers_per_rank={args.num_workers} prefetch_factor={args.prefetch_factor}", flush=True)

    distributed = torch.distributed.is_initialized()
    weight = model.classifier.weight.detach().cpu().float()
    bias = model.classifier.bias.detach().cpu().float()
    if args.external_only:
        run_external_evaluation(args, cfg, model, device, weight, bias, distributed)
        return

    alphas = parse_floats(args.alphas)
    modes = set(parse_names(args.mode))
    splits = parse_names(args.splits)
    if "raw" in modes and 0.0 not in alphas:
        alphas.insert(0, 0.0)
    if "feature_norm" in modes and 1.0 not in alphas:
        alphas.append(1.0)
    if "partial" not in modes:
        alphas = [alpha for alpha in alphas if (alpha == 0 and "raw" in modes) or (alpha == 1 and "feature_norm" in modes)]
    betas = parse_floats(args.betas)
    mean_scale = mean_feature_norm(model, build_val_loader(cfg, distributed=distributed), device, "val_mean_norm", args.log_every)

    val_scorers = {
        **{f"partial:{alpha}": {"mode": "partial", "alpha": alpha} for alpha in alphas},
        **({"mean_scale": {"mode": "mean_scale", "mean_scale": mean_scale}} if "mean_scale" in modes else {}),
        **({"feature_length": {"mode": "feature_length"}} if "feature_length" in modes else {}),
        **({f"fusion:{beta}": {"mode": "fusion", "beta": beta} for beta in betas} if "fusion" in modes else {}),
    }
    val_rows_by_scorer = collect_sample_score_rows(model, build_val_loader(cfg, distributed=distributed), device, weight, bias, val_scorers, "val_scores", args.log_every)
    val_predictions = {name: sample_rows_to_video_predictions(deduplicate_sample_rows(rows)) for name, rows in val_rows_by_scorer.items()}

    partial_thresholds = {}
    result_rows = []
    for alpha in alphas:
        labels, scores = val_predictions[f"partial:{alpha}"]
        if args.ffpp_val_threshold is None:
            threshold, val_acc = best_accuracy_threshold(torch.tensor(labels, dtype=torch.long), torch.tensor(scores, dtype=torch.float32))
        else:
            threshold = args.ffpp_val_threshold
            val_acc = compute_acc(labels, scores, threshold)
        partial_thresholds[alpha] = (threshold, val_acc)

    mean_scale_threshold = None
    mean_scale_val_acc = None
    if "mean_scale" in modes:
        labels, scores = val_predictions["mean_scale"]
        if args.ffpp_val_threshold is None:
            mean_scale_threshold, mean_scale_val_acc = best_accuracy_threshold(torch.tensor(labels, dtype=torch.long), torch.tensor(scores, dtype=torch.float32))
        else:
            mean_scale_threshold = args.ffpp_val_threshold
            mean_scale_val_acc = compute_acc(labels, scores, mean_scale_threshold)

    feature_length_threshold = None
    feature_length_val_acc = None
    if "feature_length" in modes:
        labels, scores = val_predictions["feature_length"]
        if args.ffpp_val_threshold is None:
            feature_length_threshold, feature_length_val_acc = best_accuracy_threshold(torch.tensor(labels, dtype=torch.long), torch.tensor(scores, dtype=torch.float32))
        else:
            feature_length_threshold = args.ffpp_val_threshold
            feature_length_val_acc = compute_acc(labels, scores, feature_length_threshold)

    fusion_thresholds = {}
    if "fusion" in modes:
        for beta in betas:
            labels, scores = val_predictions[f"fusion:{beta}"]
            if args.ffpp_val_threshold is None:
                threshold, val_acc = best_accuracy_threshold(torch.tensor(labels, dtype=torch.long), torch.tensor(scores, dtype=torch.float32))
            else:
                threshold = args.ffpp_val_threshold
                val_acc = compute_acc(labels, scores, threshold)
            fusion_thresholds[beta] = (threshold, val_acc)
    for scorer_name, rows in val_rows_by_scorer.items():
        maybe_write_rows(args, rows, "ffpp_val", scorer_name.replace(":", "_"))
    del val_predictions, val_rows_by_scorer

    best_cdf = None
    split_scorers = {
        **{f"partial:{alpha}": {"mode": "partial", "alpha": alpha} for alpha in alphas},
        **({"mean_scale": {"mode": "mean_scale", "mean_scale": mean_scale}} if "mean_scale" in modes else {}),
        **({"feature_length": {"mode": "feature_length"}} if "feature_length" in modes else {}),
        **({f"fusion:{beta}": {"mode": "fusion", "beta": beta} for beta in betas} if "fusion" in modes else {}),
    }
    for split in splits:
        predictions = None
        rows_by_scorer = collect_sample_score_rows(
            model,
            build_eval_loader(cfg, split, distributed=distributed),
            device,
            weight,
            bias,
            split_scorers,
            split,
            args.log_every,
        )
        predictions = {name: sample_rows_to_video_predictions(deduplicate_sample_rows(rows)) for name, rows in rows_by_scorer.items()}
        if not is_main_process():
            continue
        for alpha in alphas:
            threshold, val_acc = partial_thresholds[alpha]
            labels, scores = predictions[f"partial:{alpha}"]
            metrics_05, _, _ = video_metrics(labels, scores, threshold=0.5)
            metrics_cal, _, _ = video_metrics(labels, scores, threshold=threshold)
            row = {
                "variant": variant_name_for_partial_alpha(alpha),
                "param": alpha,
                "split": split,
                "auc": metrics_05["auc"],
                "eer": metrics_05["eer"],
                "acc_0.5": metrics_05["acc"],
                "val_threshold": threshold,
                "val_acc_at_threshold": val_acc,
                "acc_val_threshold": metrics_cal["acc"],
                **summarize_sample_rows(rows_by_scorer[f"partial:{alpha}"]),
            }
            result_rows.append(row)
            maybe_write_rows(args, rows_by_scorer[f"partial:{alpha}"], split, f"partial_{alpha}")
            if split == "cdf" and (best_cdf is None or row["auc"] > best_cdf["auc"]):
                best_cdf = row

        if "mean_scale" in modes:
            labels, scores = predictions["mean_scale"]
            metrics_05, _, _ = video_metrics(labels, scores, threshold=0.5)
            metrics_cal, _, _ = video_metrics(labels, scores, threshold=mean_scale_threshold)
            row = {
                "variant": "mean_scale_norm",
                "param": mean_scale,
                "split": split,
                "auc": metrics_05["auc"],
                "eer": metrics_05["eer"],
                "acc_0.5": metrics_05["acc"],
                "val_threshold": mean_scale_threshold,
                "val_acc_at_threshold": mean_scale_val_acc,
                "acc_val_threshold": metrics_cal["acc"],
                **summarize_sample_rows(rows_by_scorer["mean_scale"]),
            }
            result_rows.append(row)
            maybe_write_rows(args, rows_by_scorer["mean_scale"], split, "mean_scale")
            if split == "cdf" and (best_cdf is None or row["auc"] > best_cdf["auc"]):
                best_cdf = row

        if "feature_length" in modes:
            labels, scores = predictions["feature_length"]
            metrics_05, _, _ = video_metrics(labels, scores, threshold=0.5)
            metrics_cal, _, _ = video_metrics(labels, scores, threshold=feature_length_threshold)
            row = {
                "variant": "feature_length",
                "param": 0.0,
                "split": split,
                "auc": metrics_05["auc"],
                "eer": metrics_05["eer"],
                "acc_0.5": metrics_05["acc"],
                "val_threshold": feature_length_threshold,
                "val_acc_at_threshold": feature_length_val_acc,
                "acc_val_threshold": metrics_cal["acc"],
                **summarize_sample_rows(rows_by_scorer["feature_length"]),
            }
            result_rows.append(row)
            maybe_write_rows(args, rows_by_scorer["feature_length"], split, "feature_length")
            if split == "cdf" and (best_cdf is None or row["auc"] > best_cdf["auc"]):
                best_cdf = row

        if "fusion" in modes:
            for beta in betas:
                threshold, val_acc = fusion_thresholds[beta]
                labels, scores = predictions[f"fusion:{beta}"]
                metrics_05, _, _ = video_metrics(labels, scores, threshold=0.5)
                metrics_cal, _, _ = video_metrics(labels, scores, threshold=threshold)
                row = {
                    "variant": "raw_norm_score_fusion",
                    "param": beta,
                    "split": split,
                    "auc": metrics_05["auc"],
                    "eer": metrics_05["eer"],
                    "acc_0.5": metrics_05["acc"],
                    "val_threshold": threshold,
                    "val_acc_at_threshold": val_acc,
                    "acc_val_threshold": metrics_cal["acc"],
                    **summarize_sample_rows(rows_by_scorer[f"fusion:{beta}"]),
                }
                result_rows.append(row)
                maybe_write_rows(args, rows_by_scorer[f"fusion:{beta}"], split, f"fusion_{beta}")
                if split == "cdf" and (best_cdf is None or row["auc"] > best_cdf["auc"]):
                    best_cdf = row
        del predictions

    if not is_main_process():
        maybe_destroy_process_group()
        return

    meta = {
        "summary_lines": [
            "Exp: exp3_norm_correction",
            f"Checkpoint: {args.checkpoint}",
            f"Batch size per GPU: {args.batch_size}",
            f"Num workers per rank: {args.num_workers}",
            f"Mean feature norm from FF++ val: {mean_scale:.6f}",
            f"Best CDF: {best_cdf['variant']} param={best_cdf['param']} AUC={best_cdf['auc']:.4f} EER={best_cdf['eer']:.4f} ACC@0.5={best_cdf['acc_0.5']:.4f} ACC@val_thr={best_cdf['acc_val_threshold']:.4f}" if best_cdf else "Best CDF: not evaluated",
        ],
        "json": {"mean_scale": mean_scale, "best_cdf": best_cdf, "checkpoint": args.checkpoint},
    }
    summary_path = write_results(args, result_rows, meta)
    print(f"Best CDF: {best_cdf}", flush=True)
    maybe_destroy_process_group()


if __name__ == "__main__":
    main()
