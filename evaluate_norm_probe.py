from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from deepfake_detection.data.builders import build_eval_loader, build_val_loader
from deepfake_detection.engine.ddp import init_ddp, is_main_process
from deepfake_detection.engine.metrics import merge_video_aggregates, update_video_aggregate, video_aggregates_to_predictions
from deepfake_detection.models.factory import build_model
from train import load_config


@torch.no_grad()
def collect_norm_predictions(model, loader, device, target_fn, log_prefix: str, log_every: int):
    model.eval()
    aggregates = {}
    rows_seen = 0
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        video_ids = batch["video_id"]
        features = model.visual(images.float()).float()
        norms = features.norm(dim=-1).detach().cpu()
        targets = target_fn(batch)
        for i in range(norms.shape[0]):
            video_id = video_ids[i] if isinstance(video_ids, list) else video_ids[i]
            update_video_aggregate(aggregates, video_id, float(norms[i]), int(targets[i]))
        rows_seen += features.shape[0]
        if log_every > 0 and step % log_every == 0 and is_main_process():
            elapsed = max(time.time() - start, 1e-6)
            print(f"[{log_prefix}] step={step} rows_rank0={rows_seen} rows_per_sec_rank0={rows_seen / elapsed:.1f}", flush=True)

    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, aggregates)
        merged = {}
        for rank_aggregates in gathered:
            merge_video_aggregates(merged, rank_aggregates)
        aggregates = merged
    labels, scores = video_aggregates_to_predictions(aggregates)
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def fit_probe(train_norms, train_labels):
    probe = LogisticRegression(class_weight="balanced", solver="lbfgs")
    probe.fit(train_norms.reshape(-1, 1), train_labels)
    return probe


def eval_probe(probe, norms, labels):
    probs = probe.predict_proba(norms.reshape(-1, 1))[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else 0.0,
        "acc": float(accuracy_score(labels, preds)),
        "coef": float(probe.coef_[0, 0]),
        "intercept": float(probe.intercept_[0]),
        "norm_mean_0": float(norms[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "norm_mean_1": float(norms[labels == 1].mean()) if np.any(labels == 1) else 0.0,
    }


def label_target(batch):
    return batch["label"].detach().cpu().numpy().astype(np.int64)


def zeros_target(batch):
    return np.zeros(len(batch["label"]), dtype=np.int64)


def ones_target(batch):
    return np.ones(len(batch["label"]), dtype=np.int64)


def write_outputs(args, rows, meta):
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(run_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        for row in rows:
            f.write(
                f"{row['task']} train={row['train_split']} eval={row['eval_split']} "
                f"AUC={row['auc']:.4f} ACC={row['acc']:.4f} "
                f"norm_mean_0={row['norm_mean_0']:.4f} norm_mean_1={row['norm_mean_1']:.4f}\n"
            )
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Results written to {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", default="experiments/outputs/norm_only_probe")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["per_gpu_batch"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers
    cfg["train"]["prefetch_factor"] = args.prefetch_factor

    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    distributed = torch.distributed.is_initialized()
    if is_main_process():
        print(f"Norm-only probe: batch_size_per_gpu={args.batch_size} num_workers_per_rank={args.num_workers}", flush=True)

    val_labels, val_norms = collect_norm_predictions(model, build_val_loader(cfg, distributed=distributed), device, label_target, "ffpp_val_real_fake", args.log_every)
    ffpp_labels, ffpp_norms = collect_norm_predictions(model, build_eval_loader(cfg, "ffpp", distributed=distributed), device, label_target, "ffpp_test_real_fake", args.log_every)
    cdf_labels, cdf_norms = collect_norm_predictions(model, build_eval_loader(cfg, "cdf", distributed=distributed), device, label_target, "cdf_real_fake", args.log_every)

    dfd_labels, dfd_norms = collect_norm_predictions(model, build_eval_loader(cfg, "dfd", distributed=distributed), device, label_target, "dfd_real_fake", args.log_every)

    val_domain_labels, val_domain_norms = collect_norm_predictions(model, build_val_loader(cfg, distributed=distributed), device, zeros_target, "ffpp_val_domain", args.log_every)
    cdf_domain_labels, cdf_domain_norms = collect_norm_predictions(model, build_eval_loader(cfg, "cdf", distributed=distributed), device, ones_target, "cdf_domain", args.log_every)

    if not is_main_process():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    rows = []
    real_fake_probe = fit_probe(val_norms, val_labels)
    for split, labels, norms in (("ffpp_val", val_labels, val_norms), ("ffpp_test", ffpp_labels, ffpp_norms), ("cdf", cdf_labels, cdf_norms), ("dfd", dfd_labels, dfd_norms)):
        metrics = eval_probe(real_fake_probe, norms, labels)
        rows.append({"task": "real_fake", "train_split": "ffpp_val", "eval_split": split, **metrics})

    domain_train_norms = np.concatenate([val_domain_norms, cdf_domain_norms])
    domain_train_labels = np.concatenate([val_domain_labels, cdf_domain_labels])
    domain_probe = fit_probe(domain_train_norms, domain_train_labels)
    domain_metrics = eval_probe(domain_probe, domain_train_norms, domain_train_labels)
    rows.append({"task": "ffpp_val_vs_cdf_domain", "train_split": "ffpp_val+cdf", "eval_split": "ffpp_val+cdf", **domain_metrics})

    write_outputs(args, rows, {"checkpoint": args.checkpoint, "config": args.config})
    print("Norm-only probe results:", flush=True)
    for row in rows:
        print(row, flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
