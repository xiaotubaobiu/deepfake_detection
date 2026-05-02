
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from deepfake_detection.data.builders import build_eval_loader, build_val_loader
from deepfake_detection.engine.ddp import barrier, init_ddp, is_main_process
from deepfake_detection.engine.trainers import run_eval_epoch
from deepfake_detection.models.factory import build_model
from train import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/analysis/rawcls_normprompt_alpha_sweep.csv")
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.7,1")
    args = parser.parse_args()

    seeds = [42, 123, 7, 999, 2048]
    alphas = [float(x) for x in args.alphas.split(",")]
    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = torch.distributed.is_initialized()

    rows = []
    for seed in seeds:
        config_path = f"configs/exp3_rawcls_normprompt_fullffpp_ema_s{seed}.yaml"
        cfg = load_config(config_path)
        run_dirs = sorted(Path(f"outputs/exp3_rawcls_normprompt_fullffpp_ema_s{seed}").glob("*/best_model.pth"))
        if not run_dirs:
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}")
        ckpt_path = run_dirs[-1]
        model = build_model(cfg["model"]).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loader = build_val_loader(cfg, distributed=distributed)
        ffpp_loader = build_eval_loader(cfg, domain="ffpp", distributed=distributed)
        cdf_loader = build_eval_loader(cfg, domain="cdf", distributed=distributed)

        for alpha in alphas:
            eval_cfg = dict(cfg)
            eval_cfg["loss"] = dict(cfg.get("loss", {}))
            eval_cfg["loss"]["alpha"] = alpha
            val = run_eval_epoch(model, val_loader, device, eval_cfg)
            ffpp = run_eval_epoch(model, ffpp_loader, device, eval_cfg)
            cdf = run_eval_epoch(model, cdf_loader, device, eval_cfg)
            if is_main_process():
                rows.append({
                    "seed": seed,
                    "alpha": alpha,
                    "checkpoint": str(ckpt_path),
                    "val_auc": val["auc"], "val_eer": val["eer"], "val_acc": val["acc"],
                    "ffpp_auc": ffpp["auc"], "ffpp_eer": ffpp["eer"], "ffpp_acc": ffpp["acc"],
                    "cdf_auc": cdf["auc"], "cdf_eer": cdf["eer"], "cdf_acc": cdf["acc"],
                })
        barrier()

    if is_main_process():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
