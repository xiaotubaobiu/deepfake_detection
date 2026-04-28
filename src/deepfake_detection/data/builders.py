from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader, DistributedSampler

from deepfake_detection.data.constants import ALL_METHODS
from deepfake_detection.data.index_ffpp import (
    index_ffpp_method_frames,
    index_ffpp_real_frames,
    index_ffpp_aligned_triplets,
)
from deepfake_detection.data.index_cdf import index_cdf_frames, filter_cdf_records_by_methods
from deepfake_detection.data.sampling import balance_real_video_ids, sample_uniform_frame_indices
from deepfake_detection.data.datasets import FrameClassificationDataset, AlignedTripletDataset


def build_loader_kwargs(batch_size: int, num_workers: int, distributed: bool) -> dict:
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "distributed": distributed,
    }


def _subsample_records_to_frames(records, frames_per_video=8, seed=42):
    from collections import defaultdict
    import random
    rng = random.Random(seed)
    by_video = defaultdict(list)
    for r in records:
        by_video[r.pair_id].append(r)
    result = []
    for pair_id, frames in by_video.items():
        indices = sample_uniform_frame_indices(len(frames), min(frames_per_video, len(frames)))
        for i in indices:
            result.append(frames[i])
    return result


def build_train_loader(cfg, distributed=True):
    from deepfake_detection.data.constants import ALL_METHODS
    ffpp_root = cfg["dataset"]["ffpp_root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("batch_size", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    require_triplets = cfg.get("data", {}).get("require_aligned_triplets", False)
    if require_triplets:
        all_triplets = []
        for method in methods:
            triplets = index_ffpp_aligned_triplets(ffpp_root, method, "train", max_videos)
            all_triplets.extend(triplets)
        dataset = AlignedTripletDataset(all_triplets, augment=True)
    else:
        all_records = []
        for method in methods:
            fake_records = index_ffpp_method_frames(ffpp_root, method, "train", max_videos)
            real_records = index_ffpp_real_frames(ffpp_root, "train")
            real_ids = list({r.pair_id for r in real_records})
            balanced_real_ids = balance_real_video_ids(real_ids, len(fake_records), seed=42)
            real_by_id = {}
            for r in real_records:
                real_by_id.setdefault(r.pair_id, []).append(r)
            for vid_id in balanced_real_ids:
                for r in real_by_id.get(vid_id, []):
                    rec = type(r)(method=method, pair_id=r.pair_id, frame_name=r.frame_name,
                                  frame_path=r.frame_path, landmark_path=r.landmark_path, label=0)
                    all_records.append(rec)
            all_records.extend(fake_records)
        all_records = _subsample_records_to_frames(all_records, frames_per_video)
        dataset = FrameClassificationDataset(all_records, augment=True)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True)


def build_eval_loader(cfg, domain="ffpp", distributed=True):
    cdf_root = cfg["dataset"]["cdf_root"]
    ffpp_root = cfg["dataset"]["ffpp_root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    batch_size = cfg["train"].get("batch_size", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    if domain == "cdf":
        records = index_cdf_frames(cdf_root, methods)
    else:
        all_records = []
        for method in methods:
            all_records.extend(index_ffpp_method_frames(ffpp_root, method, "test"))
            all_records.extend(index_ffpp_real_frames(ffpp_root, "test"))
        records = all_records
    dataset = FrameClassificationDataset(records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
