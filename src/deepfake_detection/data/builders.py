from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, DistributedSampler

from deepfake_detection.data.constants import ALL_METHODS
from deepfake_detection.data.index_ffpp import (
    index_train_method_frames,
    index_train_real_frames,
    index_train_aligned_triplets,
    index_test_method_frames,
    index_test_real_frames,
    index_val_method_frames,
)
from deepfake_detection.data.sampling import sample_uniform_frame_indices
from deepfake_detection.data.datasets import FrameClassificationDataset, AlignedTripletDataset


def _subsample_records_to_frames(records, frames_per_video=8):
    from collections import defaultdict
    by_video: dict[str, list] = defaultdict(list)
    for r in records:
        by_video[r.pair_id].append(r)
    result = []
    for pair_id, frames in by_video.items():
        n = min(frames_per_video, len(frames))
        indices = sample_uniform_frame_indices(len(frames), n)
        for i in indices:
            result.append(frames[i])
    return result


def build_train_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    require_triplets = cfg.get("data", {}).get("require_aligned_triplets", False)

    if require_triplets:
        all_triplets = []
        for method in methods:
            triplets = index_train_aligned_triplets(root, method, max_videos)
            all_triplets.extend(triplets)
        dataset = AlignedTripletDataset(all_triplets, augment=True)
    else:
        # Collect fake records: 200 videos per method, subsample to 8 frames each
        fake_records = []
        for method in methods:
            recs = index_train_method_frames(root, method, max_videos)
            fake_records.extend(recs)
        fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

        # Collect real records: oversample to match total fake count
        real_records = index_train_real_frames(root)
        real_by_video_full = _subsample_records_to_frames(real_records, frames_per_video)
        target_real_count = len(fake_by_video)
        rng = random.Random(42)
        real_balanced = []
        while len(real_balanced) < target_real_count:
            real_balanced.extend(rng.sample(real_by_video_full, min(len(real_by_video_full), target_real_count - len(real_balanced))))

        all_records = real_balanced[:target_real_count] + fake_by_video
        dataset = FrameClassificationDataset(all_records, augment=True)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True)


def build_eval_loader(cfg, domain="ffpp", distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)

    all_records = list(index_test_real_frames(root))
    for method in methods:
        test_domain = "ff" if domain == "ffpp" else "cdf"
        all_records.extend(index_test_method_frames(root, method, test_domain))

    dataset = FrameClassificationDataset(all_records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)


def build_val_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    val_videos = cfg["dataset"].get("val_videos_per_method", 100)
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)

    fake_records = []
    for method in methods:
        recs = index_val_method_frames(root, method, val_videos)
        fake_records.extend(recs)
    fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

    real_records = index_train_real_frames(root)
    real_by_video_full = _subsample_records_to_frames(real_records, frames_per_video)
    target_real_count = len(fake_by_video)
    rng = random.Random(42)
    real_balanced = []
    while len(real_balanced) < target_real_count:
        real_balanced.extend(rng.sample(real_by_video_full, min(len(real_by_video_full), target_real_count - len(real_balanced))))

    all_records = real_balanced[:target_real_count] + fake_by_video
    dataset = FrameClassificationDataset(all_records, augment=False)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
