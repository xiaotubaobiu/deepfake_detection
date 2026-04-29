from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, DistributedSampler

from deepfake_detection.data.constants import ALL_METHODS
from deepfake_detection.data.index_ffpp import load_json_index
from deepfake_detection.data.index_ffpp import (
    index_train_aligned_triplets,
)
from deepfake_detection.data.sampling import sample_uniform_frame_indices
from deepfake_detection.data.datasets import FrameClassificationDataset, AlignedTripletDataset


def _get_normalization(cfg):
    model_name = cfg.get("model", {}).get("name", "")
    if "clip" in model_name:
        from deepfake_detection.data.transforms import CLIP_MEAN, CLIP_STD
        return CLIP_MEAN, CLIP_STD
    return None, None


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


def _deduplicate_videos(records):
    """Keep all frames for each unique pair_id from first-encountered method."""
    seen = set()
    claimed_by = {}
    result = []
    for r in records:
        if r.pair_id not in seen:
            seen.add(r.pair_id)
            claimed_by[r.pair_id] = r.method
        if claimed_by.get(r.pair_id) == r.method:
            result.append(r)
    return result


def build_train_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    frames_per_video = cfg["dataset"]["frames_per_video"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    require_triplets = cfg.get("data", {}).get("require_aligned_triplets", False)
    norm_mean, norm_std = _get_normalization(cfg)

    if require_triplets:
        all_triplets = []
        for method in methods:
            triplets = index_train_aligned_triplets(root, method, max_videos)
            all_triplets.extend(triplets)
        dataset = AlignedTripletDataset(all_triplets, augment=True)
    else:
        # Collect fake records from JSON: first max_videos per method
        fake_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 1, max_videos=max_videos)
            fake_records.extend(recs)
        fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

        # Collect real records from JSON: FF++ original, aligned range
        real_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 0, max_videos=max_videos)
            real_records.extend(recs)
        real_deduped = _deduplicate_videos(real_records)
        real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
        target_real_count = len(fake_by_video)
        rng = random.Random(42)
        real_balanced = []
        while len(real_balanced) < target_real_count:
            real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

        all_records = real_balanced[:target_real_count] + fake_by_video
        dataset = FrameClassificationDataset(all_records, augment=True,
                                            normalize_mean=norm_mean, normalize_std=norm_std)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True)


def build_eval_loader(cfg, domain="ffpp", distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    norm_mean, norm_std = _get_normalization(cfg)

    test_domain = "ff" if domain == "ffpp" else "cdf"

    # Real: from JSON test split (CDF JSON uses Celeb-DF-v2 real)
    real_records = []
    for method in methods:
        recs = load_json_index(root, method, test_domain, "test", 0)
        real_records.extend(recs)
    real_deduped = _deduplicate_videos(real_records)

    # Fake: from JSON test split
    fake_records = []
    for method in methods:
        recs = load_json_index(root, method, test_domain, "test", 1)
        fake_records.extend(recs)

    all_records = real_deduped + fake_records
    dataset = FrameClassificationDataset(all_records, augment=False,
                                        normalize_mean=norm_mean, normalize_std=norm_std)
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
    norm_mean, norm_std = _get_normalization(cfg)

    # Val fake: videos [max_videos : max_videos + val_videos] per method
    max_videos = cfg["dataset"]["train_videos_per_method"]
    fake_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 1,
                               video_range=(max_videos, max_videos + val_videos))
        fake_records.extend(recs)
    fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

    # Val real: FF++ original, same range — NO overlap with training real
    real_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 0,
                               video_range=(max_videos, max_videos + val_videos))
        real_records.extend(recs)
    real_deduped = _deduplicate_videos(real_records)
    real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
    target_real_count = len(fake_by_video)
    rng = random.Random(42)
    real_balanced = []
    while len(real_balanced) < target_real_count:
        real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

    all_records = real_balanced[:target_real_count] + fake_by_video
    dataset = FrameClassificationDataset(all_records, augment=False,
                                        normalize_mean=norm_mean, normalize_std=norm_std)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=num_workers, pin_memory=True)
