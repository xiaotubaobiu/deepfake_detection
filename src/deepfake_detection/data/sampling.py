from __future__ import annotations

import random
from math import floor


def sample_uniform_frame_indices(num_frames: int, num_samples: int = 8) -> list[int]:
    if num_frames < 1:
        return []
    if num_samples <= 1:
        return [0]
    num_samples = min(num_samples, num_frames)
    return [floor(i * (num_frames - 1) / (num_samples - 1)) for i in range(num_samples)]


def balance_real_video_ids(real_video_ids: list[str], target_count: int, seed: int) -> list[str]:
    if not real_video_ids:
        raise ValueError("real_video_ids must not be empty")
    rng = random.Random(seed)
    if len(real_video_ids) >= target_count:
        return real_video_ids[:target_count]
    padded = list(real_video_ids)
    while len(padded) < target_count:
        padded.append(rng.choice(real_video_ids))
    return padded
