from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ExternalFrameRecord:
    frame_path: str
    label: int
    video_id: str

    @property
    def sample_id(self) -> str:
        return f"{self.frame_path}::{self.label}::{self.video_id}"


def stable_video_seed(base_seed: int, dataset_name: str, class_name: str, video_name: str) -> int:
    key = f"{base_seed}:{dataset_name}:{class_name}:{video_name}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def list_image_frames(video_dir: Path) -> list[Path]:
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_seeded_frames(frames: list[Path], frames_per_video: int, seed: int) -> list[Path]:
    if frames_per_video <= 0:
        return frames
    if len(frames) <= frames_per_video:
        return frames
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(frames)), frames_per_video))
    return [frames[index] for index in indices]


def sample_seeded_image_frames(video_dir: Path, frames_per_video: int, seed: int) -> list[Path]:
    frames = list_image_frames(video_dir)
    return sample_seeded_frames(frames, frames_per_video, seed)


def index_labeled_video_root(
    root: str,
    dataset_name: str,
    class_name: str,
    label: int,
    frames_per_video: int,
    seed: int,
) -> list[ExternalFrameRecord]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Missing external dataset root: {root}")

    records = []
    for video_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        video_seed = stable_video_seed(seed, dataset_name, class_name, video_dir.name)
        for frame_path in sample_seeded_image_frames(video_dir, frames_per_video, video_seed):
            records.append(ExternalFrameRecord(
                frame_path=str(frame_path),
                label=label,
                video_id=f"{dataset_name}/{class_name}/{video_dir.name}",
            ))
    return records


def index_binary_external_dataset(
    dataset_name: str,
    real_root: str,
    fake_root: str,
    frames_per_video: int,
    seed: int,
) -> list[ExternalFrameRecord]:
    real_records = index_labeled_video_root(real_root, dataset_name, "real", 0, frames_per_video, seed)
    fake_records = index_labeled_video_root(fake_root, dataset_name, "fake", 1, frames_per_video, seed)
    return real_records + fake_records


def write_external_index_cache(records: list[ExternalFrameRecord], cache_path: str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([record.__dict__ for record in records], f)


def load_external_index_cache(cache_path: str) -> list[ExternalFrameRecord]:
    with open(cache_path) as f:
        rows = json.load(f)
    return [ExternalFrameRecord(**row) for row in rows]
