from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    method: str
    pair_id: str
    frame_name: str
    frame_path: str
    landmark_path: str | None
    label: int


def build_aligned_pair_key(method: str, pair_id: str, frame_name: str) -> str:
    frame_id = Path(frame_name).stem
    return f"{method}::{pair_id}::{frame_id}"


def index_ffpp_method_frames(ffpp_root: str, method: str, split: str, max_videos: int | None = None) -> list[FrameRecord]:
    split_dir = Path(ffpp_root) / "FF++_c23_32" / split / method
    if not split_dir.is_dir():
        return []
    records = []
    video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    for video_dir in video_dirs:
        pair_id = video_dir.name
        frames = sorted(video_dir.glob("frame_*.jpg"))
        for frame_path in frames:
            records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=None,
                label=1,
            ))
    return records


def index_ffpp_real_frames(ffpp_root: str, split: str) -> list[FrameRecord]:
    return index_ffpp_method_frames(ffpp_root, "original", split)


def index_ffpp_aligned_triplets(ffpp_root: str, method: str, split: str, max_videos: int | None = None) -> list[dict]:
    fake_records = index_ffpp_method_frames(ffpp_root, method, split, max_videos)
    real_index: dict[str, FrameRecord] = {}
    for r in index_ffpp_real_frames(ffpp_root, split):
        real_index[(r.pair_id, r.frame_name)] = r
    triplets = []
    for fake in fake_records:
        key = (fake.pair_id, fake.frame_name)
        real = real_index.get(key)
        if real is not None:
            triplets.append({
                "method": method,
                "pair_id": fake.pair_id,
                "frame_name": fake.frame_name,
                "real_frame_path": real.frame_path,
                "fake_frame_path": fake.frame_path,
                "real_landmark_path": real.landmark_path,
                "fake_landmark_path": fake.landmark_path,
            })
    return triplets
