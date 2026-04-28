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


def index_train_method_frames(root: str, method: str, max_videos: int | None = None) -> list[FrameRecord]:
    frames_dir = Path(root) / "DF40_train" / method / "frames"
    if not frames_dir.is_dir():
        return []
    records = []
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    for video_dir in video_dirs:
        pair_id = video_dir.name
        lm_dir = Path(root) / "DF40_train" / method / "landmarks" / pair_id
        for frame_path in sorted(video_dir.glob("*.png")):
            lm_file = lm_dir / (frame_path.stem + ".npy") if lm_dir.is_dir() else None
            records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=str(lm_file) if lm_file and lm_file.exists() else None,
                label=1,
            ))
    return records


def index_val_method_frames(root: str, method: str, val_videos_per_method: int = 100) -> list[FrameRecord]:
    frames_dir = Path(root) / "DF40_train" / method / "frames"
    if not frames_dir.is_dir():
        return []
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    val_dirs = video_dirs[200:200 + val_videos_per_method]
    records = []
    for video_dir in val_dirs:
        pair_id = video_dir.name
        lm_dir = Path(root) / "DF40_train" / method / "landmarks" / pair_id
        for frame_path in sorted(video_dir.glob("*.png")):
            lm_file = lm_dir / (frame_path.stem + ".npy") if lm_dir.is_dir() else None
            records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=str(lm_file) if lm_file and lm_file.exists() else None,
                label=1,
            ))
    return records


def index_train_real_frames(root: str) -> list[FrameRecord]:
    anchor_dir = Path(root) / "DF40_train" / "anchor"
    if not anchor_dir.is_dir():
        return []
    records = []
    for video_dir in sorted(anchor_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        pair_id = video_dir.name
        for frame_path in sorted(video_dir.glob("*.png")):
            records.append(FrameRecord(
                method="real",
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=None,
                label=0,
            ))
    return records


def index_test_method_frames(root: str, method: str, domain: str) -> list[FrameRecord]:
    frames_dir = Path(root) / "DF40_test" / method / domain / "frames"
    if not frames_dir.is_dir():
        return []
    records = []
    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        pair_id = video_dir.name
        lm_dir = Path(root) / "DF40_test" / method / domain / "landmarks" / pair_id
        for frame_path in sorted(video_dir.glob("*.png")):
            lm_file = lm_dir / (frame_path.stem + ".npy") if lm_dir.is_dir() else None
            records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=str(lm_file) if lm_file and lm_file.exists() else None,
                label=1,
            ))
    return records


def index_test_real_frames(root: str) -> list[FrameRecord]:
    anchor_dir = Path(root) / "DF40_test" / "anchor"
    if not anchor_dir.is_dir():
        return []
    records = []
    for video_dir in sorted(anchor_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        pair_id = video_dir.name
        for frame_path in sorted(video_dir.glob("*.png")):
            records.append(FrameRecord(
                method="real",
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=None,
                label=0,
            ))
    return records


def index_train_aligned_triplets(root: str, method: str, max_videos: int | None = None) -> list[dict]:
    fake_records = index_train_method_frames(root, method, max_videos)
    real_index: dict[str, FrameRecord] = {}
    for r in index_train_real_frames(root):
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
