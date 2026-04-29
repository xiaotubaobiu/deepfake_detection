from __future__ import annotations

import json
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


def _remap_path(json_path: str, root: str) -> str:
    """Remap JSON path to actual filesystem path."""
    rel = json_path.replace("deepfakes_detection_datasets/", "")
    rel = rel.replace("DF40/", "DF40_test/", 1)
    return str(Path(root) / rel)


def load_json_index(
    root: str,
    method: str,
    domain: str,
    split: str,
    label: int,
    max_videos: int | None = None,
    video_range: tuple[int, int] | None = None,
) -> list[FrameRecord]:
    """Load frame records from a pre-built JSON index file."""
    json_dir = Path(root) / "dataset_json"
    json_path = json_dir / f"{method}_{domain}.json"
    if not json_path.exists():
        return []

    with open(json_path) as f:
        data = json.load(f)

    top_key = f"{method}_{domain}"
    category = f"{method}_Real" if label == 0 else f"{method}_Fake"
    section = data.get(top_key, {}).get(category, {}).get(split, {})
    if not section:
        return []

    video_ids = sorted(section.keys())
    if video_range is not None:
        video_ids = video_ids[video_range[0]:video_range[1]]
    elif max_videos is not None:
        video_ids = video_ids[:max_videos]

    records = []
    for vid_id in video_ids:
        vid_data = section[vid_id]
        frames = vid_data.get("frames", [])
        landmarks = vid_data.get("landmarks", [])
        for i, frame_path in enumerate(frames):
            actual_path = _remap_path(frame_path, root)
            lm_path = None
            if i < len(landmarks) and landmarks[i]:
                lm_path = _remap_path(landmarks[i], root)
            records.append(FrameRecord(
                method=method if label == 1 else "real",
                pair_id=vid_id,
                frame_name=Path(actual_path).name,
                frame_path=actual_path,
                landmark_path=lm_path,
                label=label,
            ))
    return records


def build_aligned_pair_key(method: str, pair_id: str, frame_name: str) -> str:
    frame_id = Path(frame_name).stem
    return f"{method}::{pair_id}::{frame_id}"


def index_train_aligned_triplets(root: str, method: str, max_videos: int | None = None) -> list[dict]:
    """Build aligned (real, fake) frame pairs from directory structure."""
    frames_dir = Path(root) / "DF40_train" / method / "frames"
    if not frames_dir.is_dir():
        return []

    # --- inline: index fake frames ---
    fake_records: list[FrameRecord] = []
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    for video_dir in video_dirs:
        pair_id = video_dir.name
        lm_dir = Path(root) / "DF40_train" / method / "landmarks" / pair_id
        for frame_path in sorted(video_dir.glob("*.png")):
            lm_file = lm_dir / (frame_path.stem + ".npy") if lm_dir.is_dir() else None
            fake_records.append(FrameRecord(
                method=method,
                pair_id=pair_id,
                frame_name=frame_path.name,
                frame_path=str(frame_path),
                landmark_path=str(lm_file) if lm_file and lm_file.exists() else None,
                label=1,
            ))

    # --- inline: index real (anchor) frames ---
    anchor_dir = Path(root) / "DF40_train" / "anchor"
    real_index: dict[tuple[str, str], FrameRecord] = {}
    if anchor_dir.is_dir():
        for video_dir in sorted(anchor_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            pair_id = video_dir.name
            for frame_path in sorted(video_dir.glob("*.png")):
                rec = FrameRecord(
                    method="real",
                    pair_id=pair_id,
                    frame_name=frame_path.name,
                    frame_path=str(frame_path),
                    landmark_path=None,
                    label=0,
                )
                real_index[(rec.pair_id, rec.frame_name)] = rec

    # --- match fake to real ---
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
