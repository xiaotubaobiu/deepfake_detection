from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deepfake_detection.data.constants import ALL_METHODS


@dataclass(frozen=True)
class CDFRecord:
    method: str
    video_id: str
    frame_path: str
    label: int


def normalize_cdf_method_name(name: str) -> str:
    return name.strip()


def index_cdf_frames(cdf_root: str, methods: list[str] | None = None) -> list[CDFRecord]:
    target_methods = methods or ALL_METHODS
    test_dir = Path(cdf_root) / "DF40_test"
    if not test_dir.is_dir():
        return []
    records = []
    anchor_dir = test_dir / "anchor"
    if anchor_dir.is_dir():
        for video_dir in sorted(anchor_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for frame_path in sorted(video_dir.glob("*.png")):
                records.append(CDFRecord(
                    method="real",
                    video_id=video_dir.name,
                    frame_path=str(frame_path),
                    label=0,
                ))
    for method in target_methods:
        for domain in ("cdf", "ff"):
            frames_dir = test_dir / method / domain / "frames"
            if not frames_dir.is_dir():
                continue
            for video_dir in sorted(frames_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                for frame_path in sorted(video_dir.glob("*.png")):
                    records.append(CDFRecord(
                        method=method,
                        video_id=video_dir.name,
                        frame_path=str(frame_path),
                        label=1,
                    ))
    return records


def filter_cdf_records_by_methods(records: list[CDFRecord], methods: list[str]) -> list[CDFRecord]:
    method_set = set(methods)
    return [r for r in records if r.method in method_set or r.method == "real"]
