from pathlib import Path

import cv2
import numpy as np

from deepfake_detection.data.datasets import FrameClassificationDataset
from deepfake_detection.data.index_external import ExternalFrameRecord
from deepfake_detection.data.index_ffpp import FrameRecord


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_frame_record_sample_id_uses_path_label_and_video_id(tmp_path):
    frame = tmp_path / "video_a" / "frame_001.png"
    record = FrameRecord(
        method="Deepfakes",
        pair_id="video_a",
        frame_name="frame_001.png",
        frame_path=str(frame),
        landmark_path=None,
        label=1,
    )

    assert record.video_id == "video_a"
    assert record.sample_id == f"{frame}::1::video_a"


def test_external_record_sample_id_uses_path_label_and_video_id(tmp_path):
    frame = tmp_path / "fake" / "video_b" / "frame_001.jpg"
    record = ExternalFrameRecord(
        frame_path=str(frame),
        label=1,
        video_id="dfd/fake/video_b",
    )

    assert record.sample_id == f"{frame}::1::dfd/fake/video_b"


def test_classification_dataset_returns_audit_metadata(tmp_path):
    frame = tmp_path / "real" / "video_c" / "frame_001.jpg"
    _write_image(frame)
    record = ExternalFrameRecord(
        frame_path=str(frame),
        label=0,
        video_id="dfd/real/video_c",
    )
    dataset = FrameClassificationDataset([record], augment=False)

    item = dataset[0]

    assert item["label"].item() == 0
    assert item["video_id"] == "dfd/real/video_c"
    assert item["image_path"] == str(frame)
    assert item["sample_id"] == f"{frame}::0::dfd/real/video_c"
    assert tuple(item["image"].shape) == (3, 224, 224)
