import numpy as np
from deepfake_detection.data.crops import expand_box, crop_region


def test_expand_box_grows_face_region_for_context():
    box = (50, 60, 150, 160)
    expanded = expand_box(box, scale=1.3, image_h=224, image_w=224)
    assert expanded[0] <= 50
    assert expanded[1] <= 60
    assert expanded[2] >= 150
    assert expanded[3] >= 160


def test_crop_region_returns_correct_output_size():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    crop = crop_region(image, (32, 32, 192, 192), output_size=224)
    assert crop.shape == (224, 224, 3)


from deepfake_detection.data.index_ffpp import build_aligned_pair_key


def test_build_aligned_pair_key_uses_pair_and_frame_index():
    key = build_aligned_pair_key(method="simswap", pair_id="000_003", frame_name="012.png")
    assert key == "simswap::000_003::012"


from deepfake_detection.data.datasets import collate_video_scores


def test_collate_video_scores_groups_frames_by_video_id():
    rows = [
        {"video_id": "v0", "score": 0.2},
        {"video_id": "v0", "score": 0.4},
        {"video_id": "v1", "score": 0.9},
    ]
    grouped = collate_video_scores(rows)
    assert grouped == {"v0": [0.2, 0.4], "v1": [0.9]}
