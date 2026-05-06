import numpy as np
from deepfake_detection.data.transforms import build_rgb_augment, apply_shared_transform_pair


def test_build_rgb_augment_uses_expected_policy_names():
    transform = build_rgb_augment()
    names = [type(t).__name__ for t in transform.transforms]
    assert names == [
        "HorizontalFlip",
        "Rotate",
        "GaussianBlur",
        "OneOf",
        "ImageCompression",
        "Normalize",
    ]


def test_apply_shared_transform_pair_keeps_image_shapes_aligned():
    image_a = np.zeros((224, 224, 3), dtype=np.uint8)
    image_b = np.zeros((224, 224, 3), dtype=np.uint8)
    aug_a, aug_b = apply_shared_transform_pair(image_a, image_b)
    assert aug_a.shape == (224, 224, 3)
    assert aug_b.shape == (224, 224, 3)


from deepfake_detection.data.frequency import rgb_to_frequency_map


def test_rgb_to_frequency_map_preserves_spatial_size():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    freq = rgb_to_frequency_map(image)
    assert freq.shape[:2] == (224, 224)
