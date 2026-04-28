from __future__ import annotations

import albumentations as A


def build_rgb_augment() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5,
        ),
        A.ImageCompression(quality_lower=40, quality_upper=100, p=0.2),
    ], additional_targets={"image_pair": "image"})


def apply_shared_transform_pair(image, image_pair):
    transformed = build_rgb_augment()(image=image, image_pair=image_pair)
    return transformed["image"], transformed["image_pair"]
