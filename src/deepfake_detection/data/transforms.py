from __future__ import annotations

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_rgb_augment() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5,
        ),
        A.ImageCompression(quality_range=(40, 100), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ], additional_targets={"image_pair": "image"})


def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def apply_shared_transform_pair(image, image_pair):
    transformed = build_rgb_augment()(image=image, image_pair=image_pair)
    return transformed["image"], transformed["image_pair"]
