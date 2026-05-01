from __future__ import annotations

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.45762756, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_rgb_augment(mean=IMAGENET_MEAN, std=IMAGENET_STD, seed=42) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=[-10, 10], p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.FancyPCA(),
            A.HueSaturationValue(),
        ], p=0.5),
        A.ImageCompression(quality_range=(40, 100), p=0.5),
        A.Normalize(mean=mean, std=std),
    ], additional_targets={"image_pair": "image"}, seed=seed)


def build_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=mean, std=std),
    ])


def apply_shared_transform_pair(image, image_pair):
    transformed = build_rgb_augment()(image=image, image_pair=image_pair)
    return transformed["image"], transformed["image_pair"]
