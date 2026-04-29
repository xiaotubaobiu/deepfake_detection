from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from deepfake_detection.data.constants import ALL_METHODS, IMG_SIZE, REAL_LABEL, FAKE_LABEL
from deepfake_detection.data.crops import crop_region, expand_box
from deepfake_detection.data.frequency import rgb_to_frequency_map
from deepfake_detection.data.sampling import sample_uniform_frame_indices
from deepfake_detection.data.transforms import apply_shared_transform_pair, build_rgb_augment, build_eval_transform


def collate_video_scores(rows: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row["score"])
    return dict(grouped)


def _load_frame(frame_path: str) -> np.ndarray:
    return np.array(Image.open(frame_path).convert("RGB"))


def _detect_face_box(img: np.ndarray, landmark_path: str | None = None) -> tuple:
    h, w = img.shape[:2]
    if landmark_path is not None:
        try:
            landmarks = np.load(landmark_path)
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            return x1, y1, x2, y2
        except Exception:
            pass
    margin = 20
    return margin, margin, w - margin, h - margin


class FrameClassificationDataset(Dataset):
    def __init__(self, records, augment=False, img_size=IMG_SIZE, normalize_mean=None, normalize_std=None):
        self.records = records
        self.augment = augment
        self.img_size = img_size
        mean = normalize_mean or (0.485, 0.456, 0.406)
        std = normalize_std or (0.229, 0.224, 0.225)
        self.transform = build_rgb_augment(mean=mean, std=std) if augment else build_eval_transform(mean=mean, std=std)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = _load_frame(rec.frame_path)
        img = cv2_resize(img, self.img_size)
        img = self.transform(image=img)["image"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return {"image": tensor, "label": torch.tensor(rec.label, dtype=torch.long), "video_id": rec.pair_id if hasattr(rec, 'pair_id') else rec.video_id}


def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


class AlignedTripletDataset(Dataset):
    def __init__(self, triplets, augment=True, img_size=IMG_SIZE, context_scale=1.8):
        self.triplets = triplets
        self.augment = augment
        self.img_size = img_size
        self.context_scale = context_scale

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        real_img = _load_frame(t["real_frame_path"])
        fake_img = _load_frame(t["fake_frame_path"])
        if self.augment:
            real_img, fake_img = apply_shared_transform_pair(real_img, fake_img)
        h, w = real_img.shape[:2]
        face_box = _detect_face_box(real_img, t.get("real_landmark_path"))
        bg_box = expand_box(face_box, self.context_scale, h, w)
        bg_crop = crop_region(real_img, bg_box, self.img_size)
        real_face_crop = crop_region(real_img, face_box, self.img_size)
        fake_face_crop = crop_region(fake_img, face_box, self.img_size)
        return {
            "background": torch.from_numpy(bg_crop).permute(2, 0, 1).float() / 255.0,
            "real_face": torch.from_numpy(real_face_crop).permute(2, 0, 1).float() / 255.0,
            "fake_face": torch.from_numpy(fake_face_crop).permute(2, 0, 1).float() / 255.0,
            "label": torch.tensor(FAKE_LABEL, dtype=torch.long),
            "video_id": t["pair_id"],
        }


class BgFaceTripletDataset(Dataset):
    def __init__(self, triplets, augment=False, img_size=224, num_bg_patches=4,
                 normalize_mean=None, normalize_std=None):
        self.triplets = triplets
        self.augment = augment
        self.img_size = img_size
        self.num_bg_patches = num_bg_patches
        mean = normalize_mean or (0.485, 0.456, 0.406)
        std = normalize_std or (0.229, 0.224, 0.225)
        self.transform = build_rgb_augment(mean=mean, std=std) if augment else build_eval_transform(mean=mean, std=std)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        import random as _random
        from deepfake_detection.data.bg_patches import sample_bg_patches

        t = self.triplets[idx]

        anchor_img = _load_frame(t["real_frame_path"])
        fake_img = _load_frame(t["fake_frame_path"])

        face_box = None
        if t.get("fake_landmark_path"):
            try:
                face_box = _detect_face_box(anchor_img, t["fake_landmark_path"])
            except Exception:
                pass

        if face_box is not None:
            fx1, fy1, fx2, fy2 = expand_box(face_box, 1.5, anchor_img.shape[0], anchor_img.shape[1])
            real_face_crop = crop_region(anchor_img, (fx1, fy1, fx2, fy2), self.img_size)
        else:
            real_face_crop = cv2_resize(anchor_img, self.img_size)

        fake_resized = cv2_resize(fake_img, self.img_size)

        rng = _random.Random()
        bg_patches_raw = sample_bg_patches(
            anchor_img, num_patches=self.num_bg_patches,
            patch_size=self.img_size, face_box=face_box, rng=rng,
        )

        real_face_transformed = self.transform(image=real_face_crop)["image"]
        fake_transformed = self.transform(image=fake_resized)["image"]
        bg_tensors = [self.transform(image=bp)["image"] for bp in bg_patches_raw]

        real_tensor = torch.from_numpy(real_face_transformed).permute(2, 0, 1).float() / 255.0
        fake_tensor = torch.from_numpy(fake_transformed).permute(2, 0, 1).float() / 255.0
        bg_stacked = torch.stack([
            torch.from_numpy(bt).permute(2, 0, 1).float() / 255.0 for bt in bg_tensors
        ])

        return {
            "background": bg_stacked,
            "real_face": real_tensor,
            "fake_face": fake_tensor,
            "label": torch.tensor(1, dtype=torch.long),
            "video_id": t["pair_id"],
        }
