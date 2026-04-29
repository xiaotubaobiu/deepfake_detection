from __future__ import annotations

import random

import cv2
import numpy as np


def _detect_face_box_from_landmarks(landmarks: np.ndarray, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    margin_x = int((xs.max() - xs.min()) * 0.3)
    margin_y = int((ys.max() - ys.min()) * 0.3)
    x1 = max(0, int(xs.min()) - margin_x)
    y1 = max(0, int(ys.min()) - margin_y)
    x2 = min(img_w, int(xs.max()) + margin_x)
    y2 = min(img_h, int(ys.max()) + margin_y)
    return x1, y1, x2, y2


def _detect_face_box_fallback(img_h: int, img_w: int, margin_ratio: float = 0.2) -> tuple[int, int, int, int]:
    mx = int(img_w * margin_ratio)
    my = int(img_h * margin_ratio)
    return mx, my, img_w - mx, img_h - my


def sample_bg_patches(
    image: np.ndarray,
    num_patches: int = 4,
    patch_size: int = 224,
    face_box: tuple[int, int, int, int] | None = None,
    rng: random.Random | None = None,
) -> list[np.ndarray]:
    if rng is None:
        rng = random.Random()

    h, w = image.shape[:2]
    if face_box is None:
        face_box = _detect_face_box_fallback(h, w)

    fx1, fy1, fx2, fy2 = face_box
    patches = []
    for _ in range(num_patches):
        best = None
        best_dist = -1
        for _ in range(20):
            px = rng.randint(0, max(0, w - patch_size))
            py = rng.randint(0, max(0, h - patch_size))
            ix1 = max(px, fx1)
            iy1 = max(py, fy1)
            ix2 = min(px + patch_size, fx2)
            iy2 = min(py + patch_size, fy2)
            overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = overlap / (patch_size * patch_size) if patch_size > 0 else 0
            dist = 1.0 - iou
            if dist > best_dist:
                best_dist = dist
                best = (px, py)
            if iou == 0:
                break

        if best is not None:
            px, py = best
            patch = image[py:py + patch_size, px:px + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
        else:
            patches.append(cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR))

    return patches
