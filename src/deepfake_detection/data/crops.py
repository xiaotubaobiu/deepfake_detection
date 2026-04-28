from __future__ import annotations

import cv2


def expand_box(box, scale, image_h, image_w):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - w / 2)))
    ny1 = max(0, int(round(cy - h / 2)))
    nx2 = min(image_w, int(round(cx + w / 2)))
    ny2 = min(image_h, int(round(cy + h / 2)))
    return nx1, ny1, nx2, ny2


def crop_region(image, box, output_size=224):
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
