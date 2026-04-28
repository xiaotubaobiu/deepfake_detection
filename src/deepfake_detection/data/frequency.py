from __future__ import annotations

import cv2
import numpy as np


def rgb_to_frequency_map(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    channels = []
    for channel in cv2.split(image):
        dct = cv2.dct(channel)
        channels.append(dct)
    return np.stack(channels, axis=-1)
