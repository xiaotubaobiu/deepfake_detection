import numpy as np
from scipy.fftpack import dctn
import cv2


def rgb_to_dct_map(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert RGB image to DCT frequency feature map.

    Args:
        img: (H, W, 3) uint8 RGB image
        size: output spatial size

    Returns:
        (3, size, size) float32 in [0, 1], DCT log-amplitude map replicated to 3 channels
    """
    # 1. Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float32)

    # 2. 2D DCT type-II, orthonormal
    dct_coeffs = dctn(gray, type=2, norm='ortho')

    # 3. Take absolute value
    dct_abs = np.abs(dct_coeffs)

    # 4. Log transform
    dct_log = np.log1p(dct_abs)

    # 5. Normalize to [0, 1]
    dct_min, dct_max = dct_log.min(), dct_log.max()
    if dct_max - dct_min > 1e-8:
        dct_norm = (dct_log - dct_min) / (dct_max - dct_min)
    else:
        dct_norm = np.zeros_like(dct_log)

    # 6. Resize to target size
    dct_resized = cv2.resize(dct_norm, (size, size), interpolation=cv2.INTER_LINEAR)

    # 7. Replicate to 3 channels (ResNet18 expects 3-channel input)
    dct_3ch = np.stack([dct_resized] * 3, axis=0).astype(np.float32)

    return dct_3ch


def batch_rgb_to_dct(imgs: np.ndarray, size: int = 224) -> np.ndarray:
    """Batch convert RGB images to DCT maps.

    Args:
        imgs: (N, H, W, 3) uint8
        size: output spatial size

    Returns:
        (N, 3, size, size) float32
    """
    return np.stack([rgb_to_dct_map(img, size) for img in imgs])
