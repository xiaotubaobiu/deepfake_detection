import numpy as np
import cv2


class PairAugmentation:
    """Synchronized augmentation for (face, context) pairs.

    Both face and context receive the same geometric and photometric transforms.
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def _jpeg_compress(self, img: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression to an image."""
        _, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    def __call__(self, face: np.ndarray, context: np.ndarray):
        """Apply synchronized augmentation to face and context pair.

        Args:
            face: (H, W, 3) uint8
            context: (H, W, 3) uint8

        Returns:
            (face_aug, context_aug) same shape and dtype
        """
        # JPEG compression (quality 50-95)
        if self.rng.random() < 0.5:
            quality = int(self.rng.uniform(50, 96))
            face = self._jpeg_compress(face, quality)
            context = self._jpeg_compress(context, quality)

        # Gaussian blur (sigma 0.5-1.5)
        if self.rng.random() < 0.3:
            sigma = self.rng.uniform(0.5, 1.5)
            ksize = int(sigma * 4) * 2 + 1
            face = cv2.GaussianBlur(face, (ksize, ksize), sigma)
            context = cv2.GaussianBlur(context, (ksize, ksize), sigma)

        # Brightness/contrast jitter
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.8, 1.2)  # contrast
            beta = self.rng.uniform(-0.1, 0.1)   # brightness (in [0,1] scale)
            face = np.clip(face.astype(np.float32) * alpha + beta * 255, 0, 255).astype(np.uint8)
            context = np.clip(context.astype(np.float32) * alpha + beta * 255, 0, 255).astype(np.uint8)

        # Slight color jitter (hue/saturation shift)
        if self.rng.random() < 0.3:
            shift = self.rng.uniform(-10, 10)
            face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.int16)
            face_hsv[:, :, 0] = np.clip(face_hsv[:, :, 0] + shift, 0, 179)
            face = cv2.cvtColor(face_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            context_hsv = cv2.cvtColor(context, cv2.COLOR_RGB2HSV).astype(np.int16)
            context_hsv[:, :, 0] = np.clip(context_hsv[:, :, 0] + shift, 0, 179)
            context = cv2.cvtColor(context_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Slight resize (0.9-1.1x)
        if self.rng.random() < 0.3:
            scale = self.rng.uniform(0.9, 1.1)
            h, w = face.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR)
            context = cv2.resize(context, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            context = cv2.resize(context, (w, h), interpolation=cv2.INTER_LINEAR)

        return face, context


def get_augmentation_pair(seed: int = None):
    """Return a PairAugmentation instance."""
    return PairAugmentation(seed=seed)
