import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis


class FaceDetector:
    """SCRFD-based face detector using InsightFace buffalo_l model pack."""

    def __init__(self, model_path: str = None, ctx_id: int = 0):
        """Initialize face detector.

        Args:
            model_path: path to directory containing det_10g.onnx (e.g. ~/project/evaluation_metic/models/buffalo_l)
            ctx_id: GPU id, -1 for CPU
        """
        if model_path:
            # If custom path given, set insightface model_dir to parent of 'models/'
            model_dir = model_path.rsplit('/models/', 1)[0] if '/models/' in model_path else model_path
            self.app = FaceAnalysis(
                name='buffalo_l',
                root=model_dir,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        else:
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_largest_face(self, img: np.ndarray):
        """Detect the largest face in an image.

        Args:
            img: (H, W, 3) uint8 BGR or RGB image

        Returns:
            bbox: [x1, y1, x2, y2] or None if no face detected
        """
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        best = faces[np.argmax(areas)]
        return best.bbox.astype(int)


def landmarks_to_bbox(landmarks: np.ndarray, img_h: int, img_w: int):
    """Convert landmarks to face bounding box.

    Args:
        landmarks: (81, 2) or (N, 2) array of (x, y) points
        img_h: image height
        img_w: image width

    Returns:
        bbox: [x1, y1, x2, y2] square-padded
    """
    x1 = landmarks[:, 0].min()
    y1 = landmarks[:, 1].min()
    x2 = landmarks[:, 0].max()
    y2 = landmarks[:, 1].max()

    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(w, h)
    side = int(side * 1.2)
    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = x1 + side
    y2 = y1 + side

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return np.array([x1, y1, x2, y2])


def expand_bbox_for_context(bbox: np.ndarray, scale: float, img_h: int, img_w: int):
    """Expand bounding box for context crop.

    Args:
        bbox: [x1, y1, x2, y2]
        scale: expansion factor (e.g., 1.8)
        img_h: image height
        img_w: image width

    Returns:
        expanded: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    side = max(w, h)
    nx1 = int(cx - side / 2)
    ny1 = int(cy - side / 2)
    nx2 = int(cx + side / 2)
    ny2 = int(cy + side / 2)
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    return np.array([nx1, ny1, nx2, ny2])


def crop_face(img: np.ndarray, bbox: np.ndarray, size: int = 224):
    """Crop and resize face region.

    Args:
        img: (H, W, 3) uint8
        bbox: [x1, y1, x2, y2]
        size: output size

    Returns:
        (size, size, 3) uint8
    """
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_context(img: np.ndarray, face_bbox: np.ndarray, context_bbox: np.ndarray, size: int = 224):
    """Crop context region with face masked out.

    Args:
        img: (H, W, 3) uint8
        face_bbox: [x1, y1, x2, y2] in original image coordinates
        context_bbox: [x1, y1, x2, y2] in original image coordinates
        size: output size

    Returns:
        (size, size, 3) uint8 with face region filled black
    """
    cx1, cy1, cx2, cy2 = context_bbox
    crop = img[cy1:cy2, cx1:cx2].copy()

    fx1 = max(0, face_bbox[0] - cx1)
    fy1 = max(0, face_bbox[1] - cy1)
    fx2 = min(cx2 - cx1, face_bbox[2] - cx1)
    fy2 = min(cy2 - cy1, face_bbox[3] - cy1)
    crop[fy1:fy2, fx1:fx2] = 0

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
