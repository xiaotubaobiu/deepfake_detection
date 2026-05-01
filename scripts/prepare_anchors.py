"""
Pre-compute anchor backgrounds for FF++ and Celeb-DF v2.

For each full frame:
1. Detect 68-point face landmarks via face_alignment
2. Build convex hull mask from landmarks
3. Dilate mask by 30px to cover face margins
4. Black out the face region, keep pure background
5. Resize to 224x224 and save as PNG

Output:
  <output_dir>/
    anchors_ffpp_train.json   # {video_id: {frame: {anchor_path, face_box}}}
    anchors_ffpp_test.json
    anchors_cdf.json
    ff++/         # FF++ train anchors
    ff++_test/    # FF++ test anchors
    cdf/          # CDF anchors
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DILATE_PX = 30
OUTPUT_SIZE = 224


# ── Face detection ─────────────────────────────────────────────────────────

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        import face_alignment
        _detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device="cuda", flip_input=False
        )
    return _detector


def detect_landmarks(image: np.ndarray) -> np.ndarray | None:
    """Detect 68-point landmarks. Returns (68, 2) array or None."""
    fa = _get_detector()
    preds = fa.get_landmarks(image)
    if preds is None or len(preds) == 0:
        return None
    return preds[0]


# ── Mask-based background extraction ──────────────────────────────────────


def make_face_mask(landmarks: np.ndarray, h: int, w: int, dilate: int = DILATE_PX) -> np.ndarray:
    """Create binary mask: 0=face (to remove), 255=background (to keep)."""
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.fillConvexPoly(mask, hull, 0)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2, dilate * 2))
        face_region = (mask == 0).astype(np.uint8)
        face_region = cv2.dilate(face_region, kernel, iterations=1)
        mask[face_region > 0] = 0
    return mask


def extract_background(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
    """Mask out face and resize to OUTPUT_SIZE."""
    h, w = image.shape[:2]
    if landmarks is not None:
        mask = make_face_mask(landmarks, h, w)
        bg = image.copy()
        bg[mask == 0] = 0
    else:
        bg = image
    return cv2.resize(bg, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)


# ── CDF frame extraction ───────────────────────────────────────────────────

_cache_cap = None
_cache_path = None


def extract_frame_from_video(video_path: str, frame_idx: int) -> np.ndarray | None:
    global _cache_cap, _cache_path
    if _cache_path != video_path:
        if _cache_cap is not None:
            _cache_cap.release()
        _cache_cap = cv2.VideoCapture(video_path)
        _cache_path = video_path
        if not _cache_cap.isOpened():
            return None
    _cache_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = _cache_cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def release_video_cache():
    global _cache_cap, _cache_path
    if _cache_cap is not None:
        _cache_cap.release()
        _cache_cap = None
        _cache_path = None


# ── FF++ processing ────────────────────────────────────────────────────────


def process_ffpp(anchor_dir: str, output_dir: str, label: str) -> dict:
    anchor_dir = Path(anchor_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not anchor_dir.exists():
        print(f"  Dir not found: {anchor_dir}")
        return {}

    video_ids = sorted([d.name for d in anchor_dir.iterdir() if d.is_dir()])
    index = {}

    for vid in tqdm(video_ids, desc=label):
        vid_dir = anchor_dir / vid
        vid_out = out_dir / vid
        vid_out.mkdir(parents=True, exist_ok=True)
        index[vid] = {}

        for frame_path in sorted(vid_dir.glob("*.png")):
            full_img = np.array(Image.open(frame_path).convert("RGB"))
            lm = detect_landmarks(full_img)
            bg = extract_background(full_img, lm)

            anchor_path = vid_out / f"{frame_path.stem}_anchor.png"
            Image.fromarray(bg).save(anchor_path)

            face_box = None
            if lm is not None:
                face_box = [int(lm[:, 0].min()), int(lm[:, 1].min()),
                            int(lm[:, 0].max()), int(lm[:, 1].max())]

            index[vid][frame_path.name] = {
                "anchor_path": str(anchor_path),
                "face_box": face_box,
            }

    return index


# ── CDF processing ─────────────────────────────────────────────────────────


def process_cdf(root: str, output_dir: str) -> dict:
    face_dir = Path(root) / "Celeb-DF-v2" / "Celeb-real" / "frames"
    raw_dir = Path(root) / "Celeb-DF-v2-raw" / "Celeb-real"
    out_dir = Path(output_dir) / "cdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not face_dir.exists() or not raw_dir.exists():
        print(f"  Dirs not found: face={face_dir.exists()}, raw={raw_dir.exists()}")
        return {}

    video_ids = sorted([d.name for d in face_dir.iterdir() if d.is_dir()])
    index = {}

    for vid in tqdm(video_ids, desc="CDF"):
        video_file = raw_dir / f"{vid}.mp4"
        if not video_file.exists():
            continue

        vid_out = out_dir / vid
        vid_out.mkdir(parents=True, exist_ok=True)
        index[vid] = {}

        for frame_path in sorted((face_dir / vid).glob("*.png")):
            frame_idx = int(frame_path.stem)
            full_img = extract_frame_from_video(str(video_file), frame_idx)
            if full_img is None:
                continue

            lm = detect_landmarks(full_img)
            bg = extract_background(full_img, lm)

            anchor_path = vid_out / f"{frame_path.stem}_anchor.png"
            Image.fromarray(bg).save(anchor_path)

            face_box = None
            if lm is not None:
                face_box = [int(lm[:, 0].min()), int(lm[:, 1].min()),
                            int(lm[:, 0].max()), int(lm[:, 1].max())]

            index[vid][frame_path.name] = {
                "anchor_path": str(anchor_path),
                "face_box": face_box,
                "video_path": str(video_file),
                "frame_idx": frame_idx,
            }

    release_video_cache()
    return index


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Dataset/deepfake_detection/DF40_all")
    parser.add_argument("--output", default="/Dataset/deepfake_detection/DF40_all/precomputed_anchors")
    parser.add_argument("--only", choices=["ffpp_train", "ffpp_test", "cdf"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.only is None or args.only == "ffpp_train":
        idx = process_ffpp(f"{args.root}/DF40_train/anchor", f"{args.output}/ff++", "FF++ train")
        with open(f"{args.output}/anchors_ffpp_train.json", "w") as f:
            json.dump(idx, f)
        print(f"  FF++ train: {len(idx)} videos, {sum(len(v) for v in idx.values())} frames")

    if args.only is None or args.only == "ffpp_test":
        idx = process_ffpp(f"{args.root}/DF40_test/anchor", f"{args.output}/ff++_test", "FF++ test")
        with open(f"{args.output}/anchors_ffpp_test.json", "w") as f:
            json.dump(idx, f)
        print(f"  FF++ test: {len(idx)} videos, {sum(len(v) for v in idx.values())} frames")

    if args.only is None or args.only == "cdf":
        idx = process_cdf(args.root, args.output)
        with open(f"{args.output}/anchors_cdf.json", "w") as f:
            json.dump(idx, f)
        print(f"  CDF: {len(idx)} videos, {sum(len(v) for v in idx.values())} frames")

    print("Done.")


if __name__ == "__main__":
    main()
