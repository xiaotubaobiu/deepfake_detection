#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import dlib
import numpy as np
from imutils import face_utils
from skimage import transform as trans


PREDICTOR_PATH = Path("scripts/dlib_tools/shape_predictor_81_face_landmarks.dat")


@dataclass(frozen=True)
class DFDClass:
    key: str
    label: str
    videos_dir: Path
    frames_dir: Path
    landmarks_dir: Path
    frames_wocropface_dir: Path


def dfd_classes(root: Path, compression: str) -> tuple[DFDClass, DFDClass]:
    ffpp = root / "FaceForensics++"
    fake_base = ffpp / "manipulated_sequences" / "DeepFakeDetection" / compression
    real_base = ffpp / "original_sequences" / "actors" / compression
    return (
        DFDClass("DFD_fake", "fake", fake_base / "videos", fake_base / "frames", fake_base / "landmarks", fake_base / "frames_wocropface"),
        DFDClass("DFD_real", "real", real_base / "videos", real_base / "frames", real_base / "landmarks", real_base / "frames_wocropface"),
    )


def get_keypts(image, face, predictor, face_detector):
    shape = predictor(image, face)
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize
        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        matrix = tform.params[0:2, :]

        img = cv2.warpAffine(img, matrix, (target_size[1], target_size[0]))
        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, matrix, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        return img, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if len(faces) == 0:
        return None, None, None

    img_center = np.array([rgb.shape[1] // 2, rgb.shape[0] // 2])
    face = max(
        faces,
        key=lambda rect: rect.width() * rect.height() and np.linalg.norm(
            np.array([(rect.left() + rect.right()) // 2, (rect.top() + rect.bottom()) // 2]) - img_center
        ),
    )

    landmarks = get_keypts(rgb, face, predictor, face_detector)
    cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

    face_align = face_detector(cropped_face, 1)
    if len(face_align) == 0:
        return None, None, None
    landmark = predictor(cropped_face, face_align[0])
    landmark = face_utils.shape_to_np(landmark)
    return cropped_face, landmark, mask_face


def selected_frame_indices(frame_count: int, frames_per_video: int) -> list[int]:
    if frame_count <= 0:
        return []
    return np.linspace(0, frame_count - 1, frames_per_video, endpoint=True, dtype=int).tolist()


def expected_frame_paths(output_dir: Path, frame_indices: list[int]) -> list[Path]:
    return [output_dir / f"{frame_idx:03d}.png" for frame_idx in sorted(set(frame_indices))]


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def extract_faces(
    video_path: Path,
    frames_dir: Path,
    landmarks_dir: Path,
    frames_wocropface_dir: Path,
    frames_per_video: int,
    overwrite: bool,
    predictor_path: Path,
    keep_landmarks: bool,
    delete_full_frames: bool,
) -> tuple[list[Path], list[Path]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return [], []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = selected_frame_indices(frame_count, frames_per_video)
        expected_frames = expected_frame_paths(frames_dir, frame_indices)
        expected_landmarks = [landmarks_dir / path.with_suffix(".npy").name for path in expected_frames]
        if not overwrite and all(path.exists() for path in expected_frames) and (not keep_landmarks or all(path.exists() for path in expected_landmarks)):
            existing_landmarks = [path.resolve() for path in expected_landmarks if path.exists()]
            return [path.resolve() for path in expected_frames], existing_landmarks

        frames_dir.mkdir(parents=True, exist_ok=True)
        if keep_landmarks:
            landmarks_dir.mkdir(parents=True, exist_ok=True)
        if not delete_full_frames:
            frames_wocropface_dir.mkdir(parents=True, exist_ok=True)
        frame_idx_set = set(frame_indices)
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))
        written_frames: list[Path] = []
        written_landmarks: list[Path] = []

        for cnt_frame in range(frame_count):
            ret_org, frame_org = cap.read()
            if not ret_org:
                break

            if not delete_full_frames:
                ori_frame_path = frames_wocropface_dir / f"{cnt_frame:03d}.png"
                if overwrite or not ori_frame_path.is_file():
                    cv2.imwrite(str(ori_frame_path), frame_org)

            if cnt_frame not in frame_idx_set:
                continue

            cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, predictor, frame_org)
            if cropped_face is None or landmarks is None:
                continue

            image_path = frames_dir / f"{cnt_frame:03d}.png"
            if overwrite or not image_path.is_file():
                cv2.imwrite(str(image_path), cropped_face)

            landmark_path = landmarks_dir / f"{cnt_frame:03d}.npy"
            if keep_landmarks and (overwrite or not landmark_path.is_file()):
                np.save(str(landmark_path), landmarks)

            written_frames.append(image_path.resolve())
            if keep_landmarks:
                written_landmarks.append(landmark_path.resolve())
        if delete_full_frames:
            remove_tree(frames_wocropface_dir)
        if not keep_landmarks:
            remove_tree(landmarks_dir)
        return written_frames, written_landmarks
    finally:
        cap.release()


def _prepare_one_video(args: tuple[str, str, str, str, int, bool, str, bool, bool]) -> tuple[str, list[str], list[str]]:
    video_path_str, frames_root_str, landmarks_root_str, frames_wocropface_root_str, frames_per_video, overwrite, predictor_path_str, keep_landmarks, delete_full_frames = args
    video_path = Path(video_path_str)
    frames, landmarks = extract_faces(
        video_path,
        Path(frames_root_str) / video_path.stem,
        Path(landmarks_root_str) / video_path.stem,
        Path(frames_wocropface_root_str) / video_path.stem,
        frames_per_video,
        overwrite,
        Path(predictor_path_str),
        keep_landmarks,
        delete_full_frames,
    )
    return video_path.stem, [str(path) for path in frames], [str(path) for path in landmarks]


def prepare_class(
    spec: DFDClass,
    frames_per_video: int,
    overwrite: bool,
    limit: int | None,
    workers: int,
    predictor_path: Path,
    keep_landmarks: bool,
    delete_full_frames: bool,
) -> tuple[dict[str, tuple[list[Path], list[Path]]], int]:
    videos = sorted(spec.videos_dir.glob("*.mp4")) if spec.videos_dir.is_dir() else []
    if limit is not None:
        videos = videos[:limit]
    tasks = [
        (
            str(video),
            str(spec.frames_dir),
            str(spec.landmarks_dir),
            str(spec.frames_wocropface_dir),
            frames_per_video,
            overwrite,
            str(predictor_path),
            keep_landmarks,
            delete_full_frames,
        )
        for video in videos
    ]
    prepared: dict[str, tuple[list[Path], list[Path]]] = {}
    skipped = 0

    if workers <= 1:
        iterator = (_prepare_one_video(task) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = (future.result() for future in as_completed([executor.submit(_prepare_one_video, task) for task in tasks]))

    try:
        for idx, (video_id, frames, landmarks) in enumerate(iterator, start=1):
            if frames:
                prepared[video_id] = ([Path(path) for path in frames], [Path(path) for path in landmarks])
            else:
                skipped += 1
            if idx % 25 == 0 or idx == len(tasks):
                print(f"{spec.key}: processed={idx}/{len(tasks)} prepared={len(prepared)} skipped={skipped}", flush=True)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    return prepared, skipped


def build_json(fake: dict[str, tuple[list[Path], list[Path]]], real: dict[str, tuple[list[Path], list[Path]]], compression: str) -> dict:
    data = {
        "DeepFakeDetection": {
            "DFD_fake": {"train": {compression: {}}, "val": {compression: {}}, "test": {compression: {}}},
            "DFD_real": {"train": {compression: {}}, "val": {compression: {}}, "test": {compression: {}}},
        }
    }
    for key, videos, label in (("DFD_fake", fake, "DFD_fake"), ("DFD_real", real, "DFD_real")):
        for video_id, (frames, landmarks) in sorted(videos.items()):
            entry = {
                "label": label,
                "frames": [str(frame) for frame in frames],
                "landmarks": [str(landmark) for landmark in landmarks],
            }
            for split in ("train", "val", "test"):
                data["DeepFakeDetection"][key][split][compression][video_id] = entry
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DFD videos with the same dlib alignment flow as DF40/DeepfakeBench preprocessing.")
    parser.add_argument("--root", default="/Dataset/deepfake_detection", help="Root containing FaceForensics++.")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--frames-per-video", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--predictor", default=str(PREDICTOR_PATH))
    parser.add_argument("--keep-landmarks", action="store_true")
    parser.add_argument("--keep-full-frames", action="store_true")
    parser.add_argument("--output-json", default="/Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json")
    args = parser.parse_args()

    predictor_path = Path(args.predictor)
    if not predictor_path.is_file():
        raise FileNotFoundError(f"Missing dlib shape predictor: {predictor_path}")

    root = Path(args.root)
    fake_spec, real_spec = dfd_classes(root, args.compression)
    print(f"fake videos: {fake_spec.videos_dir}", flush=True)
    print(f"real videos: {real_spec.videos_dir}", flush=True)
    print(f"predictor: {predictor_path}", flush=True)

    fake, fake_skipped = prepare_class(
        fake_spec,
        args.frames_per_video,
        args.overwrite,
        args.limit,
        args.workers,
        predictor_path,
        keep_landmarks=args.keep_landmarks,
        delete_full_frames=not args.keep_full_frames,
    )
    real, real_skipped = prepare_class(
        real_spec,
        args.frames_per_video,
        args.overwrite,
        args.limit,
        args.workers,
        predictor_path,
        keep_landmarks=args.keep_landmarks,
        delete_full_frames=not args.keep_full_frames,
    )

    data = build_json(fake, real, args.compression)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2))

    fake_frames = sum(len(frames) for frames, _ in fake.values())
    real_frames = sum(len(frames) for frames, _ in real.values())
    print(f"wrote {output_json}", flush=True)
    print(f"DFD_fake videos={len(fake)} frames={fake_frames} skipped={fake_skipped}", flush=True)
    print(f"DFD_real videos={len(real)} frames={real_frames} skipped={real_skipped}", flush=True)


if __name__ == "__main__":
    main()
