import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from data.face_detector import landmarks_to_bbox, expand_bbox_for_context, crop_face, crop_context
from data.frequency import rgb_to_dct_map


class DF40Dataset(Dataset):
    """DF40 deepfake detection dataset.

    Supports:
      - Train: DF40_train/anchor (real) + DF40_train/<method>/frames (fake)
      - Test-ff: DF40_test/<method>/ff/frames (same-domain)
      - Test-cdf: DF40_test/<method>/cdf/frames (cross-domain, Celeb-DF sources)
    """

    def __init__(
        self,
        root: str = '/Dataset/deepfake_detection/DF40_all',
        split: str = 'train',
        test_domain: str = 'ff',
        methods: list = None,
        frames_per_video: int = 1,
        img_size: int = 224,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.root = root
        self.split = split
        self.test_domain = test_domain
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.rng = random.Random(seed)

        self.samples = []  # list of (frame_path, landmarks_path_or_None, label)

        if split in ('train', 'val'):
            self._load_train_val(methods, val_ratio, seed)
        else:
            self._load_test(test_domain, methods)

    def _load_train_val(self, methods, val_ratio, seed):
        train_dir = os.path.join(self.root, 'DF40_train')

        # Collect real samples from anchor
        anchor_dir = os.path.join(train_dir, 'anchor')
        real_videos = sorted(os.listdir(anchor_dir))

        # Collect fake samples
        all_methods = sorted([d for d in os.listdir(train_dir) if d != 'anchor'])
        if methods is not None:
            all_methods = [m for m in all_methods if m in methods]

        # Build video list with labels
        all_videos = []
        for vid_id in real_videos:
            vid_path = os.path.join(anchor_dir, vid_id)
            if os.path.isdir(vid_path):
                all_videos.append((vid_path, None, 0))

        for method in all_methods:
            frames_dir = os.path.join(train_dir, method, 'frames')
            lm_dir = os.path.join(train_dir, method, 'landmarks')
            if not os.path.isdir(frames_dir):
                continue
            for vid_id in sorted(os.listdir(frames_dir)):
                vid_path = os.path.join(frames_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                lm_path = os.path.join(lm_dir, vid_id) if os.path.isdir(lm_dir) else None
                all_videos.append((vid_path, lm_path, 1))

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(all_videos)
        n_val = int(len(all_videos) * val_ratio)

        if self.split == 'val':
            all_videos = all_videos[:n_val]
        else:
            all_videos = all_videos[n_val:]

        # Sample frames from each video
        for vid_path, lm_path, label in all_videos:
            frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.png')])
            if not frames:
                continue
            sampled = self.rng.choices(frames, k=min(self.frames_per_video, len(frames)))
            for frame_name in sampled:
                frame_path = os.path.join(vid_path, frame_name)
                frame_lm = None
                if lm_path is not None:
                    lm_file = os.path.splitext(frame_name)[0] + '.npy'
                    candidate = os.path.join(lm_path, lm_file)
                    if os.path.exists(candidate):
                        frame_lm = candidate
                self.samples.append((frame_path, frame_lm, label))

    def _load_test(self, domain, methods):
        test_dir = os.path.join(self.root, 'DF40_test')

        # Real samples from anchor
        anchor_dir = os.path.join(test_dir, 'anchor')
        if os.path.isdir(anchor_dir):
            for vid_id in sorted(os.listdir(anchor_dir)):
                vid_path = os.path.join(anchor_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.png')])
                for frame_name in frames:
                    self.samples.append((os.path.join(vid_path, frame_name), None, 0))

        # Fake samples
        all_methods = sorted([d for d in os.listdir(test_dir)
                              if os.path.isdir(os.path.join(test_dir, d))
                              and d != 'anchor' and not d.endswith('.zip')])
        if methods is not None:
            all_methods = [m for m in all_methods if m in methods]

        for method in all_methods:
            domain_dir = os.path.join(test_dir, method, domain)
            if not os.path.isdir(domain_dir):
                continue
            frames_dir = os.path.join(domain_dir, 'frames')
            lm_dir = os.path.join(domain_dir, 'landmarks')
            if not os.path.isdir(frames_dir):
                continue
            for vid_id in sorted(os.listdir(frames_dir)):
                vid_path = os.path.join(frames_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                lm_vid = os.path.join(lm_dir, vid_id) if os.path.isdir(lm_dir) else None
                for frame_name in sorted(os.listdir(vid_path)):
                    if not frame_name.endswith('.png'):
                        continue
                    frame_path = os.path.join(vid_path, frame_name)
                    frame_lm = None
                    if lm_vid is not None:
                        lm_file = os.path.splitext(frame_name)[0] + '.npy'
                        candidate = os.path.join(lm_vid, lm_file)
                        if os.path.exists(candidate):
                            frame_lm = candidate
                    self.samples.append((frame_path, frame_lm, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, lm_path, label = self.samples[idx]

        img = np.array(Image.open(frame_path).convert('RGB'))
        h, w = img.shape[:2]

        if lm_path is not None:
            landmarks = np.load(lm_path)
            face_bbox = landmarks_to_bbox(landmarks, h, w)
        else:
            margin = 20
            face_bbox = np.array([margin, margin, w - margin, h - margin])

        ctx_bbox = expand_bbox_for_context(face_bbox, 1.8, h, w)

        face_rgb = crop_face(img, face_bbox, self.img_size)
        context_rgb = crop_context(img, face_bbox, ctx_bbox, self.img_size)

        face_freq = rgb_to_dct_map(face_rgb, self.img_size)
        context_freq = rgb_to_dct_map(context_rgb, self.img_size)

        face_rgb = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        context_rgb = torch.from_numpy(context_rgb).permute(2, 0, 1).float() / 255.0
        face_freq = torch.from_numpy(face_freq)
        context_freq = torch.from_numpy(context_freq)

        return {
            'face_rgb': face_rgb,
            'context_rgb': context_rgb,
            'face_freq': face_freq,
            'context_freq': context_freq,
            'label': torch.tensor(label, dtype=torch.long),
        }


def get_dataloader(split, batch_size=64, num_workers=4, **kwargs):
    """Create DataLoader for DF40 dataset."""
    dataset = DF40Dataset(split=split, **kwargs)
    shuffle = split == 'train'
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
