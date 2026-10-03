import io
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset


class VideoPresetTransform:
    """Apply the native backbone's Kinetics preprocessing to a whole clip."""
    def __init__(self, model_version: str):
        from src.video.video3d import _KINETICS400_WEIGHTS
        self.transform = _KINETICS400_WEIGHTS[model_version].transforms()

    def __call__(self, clip):
        # torchvision returns C,T,H,W; DeepTune datasets expose T,C,H,W.
        return self.transform(clip).permute(1, 0, 2, 3)


def native_video_transform(model_version: str):
    from src.video.video3d import VIDEO3D_ARCHITECTURES
    return VideoPresetTransform(model_version) if model_version in VIDEO3D_ARCHITECTURES else None


def collate_video_samples(samples):
    """The default collator rejects None labels on embedding-only datasets."""
    from torch.utils.data import default_collate
    clips = default_collate([sample[0] for sample in samples])
    labels = None if all(sample[1] is None for sample in samples) else default_collate([sample[1] for sample in samples])
    if len(samples[0]) == 3:
        return clips, labels, default_collate([sample[2] for sample in samples])
    return clips, labels


class ParquetVideoDataset(Dataset):
    """
    Loads the videos and labels from a parquet file we feed to the model.

    Each video clip is stored as raw bytes (the same convention DeepTune uses
    for images), and is decoded on the fly. A fixed number of frames is
    sampled uniformly across the clip's duration, so every sample produces a
    tensor of the same shape regardless of the original clip's length or
    frame rate.

    Attributes:

        data (pd.DataFrame): The DataFrame containing video byte data and labels.
        transform: Per-frame transformation applied to every sampled frame (DeepTune
            reuses the same ImageNet-style transform already used across its vision pipeline).
        num_frames (int): Number of frames uniformly sampled from each video clip.
        video_bytes (np.ndarray): The videos in bytes format.
        labels (np.ndarray): The labels of corresponding video.

    """

    def __init__(self, df: DataFrame, transform=None, num_frames: int = 8, clip_transform=None):
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError('num_frames must be a positive integer.')
        if 'videos' not in df.columns:
            raise ValueError("Video datasets must have a 'videos' column containing clip bytes.")
        self.data = df
        self.transform = transform
        self.clip_transform = clip_transform
        self.num_frames = num_frames
        self.video_bytes = self.data['videos'].values
        self.has_labels = 'labels' in df.columns
        self.labels = self.data['labels'].values if self.has_labels else None

    @classmethod
    def from_parquet(cls, parquet_file, transform=None, num_frames: int = 8, clip_transform=None) -> "ParquetVideoDataset":
        df = pd.read_parquet(parquet_file)
        return cls(df, transform, num_frames, clip_transform)

    def __len__(self):

        """
        Returns the number of samples in dataset.
        """
        return len(self.video_bytes)

    def _sample_frames(self, video_bytes: bytes) -> list:
        """
        Decodes a video clip from raw bytes and uniformly samples `self.num_frames`
        frames across its duration.

        OpenCV cannot decode a video from an in-memory buffer directly, so the clip
        is first materialized to a temporary file, which is removed again once
        decoding is done.

        Returns:
            list[np.ndarray]: `self.num_frames` RGB uint8 frames, sized HxWx3.
        """

        if not isinstance(video_bytes, (bytes, bytearray, memoryview)) or not video_bytes:
            raise ValueError('Each video must contain nonempty encoded clip bytes.')
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = None
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError('Could not open this video. The clip may be corrupt or its codec unsupported.')
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                # Some containers don't report a reliable frame count upfront,
                # so fall back to reading the clip sequentially.
                frames = []
                ok, frame = cap.read()
                while ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ok, frame = cap.read()
                cap.release()

                if not frames:
                    raise ValueError("Could not decode any frames from the provided video.")

                indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
                return [frames[i] for i in indices]

            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            sampled = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    # Never manufacture black training data for an undecodable
                    # clip. A failed later seek can reuse a decoded frame.
                    if not sampled:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError('Could not decode any frames from this video.')
                        sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        sampled.append(sampled[-1].copy())
                    continue
                sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return sampled
        finally:
            if cap is not None:
                cap.release()
            os.unlink(tmp_path)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        """
        Loads and processes the video and label at a given index idx.
        """
        video_bytes = row['videos']
        try:
            frames = self._sample_frames(video_bytes)
        except ValueError as exc:
            raise ValueError(f'Video row {idx}: {exc}') from exc
        label = torch.tensor(row['labels'], dtype=torch.float32 if pd.api.types.is_float_dtype(self.data['labels']) else torch.long) if self.has_labels else None
        extras = row.drop(labels=['videos', 'labels'], errors='ignore').to_dict()

        if self.clip_transform is not None:
            clip = self.clip_transform(torch.stack([torch.from_numpy(frame).permute(2, 0, 1) for frame in frames]))
            return (clip, label, extras) if extras else (clip, label)
        if self.transform:
            processed_frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        else:
            processed_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]

        clip = torch.stack(processed_frames, dim=0)  # [num_frames, C, H, W]

        if extras:
            return clip, label, extras
        else:
            return clip, label
