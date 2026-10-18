import cv2
import numpy as np
import pandas as pd
import torch
import pytest

from datasets.video_datasets import ParquetVideoDataset


def test_invalid_video_reports_row_and_releases_temp_file(tmp_path):
    dataset = ParquetVideoDataset(pd.DataFrame({'videos': [b'broken clip'], 'labels': [0]}))
    with pytest.raises(ValueError, match='Video row 0'):
        dataset[0]


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_invalid_frame_count_is_rejected(count):
    with pytest.raises(ValueError, match='positive integer'):
        ParquetVideoDataset(pd.DataFrame({'videos': []}), num_frames=count)


def test_fractional_video_targets_are_preserved(tmp_path):
    clip = _make_clip_bytes(tmp_path, 'regression.mp4')
    dataset = ParquetVideoDataset(pd.DataFrame({'videos': [clip], 'labels': [1.75]}), num_frames=2)
    assert dataset[0][1].item() == pytest.approx(1.75)


@pytest.mark.parametrize('version,size', [('r3d_18', 112), ('mvit_v2_s', 224), ('swin3d_t', 224)])
def test_native_backbones_receive_their_weight_preprocessing(tmp_path, version, size):
    from datasets.video_datasets import native_video_transform
    clip = _make_clip_bytes(tmp_path, 'native.mp4')
    dataset = ParquetVideoDataset(pd.DataFrame({'videos': [clip], 'labels': [0]}), num_frames=2,
                                   clip_transform=native_video_transform(version))
    tensor, _ = dataset[0]
    assert tensor.shape == (2, 3, size, size)
    assert torch.isfinite(tensor).all()


def test_unlabelled_video_embedding_batches(tmp_path):
    from torch.utils.data import DataLoader
    from datasets.video_datasets import collate_video_samples
    clip = _make_clip_bytes(tmp_path, 'unlabelled.mp4')
    dataset = ParquetVideoDataset(pd.DataFrame({'videos': [clip, clip]}), num_frames=2)
    clips, labels = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_video_samples)))
    assert clips.shape[:3] == (2, 2, 3)
    assert labels is None


def _make_clip_bytes(tmp_path, name: str, num_raw_frames: int = 10, size=(48, 48)) -> bytes:
    """
    Writes a tiny synthetic .mp4 clip to disk and returns its raw bytes, mirroring
    how DeepTune stores video clips in a parquet "videos" column.
    """
    path = tmp_path / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(num_raw_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_parquet_video_dataset_samples_fixed_frame_count(tmp_path):
    clip_bytes = _make_clip_bytes(tmp_path, "clip.mp4", num_raw_frames=10)

    df = pd.DataFrame({"videos": [clip_bytes, clip_bytes], "labels": [0, 1]})

    dataset = ParquetVideoDataset(df, transform=None, num_frames=4)

    assert len(dataset) == 2

    clip, label = dataset[0]

    assert clip.shape == (4, 3, 48, 48), "Should sample exactly num_frames frames, each RGB HxWxC permuted to CxHxW"
    assert clip.dtype == torch.float32
    assert label.item() == 0


def test_parquet_video_dataset_with_transform(tmp_path):
    from helpers import transformations

    clip_bytes = _make_clip_bytes(tmp_path, "clip.mp4", num_raw_frames=6)
    df = pd.DataFrame({"videos": [clip_bytes], "labels": [1]})

    dataset = ParquetVideoDataset(df, transform=transformations, num_frames=3)
    clip, label = dataset[0]

    # DeepTune's shared vision transform resizes every frame to 224x224.
    assert clip.shape == (3, 3, 224, 224)
    assert label.item() == 1


def test_parquet_video_dataset_without_labels(tmp_path):
    clip_bytes = _make_clip_bytes(tmp_path, "clip.mp4", num_raw_frames=5)
    df = pd.DataFrame({"videos": [clip_bytes]})

    dataset = ParquetVideoDataset(df, transform=None, num_frames=2)
    clip, label = dataset[0]

    assert clip.shape == (2, 3, 48, 48)
    assert label is None


def test_parquet_video_dataset_from_parquet_roundtrip(tmp_path):
    clip_bytes = _make_clip_bytes(tmp_path, "clip.mp4", num_raw_frames=5)
    df = pd.DataFrame({"videos": [clip_bytes], "labels": [0]})

    parquet_path = tmp_path / "videos.parquet"
    df.to_parquet(parquet_path)

    dataset = ParquetVideoDataset.from_parquet(parquet_path, transform=None, num_frames=2)
    clip, label = dataset[0]

    assert clip.shape == (2, 3, 48, 48)
    assert label.item() == 0
