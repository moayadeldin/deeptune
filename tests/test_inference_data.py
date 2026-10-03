"""inference/data.py builds the in-memory DataFrame the Test page's new
data feeds into inference/vision.py, inference/video.py, etc. -- separate
from handlers/raw_to_parquet_dataset.py's training-time loaders because
inference must also work against a flat, unlabeled folder (no known
answer to predict against), which those loaders reject outright.
"""
from pathlib import Path

import pandas as pd
import pytest

from inference.data import (
    IMAGE_EXTENSIONS,
    load_labeled_or_flat_files,
    load_single_file,
    load_tabular_like,
    load_video_manifest_for_inference,
)


def _write_png(path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)


def test_load_labeled_or_flat_files_reads_class_subfolders(tmp_path):
    _write_png(tmp_path / "cat" / "a.png")
    _write_png(tmp_path / "cat" / "b.png")
    _write_png(tmp_path / "dog" / "c.png")

    df, has_labels = load_labeled_or_flat_files(tmp_path, IMAGE_EXTENSIONS, "images")
    assert has_labels is True
    assert len(df) == 3
    assert set(df["labels"]) == {"cat", "dog"}
    assert all(isinstance(b, bytes) for b in df["images"])
    assert all(str(tmp_path) in s for s in df["__sample_id"])


def test_load_labeled_or_flat_files_reads_split_wrapped_classes(tmp_path):
    _write_png(tmp_path / "train" / "cat" / "a.png")
    _write_png(tmp_path / "val" / "dog" / "b.png")

    df, has_labels = load_labeled_or_flat_files(tmp_path, IMAGE_EXTENSIONS, "images")
    assert has_labels is True
    assert set(df["labels"]) == {"cat", "dog"}


def test_load_labeled_or_flat_files_falls_back_to_unlabeled_flat_folder(tmp_path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")

    df, has_labels = load_labeled_or_flat_files(tmp_path, IMAGE_EXTENSIONS, "images")
    assert has_labels is False
    assert "labels" not in df.columns
    assert len(df) == 2


def test_load_labeled_or_flat_files_raises_on_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        load_labeled_or_flat_files(tmp_path, IMAGE_EXTENSIONS, "images")


def test_load_labeled_or_flat_files_raises_on_non_directory(tmp_path):
    data = tmp_path / "not_a_dir.csv"
    data.write_text("x")
    with pytest.raises(ValueError):
        load_labeled_or_flat_files(data, IMAGE_EXTENSIONS, "images")


def test_load_single_file_reads_one_file_unlabeled(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video-bytes")
    df = load_single_file(clip, "videos")
    assert len(df) == 1
    assert "labels" not in df.columns
    assert df["videos"].iloc[0] == b"fake-video-bytes"
    assert df["__sample_id"].iloc[0] == str(clip)


def test_load_single_file_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError):
        load_single_file(tmp_path / "does_not_exist.mp4", "videos")


def test_load_single_file_raises_on_directory(tmp_path):
    with pytest.raises(ValueError):
        load_single_file(tmp_path, "videos")


def test_load_video_manifest_for_inference_labeled(tmp_path):
    clip = tmp_path / "clip1.mp4"
    clip.write_bytes(b"fake-video-bytes")
    manifest = tmp_path / "list.csv"
    pd.DataFrame({"videos": ["clip1.mp4"], "labels": ["walking"]}).to_csv(manifest, index=False)

    df, has_labels = load_video_manifest_for_inference(manifest)
    assert has_labels is True
    assert df["videos"].iloc[0] == b"fake-video-bytes"
    assert df["labels"].iloc[0] == "walking"


def test_load_video_manifest_for_inference_unlabeled(tmp_path):
    clip = tmp_path / "clip1.mp4"
    clip.write_bytes(b"fake-video-bytes")
    manifest = tmp_path / "list.csv"
    pd.DataFrame({"videos": ["clip1.mp4"]}).to_csv(manifest, index=False)

    df, has_labels = load_video_manifest_for_inference(manifest)
    assert has_labels is False
    assert "labels" not in df.columns


def test_load_video_manifest_for_inference_missing_file_raises(tmp_path):
    manifest = tmp_path / "list.csv"
    pd.DataFrame({"videos": ["does_not_exist.mp4"]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        load_video_manifest_for_inference(manifest)


def test_load_tabular_like_reads_csv_and_adds_sample_id(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feat1": [1, 2], "labels": [0, 1]}).to_csv(csv_path, index=False)

    df = load_tabular_like(csv_path)
    assert "__sample_id" in df.columns
    assert list(df["__sample_id"]) == ["0", "1"]


def test_load_tabular_like_reads_parquet(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    pd.DataFrame({"feat1": [1, 2]}).to_parquet(parquet_path)
    df = load_tabular_like(parquet_path)
    assert len(df) == 2


def test_load_tabular_like_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_tabular_like(path)


def test_load_tabular_like_rejects_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame({"feat1": []}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_tabular_like(csv_path)
