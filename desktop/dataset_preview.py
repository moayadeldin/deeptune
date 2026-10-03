"""Bounded dataset previews, safe to load on a worker without GUI imports."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from desktop.config import RunConfig
from handlers.video_manifest import VIDEO_EXTENSIONS, read_video_manifest
from handlers.split_planning import classification_split_issues


@dataclass
class DatasetPreview:
    note: str
    frame: pd.DataFrame | None = None
    class_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    thumbnail: tuple | None = None
    video_entries: dict[str, str] | None = None


def preview_video_file(path: Path) -> DatasetPreview:
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Could not open {path.name}. The file may be corrupt or its video codec unsupported.')
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f'No video frame could be decoded from {path.name}.')
        width, height = frame.shape[1], frame.shape[0]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = int(reported_frames) if np.isfinite(reported_frames) else 0
        duration = f'{frames / fps:.1f} seconds' if fps > 0 and np.isfinite(fps) and frames > 0 else 'Duration unavailable'
        scale = min(480 / width, 240 / height, 1)
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        pixels = cv2.cvtColor(preview, cv2.COLOR_BGR2RGBA).astype(np.float32).ravel() / 255
        rate = f'{fps:.2f} fps' if fps > 0 and np.isfinite(fps) else 'Frame rate unavailable'
        note = f'{path.name}  |  {width} x {height}  |  {rate}  |  {duration}'
        return DatasetPreview(note=note, thumbnail=(preview.shape[1], preview.shape[0], pixels),
                              warnings=['To train, give clips a class label in the clip list and click Use clip dataset.'])
    finally:
        capture.release()


def preview_dataset(cfg: RunConfig, limit: int = 50) -> DatasetPreview:
    path = Path(cfg.df).expanduser()
    if cfg.modality == 'video' and path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return preview_video_file(path)
    if cfg.modality == 'video' and cfg.raw_data and path.is_file() and path.suffix.lower() in ('.csv', '.xlsx'):
        frame = read_video_manifest(path)
        count = frame['labels'].nunique() if cfg.mode == 'cls' else None
        issues = classification_split_issues(frame['labels'].value_counts().to_dict(), unit='clips') if cfg.mode == 'cls' else []
        return DatasetPreview(f'{len(frame):,} labeled clips' + (f' across {count} classes.' if count is not None else '.'),
                              frame.head(limit), count, warnings=issues, video_entries=dict(zip(frame.videos, frame.labels.astype(str))))
    if cfg.raw_data and cfg.modality in ('images', 'video'):
        extensions = ({'.png', '.jpg', '.jpeg'} if cfg.modality == 'images' else
                      {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg'})
        counts: dict[str, int] = {}
        for entry in sorted(path.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            children = sorted(child for child in entry.iterdir() if child.is_dir() and not child.name.startswith('.'))
            for class_dir in children or [entry]:
                count = sum(item.is_file() and item.suffix.lower() in extensions for item in class_dir.iterdir())
                counts[class_dir.name] = counts.get(class_dir.name, 0) + count
        if not sum(counts.values()):
            raise ValueError('No supported files found. Use a folder containing one subfolder per class.')
        frame = pd.DataFrame(sorted(counts.items()), columns=['Class', 'Files'])
        warnings = classification_split_issues(counts, unit='clips' if cfg.modality == 'video' else 'images')
        return DatasetPreview(f'{sum(counts.values()):,} files across {sum(v > 0 for v in counts.values())} classes.',
                              frame, sum(v > 0 for v in counts.values()), warnings)
    if path.suffix.lower() == '.parquet' and not cfg.raw_data:
        import pyarrow.parquet as pq
        file = pq.ParquetFile(path)
        # Read only one small batch. Never materialize all image/video bytes for a preview.
        batch = next(file.iter_batches(batch_size=limit), None)
        frame = batch.to_pandas() if batch is not None else pd.DataFrame(columns=file.schema_arrow.names)
        note = f'{file.metadata.num_rows:,} rows, {len(file.schema_arrow.names)} columns. Showing up to {limit} rows.'
    elif cfg.raw_data and path.suffix.lower() == '.csv':
        frame = pd.read_csv(path, nrows=limit)
        note = f'Preview of up to {limit} rows, {len(frame.columns)} columns.'
    elif cfg.raw_data and path.suffix.lower() == '.xlsx':
        frame = pd.read_excel(path, nrows=limit)
        note = f'Preview of up to {limit} rows, {len(frame.columns)} columns.'
    else:
        raise ValueError('Choose CSV/XLSX for raw table data, or turn off raw data for a parquet file.')
    warnings = []
    if cfg.target not in frame:
        warnings.append(f'Target column {cfg.target!r} is missing.')
    elif frame[cfg.target].isna().any():
        warnings.append('The preview contains missing target values; fill or remove them before training.')
    expected = {'images': 'images', 'video': 'videos', 'text': 'text'}.get(cfg.modality)
    if expected and expected not in frame:
        warnings.append(f'This modality needs a {expected!r} column.')
    if cfg.grouper and cfg.grouper not in frame:
        warnings.append(f'Grouper column {cfg.grouper!r} is missing.')
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: f'<{len(value):,} bytes>' if isinstance(value, (bytes, bytearray)) else value)
    return DatasetPreview(note, frame, warnings=warnings)
