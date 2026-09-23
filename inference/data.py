"""Building an inference-ready DataFrame from NEW data (a folder or file the
Test page points at), for the images/video modalities.

Deliberately separate from handlers/raw_to_parquet_dataset.py's loaders:
those require every top-level entry of a folder to be a class (or a split
wrapping classes) and raise if no subdirectories exist at all, because
they're built for training, where labels are mandatory. Testing a model
against a flat folder of files with no known answer needs to work too, so
the loaders here fall back to an unlabeled read instead of raising. Each
returned row also keeps the source file's path as `__sample_id`, which
raw_to_parquet_dataset's loaders don't track, so predictions can be matched
back to the file that produced them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg')


def _iter_files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def load_labeled_or_flat_files(data_path: Path, extensions: tuple[str, ...], bytes_col: str) -> tuple[pd.DataFrame, bool]:
    """Reads raw image/video files from `data_path` for inference.

    Follows the same "every top-level entry is either a class, or a split
    wrapping classes" convention as raw_to_parquet_dataset.py when
    `data_path` has subdirectories (returned with a 'labels' column, so
    predictions can be compared against these folder-derived labels), and
    falls back to a flat, unlabeled read of every recognised file directly
    inside `data_path` otherwise.

    Returns (dataframe, has_labels).
    """
    data_path = Path(data_path)
    if not data_path.is_dir():
        raise ValueError(f'{data_path} is not a folder.')

    top_level = sorted(e for e in data_path.iterdir() if e.is_dir() and not e.name.startswith('.'))
    rows: dict[str, list] = {bytes_col: [], 'labels': [], '__sample_id': []}
    for entry in top_level:
        class_dirs = sorted(c for c in entry.iterdir() if c.is_dir() and not c.name.startswith('.'))
        if class_dirs:
            for class_dir in class_dirs:
                for file in _iter_files(class_dir, extensions):
                    rows[bytes_col].append(file.read_bytes())
                    rows['labels'].append(class_dir.name)
                    rows['__sample_id'].append(str(file))
        else:
            for file in _iter_files(entry, extensions):
                rows[bytes_col].append(file.read_bytes())
                rows['labels'].append(entry.name)
                rows['__sample_id'].append(str(file))
    if rows[bytes_col]:
        return pd.DataFrame(rows), True

    # No subdirectories, or they held nothing recognised: treat data_path
    # itself as one flat, unlabeled folder of files.
    files = _iter_files(data_path, extensions)
    if not files:
        raise ValueError(f'No supported files found under {data_path}.')
    return pd.DataFrame({
        bytes_col: [f.read_bytes() for f in files],
        '__sample_id': [str(f) for f in files],
    }), False


def load_single_file(path: Path, bytes_col: str) -> pd.DataFrame:
    """Reads one image/video file for inference -- the common "I have one
    new clip, what does the model think it is" case, distinct from both the
    labeled-folder and flat-unlabeled-folder cases load_labeled_or_flat_files
    handles: a lone file has no folder name to read a label from, so this
    always returns unlabeled (a single-row DataFrame with no 'labels'
    column, matching what the caller already does for an unlabeled folder).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} is not a file.')
    return pd.DataFrame({bytes_col: [path.read_bytes()], '__sample_id': [str(path)]})


def load_video_manifest_for_inference(path: Path) -> tuple[pd.DataFrame, bool]:
    """Reads a CSV/XLSX list of video file paths for inference. Like
    handlers/video_manifest.py's read_video_manifest, but a 'labels' column
    is optional here rather than mandatory -- inference data may genuinely
    have no known answer to compare against.
    """
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
    if 'videos' not in frame.columns:
        raise ValueError("A video list needs a 'videos' column of file paths.")
    if frame.empty:
        raise ValueError('The video list is empty.')
    has_labels = 'labels' in frame.columns and not frame['labels'].isna().any()

    resolved: list[str] = []
    for index, value in enumerate(frame['videos']):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Video list row {index + 1} has no file path.')
        clip = Path(value).expanduser()
        clip = (path.parent / clip).resolve() if not clip.is_absolute() else clip.resolve()
        if not clip.is_file():
            raise ValueError(f'Video list row {index + 1}: file not found: {clip}')
        if clip.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f'Video list row {index + 1}: unsupported video extension: {clip.suffix}')
        resolved.append(str(clip))

    frame = frame.copy()
    frame['__sample_id'] = resolved
    frame['videos'] = [Path(p).read_bytes() for p in resolved]
    if not has_labels and 'labels' in frame.columns:
        frame = frame.drop(columns=['labels'])
    return frame, has_labels


def load_tabular_like(data_path: Path) -> pd.DataFrame:
    """Reads a CSV/XLSX/parquet file for text/tabular/timeseries inference.
    Column conventions (which column is text, which are features, which is
    the target/time index) are the caller's concern -- this just gets a
    DataFrame with a __sample_id column into memory, the same as
    handlers/raw_to_parquet_dataset.py does for training, minus the
    mandatory-target-column assumption.
    """
    data_path = Path(data_path)
    suffix = data_path.suffix.lower()
    if suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif suffix == '.csv':
        df = pd.read_csv(data_path)
    elif suffix == '.xlsx':
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f'Unsupported file type: {data_path.suffix}. Use a CSV, XLSX, or parquet file.')
    if df.empty:
        raise ValueError(f'{data_path} has no rows.')
    df = df.reset_index(drop=True)
    if '__sample_id' not in df.columns:
        df.insert(0, '__sample_id', df.index.astype(str))
    return df
