"""Read a labeled list of local video files without loading the ML stack."""
from pathlib import Path
import os

import pandas as pd

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg'}


def read_video_manifest(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
    if 'videos' not in frame or 'labels' not in frame:
        raise ValueError("A video list needs 'videos' (file paths) and 'labels' columns.")
    if frame.empty or frame['labels'].isna().any():
        raise ValueError('Add at least one clip and give every clip a label.')
    paths = []
    for index, value in enumerate(frame['videos']):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Video list row {index + 1} has no file path.')
        clip = Path(value).expanduser()
        clip = (path.parent / clip).resolve() if not clip.is_absolute() else clip.resolve()
        if not clip.is_file():
            raise ValueError(f'Video list row {index + 1}: file not found: {clip}')
        if clip.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f'Video list row {index + 1}: unsupported video extension: {clip.suffix}')
        paths.append(str(clip))
    if len({os.path.normcase(value) for value in paths}) != len(paths):
        raise ValueError('The same video appears more than once. Remove duplicates before training.')
    frame['videos'] = paths
    return frame


def load_video_manifest(path: Path) -> pd.DataFrame:
    frame = read_video_manifest(path)
    frame['videos'] = frame['videos'].map(lambda name: Path(name).read_bytes())
    return frame
