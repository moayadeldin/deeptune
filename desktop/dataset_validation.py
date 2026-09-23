"""Read video labels for preflight checks without decoding media or loading torch."""
from functools import lru_cache
from pathlib import Path

from handlers.split_planning import classification_split_issues


@lru_cache(maxsize=32)
def _file_split_issues(path: str, modified: int, size: int, target: str) -> tuple[str, ...]:
    import pandas as pd
    source = Path(path)
    if source.suffix.lower() == '.csv':
        # Video bytes are not in these lists. Read only the label column.
        frame = pd.read_csv(source, usecols=[target])
    elif source.suffix.lower() == '.xlsx':
        frame = pd.read_excel(source, usecols=[target])
    else:
        frame = pd.read_parquet(source, columns=[target])
    if frame.empty:
        return ('The video dataset is empty. Add labeled clips first.',)
    if frame[target].isna().any():
        return ('Every video clip needs a label before training.',)
    return tuple(classification_split_issues(frame[target].value_counts().to_dict(), unit='clips'))


def video_file_split_issues(cfg) -> list[str]:
    if cfg.modality != 'video' or cfg.mode != 'cls' or cfg.df is None:
        return []
    source = Path(cfg.df)
    allowed = ('.csv', '.xlsx') if cfg.raw_data else ('.parquet',)
    if not source.is_file() or source.suffix.lower() not in allowed:
        return []
    try:
        metadata = source.stat()
        return list(_file_split_issues(str(source.resolve()), metadata.st_mtime_ns, metadata.st_size, cfg.target))
    except Exception as exc:
        return [f'Could not check video labels in {source.name}: {exc}']
