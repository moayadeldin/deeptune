"""Consistent indexing for DeepAR training, evaluation, and embeddings."""
import numpy as np
import pandas as pd


def prepare_timeseries(frame, time_column, target_column, group_ids=None):
    frame = frame.copy()
    groups = list(group_ids) if group_ids else ['group']
    if not group_ids:
        frame['group'] = '0'
    for group in groups:
        if group not in frame or frame[group].isna().any():
            raise ValueError(f'Series group {group!r} must exist and have no missing values.')
        frame[group] = frame[group].astype(str)
    values = frame[time_column]
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors='coerce')
        if not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError('Numeric time indices must be finite integers.')
        frame['time_idx'] = numeric.astype('int64')
    else:
        timestamps = pd.to_datetime(values, errors='raise', utc=True)
        if timestamps.isna().any():
            raise ValueError('The time index cannot contain missing values.')
        # One step per observation timestamp, without collapsing minute/day
        # data into the same hour as the previous conversion did.
        frame['time_idx'] = pd.factorize(timestamps, sort=True)[0].astype('int64')
    frame[target_column] = pd.to_numeric(frame[target_column], errors='raise').astype('float64')
    if not np.isfinite(frame[target_column]).all():
        raise ValueError('Forecast targets must be finite numeric values.')
    if frame.duplicated([*groups, 'time_idx']).any():
        raise ValueError('Each series must have exactly one observation per time index.')
    return frame.sort_values([*groups, 'time_idx'], kind='stable'), groups
