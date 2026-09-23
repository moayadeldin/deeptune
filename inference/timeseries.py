"""Inference for DeepTune's timeseries modality (DeepAR).

DeepAR needs `max_encoder_length` rows of history immediately preceding
whatever it forecasts -- there is no way around that; a single new
observation carries no usable context on its own. This module supplies
that history from the original run's own train/validation splits (found
via desktop/config.py's find_history_paths), the same way
evaluators/timeseries/evaluate_deepar.py builds its test window, rather
than requiring the user to paste that much history into new data by hand.

If the new data has no target column (or it's all missing), a 0.0
placeholder is used so pytorch_forecasting's TimeSeriesDataSet -- which
needs the target column to exist and be numeric across the whole frame --
can still build the decoder window; DeepAR's own decoder-side forward pass
doesn't consume that placeholder value, only compares against it
afterwards for metrics, which is why has_labels is tracked separately and
those rows are reported with no ground truth rather than a fabricated one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def predict_timeseries(
    df: pd.DataFrame,
    checkpoint_path: Path,
    history_df_paths: list[Path],
    timeindex_column: str,
    target_column: str,
    batch_size: int,
    max_encoder_length: int = 30,
    group_ids: Optional[list] = None,
) -> pd.DataFrame:
    from datasets.timeseries_data import prepare_timeseries
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.models.deepar import DeepAR

    has_labels = target_column in df.columns and not df[target_column].isna().all()
    df = df.copy()
    if target_column not in df.columns:
        df[target_column] = 0.0
    else:
        df[target_column] = df[target_column].fillna(0.0)

    df['__new_row'] = np.arange(len(df))
    history = []
    for path in history_df_paths:
        previous = pd.read_parquet(path)
        previous['__new_row'] = -1
        history.append(previous)
    combined = pd.concat([*history, df], ignore_index=True)

    frame, groups = prepare_timeseries(combined, timeindex_column, target_column, group_ids)
    new_rows = frame[frame['__new_row'] >= 0].sort_values('__new_row')

    ckpt_path = Path(checkpoint_path)
    ckpt_path = ckpt_path if ckpt_path.suffix == '.ckpt' else next(ckpt_path.glob('*.ckpt'))
    model = DeepAR.load_from_checkpoint(ckpt_path)
    model.eval()

    # prepare_timeseries always names its output column 'time_idx' regardless
    # of what `timeindex_column` was called in the original data.
    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, frame, predict=False, stop_randomization=True,
        min_encoder_length=2, min_prediction_idx=int(new_rows['time_idx'].min()),
    )
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    import torch
    with torch.no_grad():
        pred = model.predict(loader, mode='prediction', return_x=True, return_index=True)

    predicted = pred.output.detach().cpu().numpy().reshape(-1)
    index_df = pred.index
    result = pd.DataFrame({
        'sample_id': index_df['time_idx'].tolist() if 'time_idx' in index_df.columns else list(range(len(predicted))),
        'predicted_value': predicted[:len(index_df)],
    })
    if has_labels:
        # Match each predicted row back to its true target by time_idx (and
        # group, if this run has more than one series) rather than assuming
        # row order, since TimeSeriesDataSet's own internal ordering isn't
        # guaranteed to match new_rows'.
        lookup_cols = [*groups, 'time_idx'] if groups != ['group'] else ['time_idx']
        true_lookup = new_rows.set_index([*groups, 'time_idx'] if groups != ['group'] else 'time_idx')[target_column]
        keys = index_df[[*groups, 'time_idx']] if groups != ['group'] else index_df['time_idx']
        true_values = []
        for _, row in index_df.iterrows():
            try:
                key = tuple(row[g] for g in groups) + (row['time_idx'],) if groups != ['group'] else row['time_idx']
                true_values.append(float(true_lookup.loc[key]))
            except (KeyError, ValueError):
                true_values.append(None)
        result['true_value'] = true_values
        result['abs_error'] = [None if t is None else abs(t - p) for t, p in zip(true_values, result['predicted_value'])]
    return result
