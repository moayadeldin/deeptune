"""Shared helpers across every modality's inference module."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def summarize_predictions(predictions: pd.DataFrame, mode: str) -> Optional[dict]:
    """Aggregate accuracy (classification) or MAE/RMSE (regression) across
    `predictions`, or None if it has no ground truth to compare against."""
    if mode == 'cls' and 'correct' in predictions.columns:
        known = predictions['correct'].dropna()
        if known.empty:
            return None
        return {'accuracy': float(known.mean()), 'num_labeled': int(len(known)), 'num_samples': int(len(predictions))}
    if mode == 'reg' and 'abs_error' in predictions.columns:
        known = predictions['abs_error'].dropna()
        if known.empty:
            return None
        errors = known.astype(float)
        return {'mae': float(errors.mean()), 'rmse': float((errors ** 2).mean() ** 0.5),
                'num_labeled': int(len(known)), 'num_samples': int(len(predictions))}
    return None
