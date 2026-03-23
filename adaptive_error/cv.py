from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


CV_RANDOM_STATE = 42


def resolve_groups(groups: Any | None, n_samples: int) -> np.ndarray | None:
    if groups is None:
        return None

    group_array = np.asarray(groups, dtype=object).ravel()
    if group_array.shape[0] != int(n_samples):
        raise ValueError("groups must contain one value per sample.")
    return group_array


def _candidate_split_counts(
    y_true: np.ndarray,
    *,
    n_splits: int,
    groups: np.ndarray | None = None,
) -> list[int]:
    y = np.asarray(y_true).ravel()
    n_samples = int(y.shape[0])
    if n_samples < 2:
        return []

    requested = max(2, int(n_splits))
    max_splits = min(requested, n_samples)

    _, class_counts = np.unique(y, return_counts=True)
    if class_counts.size == 0:
        return []
    max_splits = min(max_splits, int(np.min(class_counts)))

    if groups is not None:
        max_splits = min(max_splits, int(np.unique(groups).size))

    if max_splits < 2:
        return []
    return list(range(max_splits, 1, -1))


def build_stratified_cv_splits(
    y_true: Any,
    *,
    groups: Any | None = None,
    n_splits: int,
    random_state: int = CV_RANDOM_STATE,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, dict[str, Any], list[dict[str, Any]]]:
    y = np.asarray(y_true).ravel()
    n_samples = int(y.shape[0])
    group_array = resolve_groups(groups, n_samples) if groups is not None else None

    cv_info: dict[str, Any] = {
        "requested_n_splits": max(2, int(n_splits)),
        "n_splits": None,
        "random_state": int(random_state),
        "requested_grouped": group_array is not None,
        "grouped": False,
        "splitter": None,
        "fallback": None,
        "error": None,
    }
    warnings: list[dict[str, Any]] = []

    candidate_counts = _candidate_split_counts(
        y,
        n_splits=n_splits,
        groups=group_array,
    )
    if not candidate_counts:
        cv_info["fallback"] = "full_fit"
        cv_info["error"] = "insufficient_samples_for_stratified_cv"
        return None, cv_info, warnings

    x_dummy = np.zeros((n_samples, 1), dtype=float)
    last_group_error: str | None = None
    last_ungrouped_error: str | None = None

    if group_array is not None:
        for split_count in candidate_counts:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=int(split_count),
                    shuffle=True,
                    random_state=int(random_state),
                )
                splits = [
                    (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
                    for train_idx, valid_idx in splitter.split(
                        x_dummy,
                        y,
                        groups=group_array,
                    )
                ]
                if splits:
                    cv_info.update(
                        {
                            "n_splits": int(split_count),
                            "grouped": True,
                            "splitter": "StratifiedGroupKFold",
                        }
                    )
                    return splits, cv_info, warnings
            except Exception as exc:
                last_group_error = str(exc)

        warnings.append(
            {
                "type": "grouped_cv_fallback",
                "message": (
                    "Grouped stratified CV could not be constructed; "
                    "falling back to StratifiedKFold."
                ),
                "error": last_group_error,
            }
        )

    for split_count in candidate_counts:
        try:
            splitter = StratifiedKFold(
                n_splits=int(split_count),
                shuffle=True,
                random_state=int(random_state),
            )
            splits = [
                (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
                for train_idx, valid_idx in splitter.split(x_dummy, y)
            ]
            if splits:
                cv_info.update(
                    {
                        "n_splits": int(split_count),
                        "grouped": False,
                        "splitter": "StratifiedKFold",
                        "fallback": None if group_array is None else "grouped_to_ungrouped",
                        "error": last_group_error,
                    }
                )
                return splits, cv_info, warnings
        except Exception as exc:
            last_ungrouped_error = str(exc)

    cv_info["fallback"] = "full_fit"
    cv_info["error"] = last_ungrouped_error or last_group_error
    return None, cv_info, warnings
