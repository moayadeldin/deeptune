from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .cv import CV_RANDOM_STATE, build_stratified_cv_splits, resolve_groups
from .core import (
    AdaptiveErrorCalculator,
    confidence_from_proba,
    confidence_from_proba_for_pred,
    proba_for_predicted_label,
)

ConfidenceMetricName = Literal[
    "margin_for_raw_pred",
    "pmax_for_raw_pred",
    "top1_minus_top2",
]
CONFIDENCE_SELECTION_SPLITS = 5


@dataclass(frozen=True)
class ConfidenceMetricScore:
    name: ConfidenceMetricName
    brier: float
    n_valid: int


@dataclass(frozen=True)
class ConfidenceMetricSelection:
    selected: ConfidenceMetricName
    scores: list[ConfidenceMetricScore]

    def to_json(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "scores": [
                {
                    "name": score.name,
                    "brier": score.brier,
                    "n_valid": score.n_valid,
                }
                for score in self.scores
            ],
            "selection_mode": "auto",
        }


def build_confidence_candidates(
    proba: Any,
    predicted_labels: Any,
    class_labels: list[Any],
) -> dict[ConfidenceMetricName, np.ndarray]:
    return {
        "margin_for_raw_pred": confidence_from_proba_for_pred(
            proba,
            predicted_labels,
            class_labels,
        ),
        "pmax_for_raw_pred": proba_for_predicted_label(
            proba,
            predicted_labels,
            class_labels,
        ),
        "top1_minus_top2": confidence_from_proba(proba),
    }


def compute_confidence(
    metric_name: ConfidenceMetricName,
    proba: Any,
    predicted_labels: Any,
    class_labels: list[Any],
) -> np.ndarray:
    candidates = build_confidence_candidates(
        proba,
        predicted_labels,
        class_labels,
    )
    if metric_name not in candidates:
        raise ValueError(f"Unsupported confidence metric: {metric_name}")
    return np.asarray(candidates[metric_name], dtype=float)


def _brier_score(predicted_error: np.ndarray, incorrect: np.ndarray) -> float:
    pred = np.clip(np.asarray(predicted_error, dtype=float).ravel(), 0.0, 1.0)
    obs = np.asarray(incorrect, dtype=float).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError("predicted_error and incorrect must have the same length.")
    valid_mask = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(valid_mask):
        return float("nan")
    return float(np.mean((pred[valid_mask] - obs[valid_mask]) ** 2))


def _crossfit_predicted_error(
    confidence: Any,
    incorrect: Any,
    *,
    groups: Any | None = None,
    aer_kwargs: dict[str, Any],
    n_splits: int = CONFIDENCE_SELECTION_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> np.ndarray:
    conf = np.asarray(confidence, dtype=float).ravel()
    inc = np.asarray(incorrect, dtype=int).ravel()
    if conf.shape[0] != inc.shape[0]:
        raise ValueError("confidence and incorrect must have the same length.")

    group_array = resolve_groups(groups, conf.shape[0]) if groups is not None else None
    valid_mask = np.isfinite(conf) & np.isfinite(inc)
    if group_array is not None:
        valid_mask &= ~pd.isna(group_array)

    valid_index = np.where(valid_mask)[0]
    if valid_index.size == 0:
        return np.full(conf.shape[0], np.nan, dtype=float)

    conf_valid = conf[valid_index]
    inc_valid = inc[valid_index]
    groups_valid = group_array[valid_index] if group_array is not None else None
    splits, _, _ = build_stratified_cv_splits(
        inc_valid,
        groups=groups_valid,
        n_splits=n_splits,
        random_state=random_state,
    )

    if not splits:
        aer = AdaptiveErrorCalculator(**aer_kwargs)
        aer.fit(conf_valid, inc_valid)
        predicted = np.full(conf.shape[0], np.nan, dtype=float)
        predicted[valid_index] = aer.get_expected_error(conf_valid)
        return predicted

    predicted_valid = np.full(conf_valid.shape[0], np.nan, dtype=float)
    for train_idx, valid_idx in splits:
        aer = AdaptiveErrorCalculator(**aer_kwargs)
        aer.fit(conf_valid[train_idx], inc_valid[train_idx])
        predicted_valid[valid_idx] = aer.get_expected_error(conf_valid[valid_idx])

    if np.isnan(predicted_valid).any():
        aer = AdaptiveErrorCalculator(**aer_kwargs)
        aer.fit(conf_valid, inc_valid)
        missing = np.isnan(predicted_valid)
        predicted_valid[missing] = aer.get_expected_error(conf_valid[missing])

    predicted = np.full(conf.shape[0], np.nan, dtype=float)
    predicted[valid_index] = predicted_valid
    return predicted


def select_confidence_metric(
    candidates: dict[ConfidenceMetricName, np.ndarray],
    incorrect: Any,
    *,
    groups: Any | None = None,
    aer_kwargs: dict[str, Any],
    n_splits: int = CONFIDENCE_SELECTION_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> ConfidenceMetricSelection:
    inc = np.asarray(incorrect, dtype=int).ravel()
    if not candidates:
        raise ValueError("At least one confidence candidate is required.")

    candidate_names = list(candidates.keys())
    candidate_order = {name: idx for idx, name in enumerate(candidate_names)}
    scores: list[ConfidenceMetricScore] = []

    for name, confidence in candidates.items():
        conf = np.asarray(confidence, dtype=float).ravel()
        if conf.shape[0] != inc.shape[0]:
            raise ValueError(
                f"Confidence candidate '{name}' has {conf.shape[0]} rows, expected {inc.shape[0]}."
            )

        valid_mask = np.isfinite(conf)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            scores.append(ConfidenceMetricScore(name=name, brier=float("nan"), n_valid=0))
            continue

        predicted_error = _crossfit_predicted_error(
            conf,
            inc,
            groups=groups,
            aer_kwargs=aer_kwargs,
            n_splits=n_splits,
            random_state=random_state,
        )
        scores.append(
            ConfidenceMetricScore(
                name=name,
                brier=_brier_score(predicted_error[valid_mask], inc[valid_mask]),
                n_valid=n_valid,
            )
        )

    finite_scores = [score for score in scores if np.isfinite(score.brier)]
    if finite_scores:
        best = min(
            finite_scores,
            key=lambda score: (score.brier, candidate_order[score.name]),
        )
        selected = best.name
    else:
        selected = candidate_names[0]

    ordered_scores = sorted(
        scores,
        key=lambda score: (
            not np.isfinite(score.brier),
            score.brier if np.isfinite(score.brier) else float("inf"),
            candidate_order[score.name],
        ),
    )
    return ConfidenceMetricSelection(selected=selected, scores=ordered_scores)
