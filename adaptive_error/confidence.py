from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .core import confidence_from_proba

ConfidenceMetricName = Literal[
    "calibrated_probability_confidence",
]
CONFIDENCE_METRIC_NAME: ConfidenceMetricName = "calibrated_probability_confidence"
CONFIDENCE_SOURCE_NAME = "calibrated_class_probabilities"


@dataclass(frozen=True)
class ConfidenceMetricSelection:
    selected: ConfidenceMetricName = CONFIDENCE_METRIC_NAME

    def to_json(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "selection_mode": "fixed",
            "metric_family": "universal_probability_metric",
            "display_name": "Confidence from calibrated class probabilities",
            "description": (
                "Binary tasks use distance from maximum uncertainty; multiclass tasks use "
                "the margin between the highest and second-highest calibrated class probabilities."
            ),
            "definition": {
                "binary": "2 * abs(p_positive - 0.5)",
                "multiclass": "p_max - p_2nd",
            },
        }


def get_confidence_metric_selection() -> ConfidenceMetricSelection:
    return ConfidenceMetricSelection()


def compute_confidence(
    proba: Any,
) -> np.ndarray:
    return np.asarray(confidence_from_proba(proba), dtype=float)
