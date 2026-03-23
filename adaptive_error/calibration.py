from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .cv import CV_RANDOM_STATE, build_stratified_cv_splits, resolve_groups

CalibratorMethod = Literal["none", "temperature", "sigmoid", "isotonic"]
DRIFT_GUARD_THRESHOLD = 0.02
CALIBRATOR_SELECTION_SPLITS = 3


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_class_labels(class_labels: list[Any] | None, n_classes: int) -> list[Any]:
    if class_labels is None:
        return list(range(n_classes))
    if len(class_labels) != n_classes:
        raise ValueError("class_labels must match the number of probability columns.")
    return [_to_python_scalar(label) for label in class_labels]


def _normalize_label_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=object).ravel()
    return np.asarray([_to_python_scalar(value) for value in arr], dtype=object)


def _labels_from_proba(proba: np.ndarray, class_labels: list[Any]) -> np.ndarray:
    label_array = np.asarray([_to_python_scalar(label) for label in class_labels], dtype=object)
    return label_array[np.argmax(np.asarray(proba, dtype=float), axis=1)]


def _resolve_reference_predictions(
    raw_predicted_labels: Any | None,
    proba: np.ndarray,
    class_labels: list[Any],
) -> np.ndarray:
    p = normalize_proba(proba)
    if raw_predicted_labels is None:
        return _labels_from_proba(p, class_labels)

    reference = _normalize_label_array(raw_predicted_labels)
    if reference.shape[0] != p.shape[0]:
        raise ValueError("raw_predicted_labels must contain one label per validation sample.")
    return reference


def _prediction_drift_rate(
    calibrated_proba: np.ndarray,
    raw_predicted_labels: Any,
    class_labels: list[Any],
) -> float:
    predicted_from_calibrated = _labels_from_proba(calibrated_proba, class_labels)
    reference = _normalize_label_array(raw_predicted_labels)
    if predicted_from_calibrated.shape[0] != reference.shape[0]:
        raise ValueError("calibrated_proba and raw_predicted_labels must have the same length.")
    return float(np.mean(predicted_from_calibrated != reference))


def ensure_proba_2d(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        p1 = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
    if p.ndim != 2:
        raise ValueError(f"Expected probabilities to be 1D or 2D, got shape {p.shape}.")
    if p.shape[1] == 1:
        p1 = np.clip(p[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])
    return p


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    p = ensure_proba_2d(proba)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    n_classes = p.shape[1]
    safe = np.isfinite(row_sums) & (row_sums > 0)
    out = np.empty_like(p)
    out[safe[:, 0]] = p[safe[:, 0]] / row_sums[safe[:, 0]]
    out[~safe[:, 0]] = 1.0 / n_classes
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _logit(p: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-12, 1.0 - 1e-12)
    return np.log(arr / (1.0 - arr))


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = normalize_proba(proba)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        return p
    logp = np.log(np.clip(p, 1e-12, 1.0))
    scaled = np.exp(logp / temp)
    return normalize_proba(scaled)


@dataclass
class ProbaCalibrator:
    method: CalibratorMethod
    n_classes: int
    class_labels: list[Any]
    temperature: float | None = None
    sigmoid_coef: tuple[float, float] | None = None
    iso_model: Any | None = None
    models: list[Any] | None = None

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = normalize_proba(proba)
        if self.method == "none":
            return p

        if self.method == "temperature":
            if self.temperature is None:
                raise ValueError("Temperature calibrator is missing its fitted temperature.")
            return _apply_temperature(p, self.temperature)

        if self.method == "sigmoid":
            if self.sigmoid_coef is None:
                raise ValueError("Sigmoid calibrator is missing fitted coefficients.")
            if p.shape[1] != 2:
                raise ValueError("Sigmoid calibration is only supported for binary probabilities.")
            p1 = np.clip(p[:, 1], 1e-12, 1.0 - 1e-12)
            a, b = self.sigmoid_coef
            q1 = _sigmoid(a * _logit(p1) + b)
            return np.column_stack([1.0 - q1, q1])

        if self.method == "isotonic":
            if p.shape[1] == 2:
                if self.iso_model is None:
                    raise ValueError("Isotonic calibrator is missing its fitted model.")
                q1 = np.asarray(self.iso_model.predict(np.clip(p[:, 1], 0.0, 1.0)), dtype=float)
                q1 = np.clip(q1, 0.0, 1.0)
                return np.column_stack([1.0 - q1, q1])

            if self.models is None:
                raise ValueError("Multiclass isotonic calibrator is missing fitted models.")

            q = np.empty_like(p, dtype=float)
            for idx, model in enumerate(self.models):
                q[:, idx] = np.asarray(model.predict(p[:, idx]), dtype=float)
            return normalize_proba(q)

        raise ValueError(f"Unsupported calibrator method: {self.method}")

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "n_classes": int(self.n_classes),
            "class_labels": [_to_python_scalar(label) for label in self.class_labels],
        }
        if self.method == "temperature":
            payload["temperature"] = None if self.temperature is None else float(self.temperature)
        elif self.method == "sigmoid" and self.sigmoid_coef is not None:
            payload["sigmoid_a"] = float(self.sigmoid_coef[0])
            payload["sigmoid_b"] = float(self.sigmoid_coef[1])
        elif self.method == "isotonic":
            if self.iso_model is not None:
                payload["x_min"] = float(getattr(self.iso_model, "X_min_", np.nan))
                payload["x_max"] = float(getattr(self.iso_model, "X_max_", np.nan))
            if self.models is not None:
                payload["per_class"] = [
                    {
                        "class_label": _to_python_scalar(label),
                        "x_min": float(getattr(model, "X_min_", np.nan)),
                        "x_max": float(getattr(model, "X_max_", np.nan)),
                    }
                    for label, model in zip(self.class_labels, self.models)
                ]
        return payload


def fit_temperature_calibrator(
    proba_valid: np.ndarray,
    y_true: np.ndarray,
    *,
    class_labels: list[Any] | None = None,
    temps: Iterable[float] | None = None,
) -> ProbaCalibrator:
    p = normalize_proba(proba_valid)
    labels = _resolve_class_labels(class_labels, p.shape[1])
    y = np.asarray(y_true).ravel()

    if temps is None:
        temp_grid = np.logspace(-2, 2, 25)
    else:
        temp_grid = np.asarray(list(temps), dtype=float)

    best_temperature = 1.0
    best_log_loss = float("inf")
    for temperature in temp_grid:
        if not np.isfinite(temperature) or temperature <= 0.0:
            continue
        p_temp = _apply_temperature(p, float(temperature))
        candidate_loss = float(log_loss(y, p_temp, labels=labels))
        if candidate_loss < best_log_loss:
            best_log_loss = candidate_loss
            best_temperature = float(temperature)

    if not np.isfinite(best_log_loss):
        raise RuntimeError("Temperature scaling failed to produce a finite validation log-loss.")

    return ProbaCalibrator(
        method="temperature",
        n_classes=int(p.shape[1]),
        class_labels=labels,
        temperature=best_temperature,
    )


def fit_sigmoid_calibrator(
    proba_valid: np.ndarray,
    y_true: np.ndarray,
    *,
    class_labels: list[Any] | None = None,
) -> ProbaCalibrator:
    p = normalize_proba(proba_valid)
    if p.shape[1] != 2:
        raise ValueError("Sigmoid calibration is only supported for binary classification.")

    labels = _resolve_class_labels(class_labels, 2)
    positive_label = labels[1]
    y01 = (np.asarray(y_true).ravel() == positive_label).astype(int)
    p1 = np.clip(p[:, 1], 1e-12, 1.0 - 1e-12)
    logit = _logit(p1).reshape(-1, 1)

    if np.unique(y01).size < 2:
        prevalence = float(np.clip(np.mean(y01), 1e-6, 1.0 - 1e-6))
        return ProbaCalibrator(
            method="sigmoid",
            n_classes=2,
            class_labels=labels,
            sigmoid_coef=(0.0, float(_logit(prevalence))),
        )

    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    lr.fit(logit, y01)
    return ProbaCalibrator(
        method="sigmoid",
        n_classes=2,
        class_labels=labels,
        sigmoid_coef=(float(lr.coef_[0, 0]), float(lr.intercept_[0])),
    )


def fit_isotonic_calibrator(
    proba_valid: np.ndarray,
    y_true: np.ndarray,
    *,
    class_labels: list[Any] | None = None,
) -> ProbaCalibrator:
    from sklearn.isotonic import IsotonicRegression

    p = normalize_proba(proba_valid)
    labels = _resolve_class_labels(class_labels, p.shape[1])
    y = np.asarray(y_true).ravel()

    if p.shape[1] == 2:
        positive_label = labels[1]
        y01 = (y == positive_label).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        if np.unique(y01).size < 2:
            mean = float(np.mean(y01))
            iso.fit([0.0, 1.0], [mean, mean])
        else:
            iso.fit(np.clip(p[:, 1], 0.0, 1.0), y01)
        return ProbaCalibrator(
            method="isotonic",
            n_classes=2,
            class_labels=labels,
            iso_model=iso,
        )

    models: list[Any] = []
    for idx, label in enumerate(labels):
        yk = (y == label).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        if np.unique(yk).size < 2:
            mean = float(np.mean(yk))
            iso.fit([0.0, 1.0], [mean, mean])
        else:
            iso.fit(np.clip(p[:, idx], 0.0, 1.0), yk)
        models.append(iso)

    return ProbaCalibrator(
        method="isotonic",
        n_classes=int(p.shape[1]),
        class_labels=labels,
        models=models,
    )


def fit_calibrator(
    method: CalibratorMethod,
    proba_valid: np.ndarray,
    y_true: np.ndarray,
    *,
    class_labels: list[Any] | None = None,
) -> ProbaCalibrator:
    p = normalize_proba(proba_valid)
    labels = _resolve_class_labels(class_labels, p.shape[1])

    if method == "none":
        return ProbaCalibrator(method="none", n_classes=int(p.shape[1]), class_labels=labels)
    if method == "temperature":
        return fit_temperature_calibrator(p, y_true, class_labels=labels)
    if method == "sigmoid":
        return fit_sigmoid_calibrator(p, y_true, class_labels=labels)
    if method == "isotonic":
        return fit_isotonic_calibrator(p, y_true, class_labels=labels)
    raise ValueError(f"Unknown calibrator method: {method}")


def select_best_calibrator(
    proba_valid: np.ndarray,
    y_true: np.ndarray,
    *,
    class_labels: list[Any] | None = None,
    raw_predicted_labels: Any | None = None,
    methods: tuple[CalibratorMethod, ...] | None = None,
    groups: Any | None = None,
    n_splits: int = CALIBRATOR_SELECTION_SPLITS,
    random_state: int = CV_RANDOM_STATE,
    drift_guard_threshold: float = DRIFT_GUARD_THRESHOLD,
) -> tuple[ProbaCalibrator, dict[str, Any]]:
    p = normalize_proba(proba_valid)
    labels = _resolve_class_labels(class_labels, p.shape[1])
    y = np.asarray(y_true).ravel()
    group_array = resolve_groups(groups, p.shape[0]) if groups is not None else None
    reference_predictions = _resolve_reference_predictions(raw_predicted_labels, p, labels)
    drift_guard_threshold = float(drift_guard_threshold)

    if methods is None:
        requested = (
            ["none", "temperature", "sigmoid", "isotonic"]
            if p.shape[1] == 2
            else ["none", "temperature", "isotonic"]
        )
    else:
        requested = list(methods)
        if "none" not in requested:
            requested.insert(0, "none")

    deduped_requested: list[CalibratorMethod] = []
    for method in requested:
        if method not in deduped_requested:
            deduped_requested.append(method)

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    eligible_candidates: list[dict[str, Any]] = []
    splits, cv_info, cv_warnings = build_stratified_cv_splits(
        y,
        groups=group_array,
        n_splits=n_splits,
        random_state=random_state,
    )
    warnings.extend(cv_warnings)

    for method in deduped_requested:
        if method == "sigmoid" and p.shape[1] != 2:
            skipped.append({"method": method, "reason": "binary_only"})
            continue

        try:
            fold_losses: list[float] = []
            if splits:
                for train_idx, valid_idx in splits:
                    calibrator_cv = fit_calibrator(
                        method,
                        p[train_idx],
                        y[train_idx],
                        class_labels=labels,
                    )
                    calibrated_cv = calibrator_cv.transform(p[valid_idx])
                    fold_losses.append(
                        float(log_loss(y[valid_idx], calibrated_cv, labels=labels))
                    )

            calibrator = fit_calibrator(method, p, y, class_labels=labels)
            calibrated = calibrator.transform(p)
            candidate_loss = (
                float(np.mean(fold_losses))
                if fold_losses
                else float(log_loss(y, calibrated, labels=labels))
            )
            full_log_loss = float(log_loss(y, calibrated, labels=labels))
            drift_rate = _prediction_drift_rate(calibrated, reference_predictions, labels)
            eligible_under_drift_guard = bool(
                np.isfinite(drift_rate) and drift_rate <= drift_guard_threshold
            )
            candidate_entry = {
                "method": method,
                "log_loss": candidate_loss,
                "log_loss_cv": candidate_loss if fold_losses else None,
                "log_loss_full": full_log_loss,
                "fold_log_loss": fold_losses if fold_losses else None,
                "drift_rate": drift_rate,
                "eligible_under_drift_guard": eligible_under_drift_guard,
            }
            candidates.append(candidate_entry)

            candidate_payload = dict(candidate_entry)
            candidate_payload["calibrator"] = calibrator
            if np.isfinite(candidate_loss):
                valid_candidates.append(candidate_payload)
                if eligible_under_drift_guard:
                    eligible_candidates.append(candidate_payload)
        except Exception as exc:
            candidates.append(
                {
                    "method": method,
                    "log_loss": None,
                    "drift_rate": None,
                    "eligible_under_drift_guard": False,
                    "error": str(exc),
                }
            )

    drift_guard_applied = False
    drift_guard_fallback_used = False
    if eligible_candidates:
        chosen = min(eligible_candidates, key=lambda entry: entry["log_loss"])
        drift_guard_applied = True
    elif valid_candidates:
        chosen = min(valid_candidates, key=lambda entry: entry["log_loss"])
        drift_guard_fallback_used = True
        warnings.append(
            {
                "type": "drift_guard_fallback",
                "message": (
                    "No calibrator satisfied the drift guard; selected the best "
                    "validation log-loss candidate instead."
                ),
            }
        )
    else:
        chosen = None

    if chosen is None:
        best_calibrator = ProbaCalibrator(method="none", n_classes=int(p.shape[1]), class_labels=labels)
        selected_log_loss = None
        selected_drift_rate = None
        selected_eligible = False
    else:
        best_calibrator = chosen["calibrator"]
        selected_log_loss = float(chosen["log_loss"])
        selected_drift_rate = float(chosen["drift_rate"])
        selected_eligible = bool(chosen["eligible_under_drift_guard"])

    diagnostics = {
        "candidates": candidates,
        "skipped": skipped,
        "warnings": warnings,
        "cv": cv_info,
        "selected": {
            "method": best_calibrator.method,
            "log_loss": selected_log_loss,
            "drift_rate": selected_drift_rate,
            "eligible_under_drift_guard": selected_eligible,
            "drift_guard_threshold": drift_guard_threshold,
            "drift_guard_applied": drift_guard_applied,
            "drift_guard_fallback_used": drift_guard_fallback_used,
        },
    }
    return best_calibrator, diagnostics
