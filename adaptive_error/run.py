from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .calibration import (
    DRIFT_GUARD_THRESHOLD,
    ProbaCalibrator,
    select_best_calibrator,
)
from .collect import (
    collect_gandalf_outputs,
    collect_tabpfn_outputs,
    collect_text_outputs,
    collect_vision_outputs,
)
from .confidence import (
    CONFIDENCE_METRIC_NAME,
    CONFIDENCE_SOURCE_NAME,
    compute_confidence,
    get_confidence_metric_selection,
)
from .core import (
    AdaptiveErrorCalculator,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_AER_KWARGS = {
    "n_bins": 20,
    "min_bin_count": 10,
    "smooth": True,
    "enforce_monotonic": False,
    "prior_strength": 2.0,
    "adaptive_binning": False,
}

class AdaptiveErrorNotApplicableError(ValueError):
    """Raised when adaptive error rate post-processing does not apply to the run."""


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return _to_python_scalar(value)


def _probability_column_name(label: Any, *, prefix: str = "prob_") -> str:
    return f"{prefix}{label}"


def _parse_probability_label(label: str) -> Any:
    try:
        if "." in label:
            value = float(label)
            return int(value) if value.is_integer() else value
        return int(label)
    except ValueError:
        return label


def _extract_probability_matrix(
    frame: pd.DataFrame,
    *,
    class_labels: list[Any] | None = None,
    prefix: str = "prob_",
) -> tuple[np.ndarray, list[Any]]:
    if class_labels is None:
        prob_cols = [col for col in frame.columns if str(col).startswith(prefix)]
        if not prob_cols:
            raise ValueError(f"No probability columns found with prefix '{prefix}'.")
        labels = [_parse_probability_label(str(col)[len(prefix) :]) for col in prob_cols]
    else:
        prob_cols = [_probability_column_name(label, prefix=prefix) for label in class_labels]
        missing = [col for col in prob_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")
        labels = class_labels

    return frame[prob_cols].to_numpy(dtype=float), list(labels)


def _labels_from_proba(proba: np.ndarray, class_labels: list[Any]) -> np.ndarray:
    label_array = np.asarray(class_labels, dtype=object)
    return label_array[np.argmax(proba, axis=1)]


def _multiclass_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_labels: list[Any],
) -> float:
    y_arr = np.asarray(y_true).ravel()
    p = np.asarray(proba, dtype=float)
    one_hot = np.zeros_like(p, dtype=float)
    for idx, label in enumerate(class_labels):
        one_hot[:, idx] = (y_arr == label).astype(float)
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def _brier_error_score(predicted_error: np.ndarray, incorrect: np.ndarray) -> float:
    pred = np.clip(np.asarray(predicted_error, dtype=float).ravel(), 0.0, 1.0)
    obs = np.asarray(incorrect, dtype=float).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError("predicted_error and incorrect must have the same length.")
    return float(np.mean((pred - obs) ** 2))


def _ece_error(predicted_error: np.ndarray, incorrect: np.ndarray, *, n_bins: int = 20) -> float:
    pred = np.clip(np.asarray(predicted_error, dtype=float).ravel(), 0.0, 1.0)
    obs = np.asarray(incorrect, dtype=float).ravel()
    if pred.shape[0] != obs.shape[0]:
        raise ValueError("predicted_error and incorrect must have the same length.")
    if pred.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(pred.size)
    ece = 0.0

    for idx in range(edges.size - 1):
        left = edges[idx]
        right = edges[idx + 1]
        if idx < edges.size - 2:
            mask = (pred >= left) & (pred < right)
        else:
            mask = (pred >= left) & (pred <= right)

        if not np.any(mask):
            continue

        empirical = float(np.mean(obs[mask]))
        expected = float(np.mean(pred[mask]))
        ece += (float(np.sum(mask)) / total) * abs(empirical - expected)

    return float(ece)


def _format_label_list(labels: list[Any], *, max_items: int = 5) -> str:
    preview = [repr(_to_python_scalar(label)) for label in labels[:max_items]]
    if len(labels) > max_items:
        preview.append("...")
    return ", ".join(preview)


def _ensure_multiple_class_labels(
    frame: pd.DataFrame,
    *,
    split_name: str,
    class_labels: list[Any] | None = None,
) -> None:
    if "true_label" not in frame.columns:
        raise ValueError("Adaptive error rate requires a 'true_label' column in collected outputs.")

    observed = pd.unique(frame["true_label"])
    observed_labels = [_to_python_scalar(label) for label in observed.tolist()]
    if len(observed_labels) < 2:
        label_text = _format_label_list(observed_labels) if observed_labels else "<none>"
        raise AdaptiveErrorNotApplicableError(
            "Adaptive error rate is not applicable because the "
            f"{split_name} split contains only one observed class label ({label_text}). "
            "This usually means all class labels are the same, which commonly happens for "
            "autoencoder-style projects. Classification models require at least two distinct "
            "class labels, so adaptive error rate cannot be computed for this run."
        )

    if class_labels is not None and len(class_labels) < 2:
        label_text = _format_label_list(class_labels)
        raise AdaptiveErrorNotApplicableError(
            "Adaptive error rate is not applicable because the collected "
            f"{split_name} probabilities only cover one class ({label_text}). "
            "This usually indicates a single-class classification setup, such as an "
            "autoencoder-style project, where adaptive error rate cannot be computed."
        )


def _prepare_output_frame(
    frame: pd.DataFrame,
    calibrator: ProbaCalibrator,
    class_labels: list[Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    out = frame.copy()
    raw_proba, _ = _extract_probability_matrix(out, class_labels=class_labels)
    calibrated = calibrator.transform(raw_proba)
    raw_predicted = out["predicted_label"].to_numpy()
    predicted_from_cal = _labels_from_proba(calibrated, class_labels)
    true_label = out["true_label"].to_numpy()
    raw_correct = (raw_predicted == true_label).astype(int)
    correct_from_cal = (predicted_from_cal == true_label).astype(int)

    for idx, label in enumerate(class_labels):
        out[_probability_column_name(label, prefix="cal_prob_")] = calibrated[:, idx]

    out["predicted_label"] = raw_predicted
    out["correct"] = raw_correct
    out["predicted_label_from_cal_proba"] = predicted_from_cal
    out["correct_from_cal_proba"] = correct_from_cal
    out["prediction_changed_by_calibration"] = (predicted_from_cal != raw_predicted).astype(int)
    return out, calibrated


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=4), encoding="utf-8")


def _write_confidence_plot(bins_df: pd.DataFrame, out_path: Path) -> None:
    if plt is None or bins_df.empty:
        return

    plot_df = bins_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["bin_center", "expected_error_rate"]
    )
    plot_df = plot_df[plot_df["count"].fillna(0).astype(int) > 0]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = plot_df["bin_center"].to_numpy(dtype=float)
    y_expected = plot_df["expected_error_rate"].to_numpy(dtype=float)
    ax.plot(
        x,
        y_expected,
        marker="o",
        linewidth=2.0,
        label="Adaptive expected error rate",
    )

    ax.set_xlabel("Predictive confidence")
    ax.set_ylabel("Expected error rate")
    ax.set_title("Predictive Confidence vs Expected Error Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _collect_outputs(
    args,
    modality: str,
    model_version: str,
    ckpt_directory: Path | str,
    split_path: Path | str,
    target_column: str = "labels",
) -> pd.DataFrame:
    if modality == "text":
        return collect_text_outputs(args, model_version, ckpt_directory, split_path)
    if modality == "images":
        return collect_vision_outputs(args, model_version, ckpt_directory, split_path)
    if modality != "tabular":
        raise ValueError(f"Adaptive error is not supported for modality '{modality}'.")

    if model_version == "gandalf":
        return collect_gandalf_outputs(args, ckpt_directory, split_path,label_column=target_column)
    if model_version == "tabpfn":
        return collect_tabpfn_outputs(args, ckpt_directory, split_path,label_column=target_column)
    raise ValueError(f"Unsupported tabular model for adaptive error: {model_version}")


def _resolve_validation_groups(
    args,
    frame: pd.DataFrame,
    split_path: Path | str,
) -> np.ndarray | None:
    grouper = getattr(args, "grouper", None)
    if not grouper:
        return None

    if grouper in frame.columns:
        return frame[grouper].to_numpy()

    try:
        group_frame = pd.read_parquet(Path(split_path), columns=[grouper])
    except Exception:
        return None

    if grouper not in group_frame.columns or len(group_frame) != len(frame):
        return None
    return group_frame[grouper].to_numpy()


def _run_adaptive_error_impl(
    args,
    modality,
    model_version,
    ckpt_directory,
    val_data_path,
    test_data_path,
    out_dir,
    target_column: str = "labels",
):
    aer_dir = Path(out_dir) / "adaptive_error_rate"

    val_frame = _collect_outputs(args, modality, model_version, ckpt_directory, val_data_path,target_column=target_column)
    _ensure_multiple_class_labels(val_frame, split_name="validation")
    raw_val_proba, class_labels = _extract_probability_matrix(val_frame)
    _ensure_multiple_class_labels(
        val_frame,
        split_name="validation",
        class_labels=class_labels,
    )
    y_val = val_frame["true_label"].to_numpy()
    raw_val_predicted = val_frame["predicted_label"].to_numpy()
    validation_groups = _resolve_validation_groups(args, val_frame, val_data_path)

    calibrator, calibration_diag = select_best_calibrator(
        raw_val_proba,
        y_val,
        class_labels=class_labels,
        raw_predicted_labels=raw_val_predicted,
        groups=validation_groups,
    )

    val_out, val_calibrated = _prepare_output_frame(val_frame, calibrator, class_labels)
    val_incorrect = (val_out["predicted_label"].to_numpy() != y_val).astype(int)
    confidence_selection = get_confidence_metric_selection()
    confidence_selection_json = confidence_selection.to_json()
    confidence_metric = CONFIDENCE_METRIC_NAME
    confidence_source = CONFIDENCE_SOURCE_NAME
    val_confidence = compute_confidence(val_calibrated)
    val_out["confidence"] = val_confidence

    aer = AdaptiveErrorCalculator(**DEFAULT_AER_KWARGS)
    aer.fit(val_confidence, val_incorrect)
    val_out["adaptive_error_rate"] = aer.get_expected_error(val_confidence)
    bins_df = aer.bin_stats_df(drop_empty=False)

    test_frame = _collect_outputs(args, modality, model_version, ckpt_directory, test_data_path,target_column=target_column)
    test_out, test_calibrated = _prepare_output_frame(test_frame, calibrator, class_labels)
    test_confidence = compute_confidence(test_calibrated)
    test_out["confidence"] = test_confidence
    test_out["adaptive_error_rate"] = aer.get_expected_error(test_confidence)
    test_incorrect_raw = 1 - test_out["correct"].to_numpy(dtype=int)

    val_out.to_csv(aer_dir / "val_per_sample.csv", index=False)
    test_out.to_csv(aer_dir / "test_per_sample.csv", index=False)
    bins_df.to_csv(aer_dir / "predictive_confidence_vs_expected_error_rate.csv", index=False)
    _write_confidence_plot(
        bins_df,
        aer_dir / "predictive_confidence_vs_expected_error_rate.png",
    )

    aer_lookup = aer.to_json_dict()
    aer_lookup["class_labels"] = [_to_python_scalar(label) for label in class_labels]
    aer_lookup["calibrator"] = calibrator.to_json_dict()
    aer_lookup["confidence_metric"] = confidence_metric
    aer_lookup["confidence_source"] = confidence_source
    aer_lookup["confidence_metric_selection"] = confidence_selection_json
    _save_json(aer_dir / "aer_lookup.json", aer_lookup)

    validation_log_loss = float(log_loss(y_val, val_calibrated, labels=class_labels))
    validation_brier = _multiclass_brier_score(y_val, val_calibrated, class_labels)
    selected_calibration = calibration_diag.get("selected", {})
    validation_prediction_drift_rate = selected_calibration.get("drift_rate")
    if validation_prediction_drift_rate is None:
        validation_prediction_drift_rate = float(
            np.mean(val_out["prediction_changed_by_calibration"].to_numpy(dtype=float))
        )
    else:
        validation_prediction_drift_rate = float(validation_prediction_drift_rate)

    drift_guard_threshold = float(
        selected_calibration.get("drift_guard_threshold", DRIFT_GUARD_THRESHOLD)
    )
    drift_guard_applied = bool(selected_calibration.get("drift_guard_applied", False))
    test_accuracy_raw = float(np.mean(test_out["correct"].to_numpy(dtype=float)))
    test_accuracy_from_cal_proba = float(
        np.mean(test_out["correct_from_cal_proba"].to_numpy(dtype=float))
    )
    test_prediction_drift_rate = float(
        np.mean(test_out["prediction_changed_by_calibration"].to_numpy(dtype=float))
    )
    mean_test_adaptive_error_rate = float(
        np.mean(test_out["adaptive_error_rate"].to_numpy(dtype=float))
    )
    brier_error_rate_test = _brier_error_score(
        test_out["adaptive_error_rate"].to_numpy(dtype=float),
        test_incorrect_raw,
    )
    ece_error_rate_test = _ece_error(
        test_out["adaptive_error_rate"].to_numpy(dtype=float),
        test_incorrect_raw,
    )

    run_config = {
        "calibration_split": "validation",
        "decision_label_source": "raw_model_predictions",
        "confidence_source": confidence_source,
        "confidence_metric": confidence_metric,
        "confidence_metric_family": confidence_selection_json.get("metric_family"),
        "confidence_metric_display_name": confidence_selection_json.get("display_name"),
        "confidence_metric_description": confidence_selection_json.get("description"),
        "confidence_metric_definition": confidence_selection_json.get("definition"),
        "confidence_metric_selection": confidence_selection_json,
        "adaptive_error_rate_source": "validation_fitted_confidence_to_error_rate_mapping",
        "drift_guard_threshold": drift_guard_threshold,
        "drift_guard_applied": drift_guard_applied,
        "prediction_drift_rate": validation_prediction_drift_rate,
        "chosen_calibrator": calibrator.method,
        "calibration_diagnostics": calibration_diag,
        "aer_kwargs": dict(DEFAULT_AER_KWARGS),
        "metric_scale": "0_to_1_fraction",
        "class_labels": [_to_python_scalar(label) for label in class_labels],
    }
    _save_json(aer_dir / "run_config.json", run_config)

    summary = {
        "status": "success",
        "chosen_calibrator": calibrator.method,
        "validation_log_loss": validation_log_loss,
        "validation_brier_score": validation_brier,
        "test_accuracy_raw": test_accuracy_raw,
        "test_accuracy_from_cal_proba": test_accuracy_from_cal_proba,
        "prediction_drift_rate": test_prediction_drift_rate,
        "validation_prediction_drift_rate": validation_prediction_drift_rate,
        "mean_test_adaptive_error_rate": mean_test_adaptive_error_rate,
        "brier_error_rate_test": brier_error_rate_test,
        "ece_error_rate_test": ece_error_rate_test,
        "number_of_calibration_bins": int(len(aer.expected_error) if aer.expected_error is not None else 0),
        "n_validation_samples": int(len(val_out)),
        "n_test_samples": int(len(test_out)),
        "calibration_split": "validation",
        "decision_label_source": "raw_model_predictions",
        "confidence_source": confidence_source,
        "confidence_metric": confidence_metric,
        "confidence_metric_family": confidence_selection_json.get("metric_family"),
        "confidence_metric_display_name": confidence_selection_json.get("display_name"),
        "confidence_metric_description": confidence_selection_json.get("description"),
        "confidence_metric_definition": confidence_selection_json.get("definition"),
        "confidence_metric_selection": confidence_selection_json,
        "adaptive_error_rate_source": "validation_fitted_confidence_to_error_rate_mapping",
        "drift_guard_threshold": drift_guard_threshold,
        "drift_guard_applied": drift_guard_applied,
        "metric_scale": "0_to_1_fraction",
        "calibration_diagnostics": calibration_diag,
    }
    _save_json(aer_dir / "summary.json", summary)


def run_adaptive_error(
    args,
    modality,
    model_version,
    ckpt_directory,
    val_data_path,
    test_data_path,
    out_dir,
    target_column: str = "labels",
):
    if modality == "timeseries":
        return

    mode = getattr(args, "mode", None)
    if mode not in (None, "cls"):
        return

    if modality == "tabular" and model_version == "gandalf":
        if str(getattr(args, "type", "")).lower() != "classification":
            return

    aer_dir = Path(out_dir) / "adaptive_error_rate"
    aer_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run_adaptive_error_impl(
            args=args,
            modality=modality,
            model_version=model_version,
            ckpt_directory=ckpt_directory,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
            out_dir=out_dir,
            target_column=target_column,
        )
        
        print("Adaptive error rate post-processing completed successfully.")
    except Exception as exc:
        import traceback
        tb_str = traceback.format_exc()
        print(f"Warning: adaptive error rate post-processing failed: {exc}")
        print(f"Error traceback:\n{tb_str}")
        failure_summary = {
            "status": "failed",
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "modality": modality,
            "model_version": model_version,
            "traceback":tb_str
        }
        _save_json(aer_dir / "error.json", failure_summary)
        _save_json(aer_dir / "summary.json", failure_summary)
