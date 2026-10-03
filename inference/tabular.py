"""Inference for DeepTune's tabular modality (GANDALF and TabPFN)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def predict_gandalf(
    df: pd.DataFrame,
    checkpoint_dir: Path,
    target_column: str,
) -> pd.DataFrame:
    """Runs the trained GANDALF model saved at `checkpoint_dir` (a
    'GANDALF_model' directory, per pytorch_tabular.TabularModel.save_model
    -- desktop/config.py's RunResults.checkpoint_path glob already resolves
    all the way to that directory itself, unlike evaluate_gandalf.py's own
    --model_weights CLI arg, which takes that directory's *parent* and
    appends 'GANDALF_model' itself; passing checkpoint_dir straight through
    here matches what desktop/config.py actually hands this function)
    over every row of `df`. `target_column` may or may not be present in
    `df` -- pytorch_tabular's predict() only needs the feature columns it
    was fit on; the same call already runs, with the target column present,
    in embed/tabular/gandalf_embeddings.py.

    pytorch_tabular already returns per-row predicted class (as its
    original label, not a bare index -- it inverse-transforms internally)
    and per-class probabilities, unlike this project's other evaluate_*.py
    modules, so there's no separate label_mapping to apply here.
    """
    from pytorch_tabular import TabularModel

    model = TabularModel.load_model(str(checkpoint_dir))
    pred_df = model.predict(df, progress_bar=None)

    prediction_col = f'{target_column}_prediction'
    if prediction_col not in pred_df.columns:
        raise RuntimeError(f"GANDALF's prediction output is missing the expected '{prediction_col}' column.")
    probability_cols = [c for c in pred_df.columns if c.startswith(f'{target_column}_') and c.endswith('_probability')]

    sample_ids = df['__sample_id'].tolist() if '__sample_id' in df.columns else list(range(len(df)))
    result = pd.DataFrame({'sample_id': sample_ids, 'predicted_label': pred_df[prediction_col].tolist()})
    if probability_cols:
        # Confidence: the probability pytorch_tabular assigned to the class
        # it actually predicted for that row. Class names are recovered
        # from the probability columns as strings (sliced out of the
        # column name), while labels_prediction can come back as a native
        # dtype (e.g. int64 for integer-coded classes) -- compare both as
        # strings so e.g. predicted class 1 matches probability column
        # 'labels_1_probability' correctly.
        probs = pred_df[probability_cols]
        probs.columns = [c[len(target_column) + 1:-len('_probability')] for c in probability_cols]
        result['confidence'] = [row[str(label)] if str(label) in row.index else None
                                for label, (_, row) in zip(pred_df[prediction_col], probs.iterrows())]
    else:
        # Regression: TabularModel.predict() has no *_probability columns.
        result = result.rename(columns={'predicted_label': 'predicted_value'})

    if target_column in df.columns:
        true_values = df[target_column].tolist()
        result['true_label' if probability_cols else 'true_value'] = true_values
        if probability_cols:
            result['correct'] = [str(t) == str(p) for t, p in zip(true_values, pred_df[prediction_col])]
        else:
            result['abs_error'] = [abs(float(t) - float(p)) for t, p in zip(true_values, pred_df[prediction_col])]
    return result


def predict_tabpfn(
    df: pd.DataFrame,
    checkpoint_path: Path,
    target_column: str,
    mode: str,
    finetuning_mode: bool,
) -> pd.DataFrame:
    """Runs the trained TabPFN model saved at `checkpoint_path` (a
    .tabpfn_fit file for from-scratch training, or a .joblib file for
    finetuning-mode, matching evaluate_tabpfn.py's loading) over every row
    of `df`. `target_column`, if present in `df`, is excluded from the
    feature matrix and used as ground truth for comparison; otherwise
    prediction runs with no ground truth to compare against.
    """
    from options import DEVICE

    has_labels = target_column in df.columns
    feature_cols = [c for c in df.columns if c not in (target_column, '__sample_id')]
    X = df[feature_cols]

    if finetuning_mode:
        from joblib import load
        model = load(Path(checkpoint_path))
    else:
        from tabpfn.model_loading import load_fitted_tabpfn_model
        model = load_fitted_tabpfn_model(checkpoint_path, device=DEVICE)

    predicted = model.predict(X)
    sample_ids = df['__sample_id'].tolist() if '__sample_id' in df.columns else list(range(len(df)))

    if mode == 'cls':
        result = pd.DataFrame({'sample_id': sample_ids, 'predicted_label': predicted})
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            result['confidence'] = probabilities.max(axis=1)
        if has_labels:
            true_values = df[target_column].tolist()
            result['true_label'] = true_values
            result['correct'] = [str(t) == str(p) for t, p in zip(true_values, predicted)]
    else:
        result = pd.DataFrame({'sample_id': sample_ids, 'predicted_value': predicted})
        if has_labels:
            true_values = df[target_column].tolist()
            result['true_value'] = true_values
            result['abs_error'] = [abs(float(t) - float(p)) for t, p in zip(true_values, predicted)]
    return result
