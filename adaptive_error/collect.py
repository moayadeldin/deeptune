from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAFE_METADATA_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    "text",
)


def _parse_probability_label(label: str) -> Any:
    try:
        if "." in label:
            value = float(label)
            return int(value) if value.is_integer() else value
        return int(label)
    except ValueError:
        return label


def _probability_column_name(label: Any) -> str:
    return f"prob_{label}"


def _resolve_checkpoint_path(path: Path | str, suffix: str) -> Path:
    model_path = Path(path)
    if model_path.suffix == suffix:
        return model_path
    matches = sorted(model_path.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Could not find a checkpoint ending with '{suffix}' under {model_path}.")
    return matches[0]


def _build_output_frame(
    df: pd.DataFrame,
    *,
    metadata_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if "labels" not in df.columns:
        raise ValueError("Input split must contain a 'labels' column.")

    frame = pd.DataFrame(
        {
            "row_index": df.index.to_numpy(),
            "true_label": df["labels"].to_numpy(),
        }
    )

    selected_metadata = SAFE_METADATA_COLUMNS if metadata_columns is None else metadata_columns
    for column in selected_metadata:
        if column not in df.columns:
            continue
        column_name = str(column)
        if column_name in frame.columns:
            continue
        if column_name.startswith("prob_") or column_name.startswith("cal_prob_"):
            continue
        frame[column_name] = df[column].to_numpy()

    return frame


def _attach_predictions(
    frame: pd.DataFrame,
    *,
    proba: np.ndarray,
    predicted: np.ndarray,
    class_labels: list[Any],
) -> pd.DataFrame:
    out = frame.copy()
    for idx, label in enumerate(class_labels):
        out[_probability_column_name(label)] = proba[:, idx]
    out["predicted_label"] = np.asarray(predicted)
    out["correct"] = (out["predicted_label"].to_numpy() == out["true_label"].to_numpy()).astype(int)
    return out


def collect_text_outputs(
    args,
    model_version: str,
    ckpt_directory: Path | str,
    split_path: Path | str,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    from datasets.text_datasets import TextDataset
    from helpers import load_finetuned_gpt2, load_finetunedbert_model
    from options import DEVICE

    split_path = Path(split_path)
    df = pd.read_parquet(split_path)
    batch_size = getattr(args, "batch_size", None) or 32

    if model_version == "BERT":
        model, tokenizer = load_finetunedbert_model(ckpt_directory)
    elif model_version == "gpt2":
        model, tokenizer = load_finetuned_gpt2(ckpt_directory)
    else:
        raise ValueError(f"Unsupported text model for adaptive error collection: {model_version}")

    dataset = TextDataset(parquet_file=split_path, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device=DEVICE)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            encoding = batch[0]

            if model_version == "BERT":
                input_ids = encoding["input_ids"].to(DEVICE)
                attention_mask = encoding["attention_mask"].to(DEVICE)
                token_type_ids = encoding.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                logits = model(input_ids, attention_mask, token_type_ids)
            else:
                input_ids = encoding["input_ids"].to(DEVICE)
                attention_mask = encoding["attention_mask"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    proba = np.concatenate(all_probs, axis=0)
    predicted = np.concatenate(all_predictions, axis=0)
    class_labels = list(range(proba.shape[1]))

    frame = _build_output_frame(df)
    return _attach_predictions(frame, proba=proba, predicted=predicted, class_labels=class_labels)


def collect_vision_outputs(
    args,
    model_version: str,
    ckpt_directory: Path | str,
    split_path: Path | str,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    from datasets.image_datasets import ParquetImageDataset
    from helpers import transformations
    from options import DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
    from utils import UseCase, get_model_architecture, get_model_cls

    split_path = Path(split_path)
    df = pd.read_parquet(split_path)
    batch_size = getattr(args, "batch_size", None) or 32
    num_classes = getattr(args, "num_classes")
    added_layers = getattr(args, "added_layers", 2)
    embed_size = getattr(args, "embed_size", 1000)
    freeze_backbone = getattr(args, "freeze_backbone", False)
    use_peft = getattr(args, "use_peft", False)
    model_architecture = getattr(args, "model_architecture", None) or get_model_architecture(model_version)

    if model_architecture == "siglip" and model_version == "siglip":
        from src.vision.siglip import load_siglip_processor_offline, load_siglip_variant

        ckpt_path = _resolve_checkpoint_path(ckpt_directory, ".pt")
        processor = load_siglip_processor_offline()
        use_case = UseCase.PEFT if use_peft else UseCase.FINETUNED
        model = load_siglip_variant(
            use_case=use_case,
            num_classes=num_classes,
            added_layers=added_layers,
            embed_size=embed_size,
            freeze_backbone=freeze_backbone,
            model_weights=ckpt_path,
            device=DEVICE,
            tiny=True,
        )
        dataset = ParquetImageDataset.from_parquet(split_path, processor=processor)
    else:
        ckpt_path = _resolve_checkpoint_path(ckpt_directory, ".pth")
        model_cls = get_model_cls(model_architecture=model_architecture, use_peft=use_peft)
        model = model_cls(
            num_classes,
            model_version,
            added_layers,
            embed_size,
            freeze_backbone=freeze_backbone,
            task_type="cls",
        )
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        dataset = ParquetImageDataset.from_parquet(split_path, transform=transformations)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK,
    )

    model.to(DEVICE)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(DEVICE)
            if model_architecture == "siglip" and model_version == "siglip":
                logits = model({"pixel_values": inputs})
            else:
                logits = model(inputs)

            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    proba = np.concatenate(all_probs, axis=0)
    predicted = np.concatenate(all_predictions, axis=0)
    class_labels = list(range(proba.shape[1]))

    frame = _build_output_frame(df)
    return _attach_predictions(frame, proba=proba, predicted=predicted, class_labels=class_labels)


def collect_gandalf_outputs(
    args,
    ckpt_directory: Path | str,
    split_path: Path | str,
) -> pd.DataFrame:
    from pytorch_tabular import TabularModel

    split_path = Path(split_path)
    df = pd.read_parquet(split_path)

    model = TabularModel.load_model(str(Path(ckpt_directory) / "GANDALF_model"))
    try:
        pred_df = model.predict(df)
    except Exception:
        pred_df = model.predict(df.drop(columns=["labels"], errors="ignore"))

    if not isinstance(pred_df, pd.DataFrame):
        pred_df = pd.DataFrame(pred_df)

    prob_cols = [col for col in pred_df.columns if str(col).endswith("_probability")]
    if not prob_cols:
        raise ValueError("GANDALF predictions did not expose probability columns.")

    class_labels = [_parse_probability_label(str(col)[: -len("_probability")]) for col in prob_cols]
    proba = pred_df[prob_cols].to_numpy(dtype=float)

    if "prediction" in pred_df.columns:
        predicted = pred_df["prediction"].to_numpy()
    else:
        pred_candidates = [col for col in pred_df.columns if str(col).endswith("_prediction")]
        if pred_candidates:
            predicted = pred_df[pred_candidates[0]].to_numpy()
        else:
            label_array = np.asarray(class_labels, dtype=object)
            predicted = label_array[np.argmax(proba, axis=1)]

    frame = _build_output_frame(df)
    return _attach_predictions(frame, proba=proba, predicted=predicted, class_labels=class_labels)


def collect_tabpfn_outputs(
    args,
    ckpt_directory: Path | str,
    split_path: Path | str,
) -> pd.DataFrame:
    from joblib import load

    from options import DEVICE
    from tabpfn.model_loading import load_fitted_tabpfn_model

    split_path = Path(split_path)
    df = pd.read_parquet(split_path)
    features = df.drop(columns=["labels"])

    if getattr(args, "finetuning_mode", False):
        model = load(Path(ckpt_directory))
    else:
        model = load_fitted_tabpfn_model(Path(ckpt_directory), device=DEVICE)

    proba = np.asarray(model.predict_proba(features), dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    class_labels = [
        _parse_probability_label(str(label))
        for label in getattr(model, "classes_", list(range(proba.shape[1])))
    ]

    try:
        predicted = np.asarray(model.predict(features))
    except Exception:
        label_array = np.asarray(class_labels, dtype=object)
        predicted = label_array[np.argmax(proba, axis=1)]

    frame = _build_output_frame(df)
    return _attach_predictions(frame, proba=proba, predicted=predicted, class_labels=class_labels)
