"""Inference for DeepTune's images modality (and, via inference/video.py's
reuse of predict(), the video modality's frame-sampling models too).

SigLIP is not supported here yet: it's built and loaded through an entirely
separate code path (src/vision/siglip.py, evaluators/vision/custom_siglip_evaluate.py)
that this first version of the Test page doesn't wire up.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.image_datasets import ParquetImageDataset
from helpers import transformations
from options import DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls


def _collate_optional_labels(samples):
    """The default collator can't batch a column of all-None labels."""
    from torch.utils.data import default_collate
    images = default_collate([sample[0] for sample in samples])
    labels = None if all(sample[1] is None for sample in samples) else default_collate([sample[1] for sample in samples])
    if len(samples[0]) == 3:
        return images, labels, default_collate([sample[2] for sample in samples])
    return images, labels


def predict_images(
    df: pd.DataFrame,
    checkpoint_path: Path,
    model_architecture: str,
    model_version: str,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    freeze_backbone: bool,
    mode: str,
    use_peft: bool,
    batch_size: int,
    label_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """Runs `checkpoint_path`'s model over every row of `df` (the same
    'images' [+ optional 'labels'] byte-column convention the training
    pipeline uses) and returns a per-sample predictions DataFrame:
    sample_id, predicted_class/predicted_value, confidence (classification
    only), and true_label/correct when `df` has a 'labels' column.

    `label_mapping` is the {class_name: index} mapping written alongside
    the original training run (data_splits_*/label_mapping.json), used to
    show the predicted class's original name rather than a bare index.
    """
    adjusted_model_cls = get_model_cls(model_architecture, use_peft=use_peft)
    if adjusted_model_cls is None:
        raise ValueError(f'Unsupported model architecture for images: {model_architecture}')
    model = adjusted_model_cls(num_classes, model_version, added_layers, embed_size,
                               task_type=mode, freeze_backbone=freeze_backbone)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # ParquetImageDataset converts whatever is in 'labels' straight to a
    # tensor (torch.tensor(row['labels'], ...)), which only works for
    # already-numeric labels -- exactly what a completed run's own parquet
    # splits hold (split_dataset.py label-encodes class names to integers
    # before saving them), but not what inference.data's raw-folder loader
    # produces (the class *names* themselves, e.g. 'red'/'blue', read
    # straight from each subfolder). The model doesn't need labels to make
    # a prediction, so they're dropped here and compared back in
    # afterwards against the predicted class's own name -- sidestepping
    # needing them in any particular encoding.
    raw_labels = df['labels'].tolist() if 'labels' in df.columns else None
    dataset = ParquetImageDataset(df.drop(columns=['labels']) if raw_labels is not None else df, transform=transformations)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM, persistent_workers=PERSIST_WORK, collate_fn=_collate_optional_labels,
    )

    sample_ids = df['__sample_id'].tolist() if '__sample_id' in df.columns else list(range(len(df)))
    index_to_name = {v: k for k, v in (label_mapping or {}).items()}

    predicted, confidences = [], []
    with torch.no_grad():
        for inputs, _labels, *_ in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            if mode == 'cls':
                probs = torch.softmax(outputs, dim=1)
                conf, pred = probs.max(dim=1)
                predicted.extend(pred.cpu().tolist())
                confidences.extend(conf.cpu().tolist())
            else:
                predicted.extend(outputs.detach().cpu().view(-1).tolist())
                confidences.extend([None] * outputs.size(0))

    result = pd.DataFrame({'sample_id': sample_ids[:len(predicted)]})
    if mode == 'cls':
        result['predicted_class'] = predicted
        result['predicted_label'] = [index_to_name.get(int(p), str(int(p))) for p in predicted]
        result['confidence'] = confidences
    else:
        result['predicted_value'] = predicted

    if raw_labels is not None:
        raw_labels = raw_labels[:len(predicted)]
        if mode == 'cls':
            result['true_label'] = raw_labels
            result['correct'] = [str(t) == str(p) for t, p in zip(raw_labels, result['predicted_label'])]
        else:
            result['true_value'] = raw_labels
            result['abs_error'] = [abs(float(t) - float(p)) for t, p in zip(raw_labels, predicted)]
    return result
