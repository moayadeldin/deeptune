"""Inference for DeepTune's video modality. Mirrors inference/vision.py --
video models built by utils.build_video_model expose the same
forward(x) -> [B, num_classes] API as the image models, whether they're a
native spatio-temporal architecture or a frame-sampling wrapper around a 2D
backbone -- but need ParquetVideoDataset's clip decoding/sampling and
build_video_model's model construction instead of images' equivalents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.video_datasets import ParquetVideoDataset, collate_video_samples, native_video_transform
from helpers import transformations
from options import DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import build_video_model


def predict_video(
    df: pd.DataFrame,
    checkpoint_path: Path,
    model_version: str,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    freeze_backbone: bool,
    mode: str,
    use_peft: bool,
    batch_size: int,
    num_frames: int = 8,
    pooling: str = 'mean',
    label_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """Runs `checkpoint_path`'s model over every row of `df` (a 'videos'
    [+ optional 'labels'] byte-column convention). Returns the same shape
    of predictions DataFrame as inference.vision.predict_images."""
    model = build_video_model(
        model_version=model_version, num_classes=num_classes, added_layers=added_layers,
        embed_size=embed_size, freeze_backbone=freeze_backbone, mode=mode,
        use_peft=use_peft, pooling=pooling,
    )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # See inference.vision.predict_images's matching comment: ParquetVideoDataset
    # needs 'labels' to already be numeric (or absent) to convert it to a
    # tensor, but inference.data's raw-folder loader produces class *names*
    # -- dropped here, and matched back in afterwards by name instead.
    raw_labels = df['labels'].tolist() if 'labels' in df.columns else None
    dataset = ParquetVideoDataset(df.drop(columns=['labels']) if raw_labels is not None else df,
                                  transform=transformations, num_frames=num_frames,
                                  clip_transform=native_video_transform(model_version))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM, persistent_workers=PERSIST_WORK, collate_fn=collate_video_samples,
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
