"""Inference for DeepTune's text modality (Multilingual BERT and GPT-2).

Doesn't reuse datasets/text_datasets.py's TextDataset: it unconditionally
requires a 'labels' column (`self.labels = self.data['labels'].tolist()`
raises KeyError otherwise), which is exactly the case inference needs to
support -- new text with no known label to compare against. Tokenizing a
plain Python list of strings directly, in batches, is simple enough not to
need a Dataset/DataLoader here at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from helpers import load_finetuned_gpt2, load_finetunedbert_model
from options import DEVICE


def predict_text(
    df: pd.DataFrame,
    checkpoint_dir: Path,
    model_family: str,  # "bert" | "gpt2"
    batch_size: int,
    max_length: int = 512,
    label_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """Runs the fine-tuned model saved under `checkpoint_dir` over every row
    of `df` (a 'text' [+ optional 'labels'] column convention). Text
    classification only -- matches what DeepTune's text modality trains
    today. Returns the same shape of predictions DataFrame as
    inference.vision.predict_images.
    """
    if 'text' not in df.columns:
        raise ValueError("Text inference data needs a 'text' column.")

    if model_family == 'bert':
        model, tokenizer = load_finetunedbert_model(checkpoint_dir)
    elif model_family == 'gpt2':
        model, tokenizer = load_finetuned_gpt2(checkpoint_dir)
    else:
        raise ValueError(f'Unsupported text model family: {model_family}')
    model.to(DEVICE)
    model.eval()

    texts = df['text'].tolist()
    sample_ids = df['__sample_id'].tolist() if '__sample_id' in df.columns else list(range(len(df)))
    has_labels = 'labels' in df.columns
    labels = df['labels'].tolist() if has_labels else [None] * len(df)
    index_to_name = {v: k for k, v in (label_mapping or {}).items()}

    predicted, confidences = [], []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoding = tokenizer(
                batch_texts, padding='max_length', truncation=True,
                max_length=max_length, return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            if model_family == 'bert':
                token_type_ids = encoding.get('token_type_ids')
                token_type_ids = token_type_ids.to(DEVICE) if token_type_ids is not None else None
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(dim=1)
            predicted.extend(pred.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    result = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_class': predicted,
        'predicted_label': [index_to_name.get(int(p), str(int(p))) for p in predicted],
        'confidence': confidences,
    })
    if has_labels:
        result['true_class'] = labels
        result['true_label'] = [index_to_name.get(int(v), str(int(v))) if v is not None else None for v in labels]
        result['correct'] = [None if v is None else (int(v) == int(p)) for v, p in zip(labels, predicted)]
    return result
