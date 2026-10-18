"""DeepTune's Test/Inference CLI: load a previously trained model (a
checkpoint from a completed deeptune.py run) and run it on new data.

Mirrors deeptune.py's one-call shape (one flat argparse CLI, one output
directory per invocation) but for prediction instead of training: given
--checkpoint and enough of the original run's model-build settings (model
version, number of classes, etc. -- exactly what desktop/config.py's
InferenceConfig carries over from the picked run's own cli_arguments.json)
plus --data pointing at new data, it writes predictions.csv (one row per
sample: predicted class/value, confidence, and true label/error when the
new data has one) and, only when the new data had labels to compare
against, inference_metrics.json (aggregate accuracy or MAE/RMSE).

Usage:
    python infer.py --modality images --checkpoint <path> --data <folder>
        --out <dir> --model_version resnet18 --num_classes 3 --mode cls
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference.common import summarize_predictions
from inference.data import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, load_labeled_or_flat_files, load_single_file,
    load_tabular_like, load_video_manifest_for_inference,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run a trained DeepTune model on new data.')
    p.add_argument('--modality', required=True, choices=['images', 'text', 'tabular', 'timeseries', 'video'])
    p.add_argument('--checkpoint', required=True, type=Path, help="The run's checkpoint (file or directory, per modality).")
    p.add_argument('--data', required=True, type=Path, help='New data: a folder (images/video) or a CSV/XLSX/parquet file.')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--label_mapping', type=Path, default=None, help="The original run's data_splits_*/label_mapping.json, if any.")

    # Images/video model-build settings.
    p.add_argument('--model_version', default=None)
    p.add_argument('--model_str', choices=['BERT', 'gpt2'], default=None, help='Text model family (text modality only).')
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--mode', choices=['cls', 'reg'], default='cls')
    p.add_argument('--added_layers', type=int, default=2)
    p.add_argument('--embed_size', type=int, default=1000)
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--use-peft', action='store_true')

    # Video only.
    p.add_argument('--num_frames', type=int, default=8)
    p.add_argument('--pooling', choices=['mean', 'attention'], default='mean')

    # Tabular only.
    p.add_argument('--target', default='labels')
    p.add_argument('--finetuning-mode', action='store_true', help='TabPFN only.')

    # Timeseries only.
    p.add_argument('--time_idx_column', default='labels')
    p.add_argument('--max_encoder_length', type=int, default=30)
    p.add_argument('--group_ids', nargs='+', default=None)
    p.add_argument('--history_df', nargs='*', type=Path, default=[],
                   help="The original run's train/val split parquet files, supplying history for DeepAR.")
    return p


def _load_new_data(args: argparse.Namespace):
    """Returns (dataframe, has_labels) for --data, per modality."""
    if args.modality == 'images':
        if args.data.is_dir():
            return load_labeled_or_flat_files(args.data, IMAGE_EXTENSIONS, 'images')
        if args.data.is_file() and args.data.suffix.lower() in IMAGE_EXTENSIONS:
            return load_single_file(args.data, 'images'), False
        raise ValueError('Images inference needs a folder of image files, or a single image file.')
    if args.modality == 'video':
        if args.data.is_dir():
            return load_labeled_or_flat_files(args.data, VIDEO_EXTENSIONS, 'videos')
        if args.data.suffix.lower() in ('.csv', '.xlsx'):
            return load_video_manifest_for_inference(args.data)
        if args.data.is_file() and args.data.suffix.lower() in VIDEO_EXTENSIONS:
            return load_single_file(args.data, 'videos'), False
        raise ValueError('Video inference needs a folder of clips, a single clip, or a CSV/XLSX list of video paths.')
    df = load_tabular_like(args.data)
    if args.modality == 'text':
        return df, 'labels' in df.columns
    if args.modality == 'tabular':
        return df, args.target in df.columns
    return df, args.target in df.columns  # timeseries


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    label_mapping = None
    if args.label_mapping is not None and args.label_mapping.exists():
        label_mapping = json.loads(args.label_mapping.read_text(encoding='utf-8-sig'))

    df, has_labels = _load_new_data(args)
    print(f'Loaded {len(df)} sample(s) for inference ({"labeled" if has_labels else "unlabeled"}).')

    if args.modality == 'images':
        from utils import get_model_architecture
        from inference.vision import predict_images
        predictions = predict_images(
            df, args.checkpoint, get_model_architecture(args.model_version), args.model_version,
            args.num_classes, args.added_layers, args.embed_size, args.freeze_backbone,
            args.mode, args.use_peft, args.batch_size, label_mapping,
        )
    elif args.modality == 'video':
        from inference.video import predict_video
        predictions = predict_video(
            df, args.checkpoint, args.model_version, args.num_classes, args.added_layers,
            args.embed_size, args.freeze_backbone, args.mode, args.use_peft, args.batch_size,
            args.num_frames, args.pooling, label_mapping,
        )
    elif args.modality == 'text':
        from inference.text import predict_text
        family = 'bert' if (args.model_str or 'BERT').upper() == 'BERT' else 'gpt2'
        predictions = predict_text(df, args.checkpoint, family, args.batch_size, label_mapping=label_mapping)
    elif args.modality == 'tabular':
        if args.model_version == 'tabpfn':
            from inference.tabular import predict_tabpfn
            predictions = predict_tabpfn(df, args.checkpoint, args.target, args.mode, args.finetuning_mode)
        else:
            from inference.tabular import predict_gandalf
            predictions = predict_gandalf(df, args.checkpoint, args.target)
    elif args.modality == 'timeseries':
        from inference.timeseries import predict_timeseries
        predictions = predict_timeseries(
            df, args.checkpoint, args.history_df, args.time_idx_column, args.target,
            args.batch_size, args.max_encoder_length, args.group_ids,
        )
    else:
        raise ValueError(f'Unsupported modality: {args.modality}')

    predictions_path = args.out / 'predictions.csv'
    predictions.to_csv(predictions_path, index=False)
    print(f'Wrote {len(predictions)} prediction(s) to {predictions_path}')

    # summarize_predictions works off column names ('correct' for
    # classification, 'abs_error' for regression) every predict_* function
    # above already writes when the input data had labels, regardless of
    # modality, so one call covers all of them.
    metrics = summarize_predictions(predictions, args.mode)
    if metrics is not None:
        metrics_path = args.out / 'inference_metrics.json'
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f'Wrote metrics to {metrics_path}: {metrics}')
    else:
        print('No ground truth available in the new data; predictions only, no accuracy metrics.')

    print('Inference complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
