"""GUI <-> CLI bridge for DeepTune's desktop app.

Everything here is pure Python (dataclasses, pathlib, json/csv/parquet
reading) with no DearPyGui import, so it can be exercised directly by tests
without a display. It mirrors deeptune.py's own one-call ("onecall") CLI
argument set and on-disk output layout exactly - see cli.py's
DeepTuneVisionOptions._add_onecall_args/_add_default_args for the argument
side, and deeptune.py's main()/helpers.date_id for the output-directory side.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# What each modality actually accepts as --model_version, and what each
# needs beyond the common fields. Keep in sync with utils.MODEL_CLS_MAP,
# utils.VIDEO3D_ARCHITECTURES, and deeptune.py's per-modality dispatch.
# ---------------------------------------------------------------------------

IMAGE_MODEL_VERSIONS: dict[str, list[str]] = {
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "densenet": ["densenet121", "densenet161", "densenet169", "densenet201"],
    "swin": ["swin_t", "swin_s", "swin_b"],
    "efficientnet": [f"efficientnet_b{i}" for i in range(8)],
    "vgg": ["vgg11", "vgg13", "vgg16", "vgg19"],
    "vit": ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"],
    "convnext": ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
    "siglip": ["siglip"],
}

# The frame-pooling video path wraps a 2D backbone via get_model_cls(), which
# only resolves architectures registered in utils.MODEL_CLS_MAP -- siglip is
# handled through an entirely separate custom code path and isn't in that
# map, so it cannot be used as a video frame backbone.
VIDEO_FRAME_MODEL_VERSIONS: list[str] = [
    v for family, versions in IMAGE_MODEL_VERSIONS.items() if family != "siglip" for v in versions
]
VIDEO_NATIVE_MODEL_VERSIONS: list[str] = [
    "r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s", "swin3d_t", "swin3d_s", "swin3d_b",
]

MODEL_VERSIONS_BY_MODALITY: dict[str, list[str]] = {
    "images": [v for versions in IMAGE_MODEL_VERSIONS.values() for v in versions],
    "text": ["BERT", "gpt2"],
    "tabular": ["gandalf", "tabpfn"],
    "timeseries": ["deepAR"],
    "video": [*VIDEO_FRAME_MODEL_VERSIONS, *VIDEO_NATIVE_MODEL_VERSIONS],
}

MODALITIES = ["images", "text", "tabular", "timeseries", "video"]

# Modalities/models where --num_classes sizes a classification head.
NEEDS_NUM_CLASSES = {"images", "text", "video"}
# Modalities/models where --added_layers/--embed_size/--freeze-backbone/
# --use-peft/--mode (transfer-learning knobs) actually apply.
NEEDS_TRANSFER_LEARNING_OPTIONS = {"images", "text", "video"}


@dataclass
class RunConfig:
    """Every setting the GUI can control for one deeptune.py one-call run."""

    modality: str = "images"
    df: Optional[Path] = None  # raw dataset folder, or an existing parquet file
    raw_data: bool = True
    out: Optional[Path] = None
    target: str = "labels"
    grouper: str = ""

    model_version: str = "resnet18"
    num_classes: int = 2
    mode: str = "cls"  # "cls" | "reg"
    added_layers: int = 2
    embed_size: int = 1000
    freeze_backbone: bool = False
    use_peft: bool = False
    fixed_seed: bool = True
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4

    # video only
    num_frames: int = 8
    pooling: str = "mean"  # "mean" | "attention"

    # tabular / gandalf only
    gandalf_type: str = "classification"  # "classification" | "regression"
    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)

    # tabular / tabpfn only
    finetuning_mode: bool = False

    # timeseries / deepAR only
    time_idx_column: str = "labels"


def build_argv(cfg: RunConfig) -> list[str]:
    """Translate a RunConfig into deeptune.py's onecall CLI argument list."""
    if cfg.df is None or cfg.out is None:
        raise ValueError("Both a dataset path and an output directory are required.")

    argv = [
        "--modality", cfg.modality,
        "--df", str(cfg.df),
        "--out", str(cfg.out),
        "--model_version", cfg.model_version,
        "--mode", effective_mode(cfg),
        "--batch_size", str(cfg.batch_size),
        "--added_layers", str(cfg.added_layers),
        "--embed_size", str(cfg.embed_size),
        "--num_epochs", str(cfg.num_epochs),
        "--learning_rate", str(cfg.learning_rate),
        "--target", cfg.target or "labels",
    ]
    if cfg.num_classes:
        argv += ["--num_classes", str(cfg.num_classes)]
    if cfg.grouper.strip():
        argv += ["--grouper", cfg.grouper.strip()]
    if cfg.raw_data:
        argv.append("--raw-data")
    if cfg.freeze_backbone:
        argv.append("--freeze-backbone")
    if cfg.use_peft:
        argv.append("--use-peft")
    if cfg.fixed_seed:
        argv.append("--fixed-seed")

    if cfg.modality == "video":
        argv += ["--num_frames", str(cfg.num_frames), "--pooling", cfg.pooling]

    if cfg.modality == "tabular" and cfg.model_version == "gandalf":
        argv += ["--type", cfg.gandalf_type]
        if cfg.continuous_cols:
            argv += ["--continuous_cols", *cfg.continuous_cols]
        if cfg.categorical_cols:
            argv += ["--categorical_cols", *cfg.categorical_cols]

    if cfg.modality == "tabular" and cfg.model_version == "tabpfn" and cfg.finetuning_mode:
        argv.append("--finetuning-mode")

    if cfg.modality == "timeseries":
        argv += ["--time_idx_column", cfg.time_idx_column]

    return argv


def validate_config(cfg: RunConfig) -> list[str]:
    """Cheap, GUI-side sanity checks -- deeptune.py itself is the source of
    truth and will still validate everything again; this just avoids
    launching a process for the most common mistakes."""
    issues: list[str] = []
    if cfg.df is None or not str(cfg.df).strip():
        issues.append("Choose a dataset (a raw data folder or an existing .parquet file).")
    elif not Path(cfg.df).exists():
        issues.append(f"Dataset path does not exist: {cfg.df}")
    elif cfg.raw_data and cfg.modality in ('images', 'video') and not Path(cfg.df).is_dir():
        if cfg.modality == 'images':
            issues.append('Raw images require a folder with one subfolder per class.')
        elif Path(cfg.df).suffix.lower() not in ('.csv', '.xlsx'):
            issues.append('Video selected for preview. Add labeled clips below and click Use clip dataset to train.')
    elif not (cfg.raw_data and cfg.modality in ('images', 'video')):
        allowed = {'.csv', '.xlsx'} if cfg.raw_data else {'.parquet'}
        if not Path(cfg.df).is_file() or Path(cfg.df).suffix.lower() not in allowed:
            issues.append(f"Choose a file with one of these extensions: {', '.join(sorted(allowed))}.")
    if cfg.out is None or not str(cfg.out).strip():
        issues.append("Choose an output directory.")
    elif Path(cfg.out).exists() and not Path(cfg.out).is_dir():
        issues.append('The output directory points to a file. Choose a folder.')
    if cfg.modality not in MODALITIES:
        issues.append('Choose a supported modality.')
    elif cfg.model_version not in MODEL_VERSIONS_BY_MODALITY[cfg.modality]:
        issues.append('Choose a model supported by this modality.')
    if cfg.mode not in ('cls', 'reg'):
        issues.append('Task must be classification (cls) or regression (reg).')
    for name, label in (('batch_size', 'Batch size'), ('num_epochs', 'Epochs'), ('embed_size', 'Embedding size')):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.append(f'{label} must be a positive integer.')
    if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        issues.append('Learning rate must be a finite number greater than zero.')
    if not cfg.target.strip():
        issues.append('Enter the target column name.')
    if cfg.grouper and cfg.grouper == cfg.target:
        issues.append('The grouper and target must be different columns.')
    if cfg.modality in NEEDS_NUM_CLASSES and cfg.mode == "cls" and cfg.num_classes < 2:
        issues.append("Number of classes must be at least 2 for classification.")
    if cfg.added_layers not in (1, 2):
        issues.append("Added layers must be 1 or 2.")
    if cfg.modality == 'text' and cfg.mode != 'cls':
        issues.append('Text models currently support classification only.')
    if cfg.model_version == 'siglip' and cfg.mode != 'cls':
        issues.append('SigLIP currently supports classification only.')
    if cfg.use_peft and (cfg.modality not in NEEDS_TRANSFER_LEARNING_OPTIONS or cfg.model_version == 'gpt2'):
        issues.append('PEFT is not supported for this model.')
    if cfg.raw_data and cfg.modality in ('images', 'video') and cfg.mode == 'reg' and cfg.df is not None and Path(cfg.df).is_dir():
        issues.append('Regression requires a parquet file with numeric targets; class folders are for classification.')
    if cfg.modality == 'video':
        from desktop.dataset_validation import video_file_split_issues
        issues.extend(video_file_split_issues(cfg))
        if cfg.num_frames < 1:
            issues.append('Frames per clip must be at least 1.')
        if cfg.pooling not in ('mean', 'attention'):
            issues.append('Choose mean or attention pooling.')
        if cfg.model_version == 'mvit_v2_s' and cfg.num_frames != 16:
            issues.append('mvit_v2_s requires exactly 16 frames per clip.')
    if cfg.modality == 'timeseries' and (not cfg.time_idx_column.strip() or cfg.time_idx_column == cfg.target):
        issues.append('Forecasting needs a time index column different from the target.')
    if cfg.modality == "tabular" and cfg.model_version == "gandalf":
        if not cfg.continuous_cols and not cfg.categorical_cols:
            issues.append("GANDALF needs at least one continuous or categorical column.")
        features = cfg.continuous_cols + cfg.categorical_cols
        if len(set(features)) != len(features):
            issues.append('Feature columns must be unique across both lists.')
        if cfg.target in features or 'labels' in features:
            issues.append('The target must not be included as an input feature.')
        if cfg.gandalf_type not in ('classification', 'regression'):
            issues.append('Choose classification or regression for GANDALF.')
    return issues


def effective_mode(cfg: RunConfig) -> str:
    if cfg.modality == 'timeseries':
        return 'reg'
    if cfg.modality == 'tabular' and cfg.model_version == 'gandalf':
        return 'reg' if cfg.gandalf_type == 'regression' else 'cls'
    return cfg.mode


def resolved_config(cfg: RunConfig) -> RunConfig:
    return replace(cfg, df=Path(cfg.df).expanduser().resolve() if cfg.df is not None else None,
                   out=Path(cfg.out).expanduser().resolve() if cfg.out is not None else None)


def write_json(path: Path, payload: dict) -> None:
    """Replace a JSON file atomically; a failed save preserves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_config(cfg: RunConfig, path: Path) -> None:
    payload = asdict(resolved_config(cfg))
    for key in ('df', 'out'):
        payload[key] = str(payload[key]) if payload[key] is not None else None
    write_json(path, {'version': 1, 'config': payload})


def load_config(path: Path) -> RunConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or payload.get('version') != 1 or not isinstance(payload.get('config'), dict):
        raise ValueError('Choose a DeepTune settings file (version 1).')
    values = payload['config'].copy()
    defaults = RunConfig()
    known = {item.name for item in fields(defaults)}
    if set(values) - known:
        raise ValueError('Settings contain unknown fields; this file may need a newer DeepTune version.')
    for key, value in values.items():
        if key in ('df', 'out'):
            if value is not None and not isinstance(value, str):
                raise ValueError(f'{key} must be a path string.')
            if value:
                candidate = Path(value).expanduser()
                values[key] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
            else:
                values[key] = None
        else:
            expected = type(getattr(defaults, key))
            valid = type(value) is expected or (expected is float and type(value) is int)
            if not valid or (expected is list and not all(isinstance(v, str) for v in value)):
                raise ValueError(f'Invalid value for {key}.')
    cfg = RunConfig(**values)
    if cfg.modality not in MODALITIES or cfg.model_version not in MODEL_VERSIONS_BY_MODALITY[cfg.modality]:
        raise ValueError('Settings contain an unsupported modality or model.')
    return cfg


# ---------------------------------------------------------------------------
# Reading back what a completed (or in-progress) run produced. deeptune.py's
# main() lays every run out as:
#   <out>/deeptune-<date>-exp<N>/                     (helpers.date_id)
#       <modality>_dataset_<UNIQUE_ID>.parquet          (only if --raw-data)
#       data_splits_<UNIQUE_ID>/label_mapping.json
#       trainval_output_<model_str>_<UNIQUE_ID>/training_log.csv
#       eval_output_<model_str>_<UNIQUE_ID>/full_metrics.json
#       embed_output_<model_str>_<UNIQUE_ID>/*.parquet
# All four of the *_output_* directories share one UNIQUE_ID per run (it's a
# module-level constant computed once when options.py is first imported).
# ---------------------------------------------------------------------------

RUN_DIR_PREFIX = "deeptune-"


def list_runs(out_dir: Path) -> list[Path]:
    """Every experiment folder under `out_dir`, most recent first."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    runs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(RUN_DIR_PREFIX)]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_run(out_dir: Path) -> Optional[Path]:
    runs = list_runs(out_dir)
    return runs[0] if runs else None


def _first_glob(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


@dataclass
class RunResults:
    run_dir: Path
    metrics: Optional[dict] = None
    training_log: Optional[pd.DataFrame] = None
    label_mapping: Optional[dict] = None
    checkpoint_path: Optional[Path] = None
    embeddings_path: Optional[Path] = None
    embeddings_shape: Optional[tuple[int, int]] = None
    cli_arguments: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    run_status: Optional[dict] = None


def read_run_results(run_dir: Path) -> RunResults:
    """Best-effort read of everything a one-call run leaves behind. Missing
    pieces (e.g. a run that's still training, with no eval/embed output yet)
    are simply left as None rather than raising."""
    run_dir = Path(run_dir)
    results = RunResults(run_dir=run_dir)

    def read_json(path: Optional[Path]) -> Optional[dict]:
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError('Expected a JSON object.')
            return value
        except (OSError, ValueError) as exc:
            results.warnings.append(f'{path.name}: {exc}')
            return None

    eval_metrics_path = _first_glob(run_dir, "eval*/full_metrics.json") or _first_glob(run_dir, 'test_output_*/full_metrics.json')
    if eval_metrics_path is not None:
        results.metrics = read_json(eval_metrics_path)

    training_log_path = _first_glob(run_dir, "train*/training_log.csv")
    if training_log_path is not None:
        try:
            results.training_log = pd.read_csv(training_log_path)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            results.warnings.append(f'training_log.csv: {exc}')

    checkpoint_path = next((match for pattern in ('train*/model_weights.pth', 'train*/*.ckpt', 'train*/*.tabpfn_fit', 'train*/*.joblib', 'train*/GANDALF_model')
                            if (match := _first_glob(run_dir, pattern)) is not None), None)
    if checkpoint_path is not None:
        results.checkpoint_path = checkpoint_path

    label_mapping_path = _first_glob(run_dir, "data_splits_*/label_mapping.json")
    if label_mapping_path is not None:
        results.label_mapping = read_json(label_mapping_path)

    embed_dir = _first_glob(run_dir, "embed*")
    if embed_dir is not None and embed_dir.is_dir():
        embed_file = _first_glob(embed_dir, "*.parquet")
        if embed_file is not None:
            results.embeddings_path = embed_file
            try:
                import pyarrow.parquet as pq
                metadata = pq.read_metadata(embed_file)
                results.embeddings_shape = (metadata.num_rows, metadata.num_columns)
            except (OSError, ValueError) as exc:
                results.warnings.append(f'{embed_file.name}: {exc}')

    cli_args_path = _first_glob(run_dir, "eval*/cli_arguments.json") or _first_glob(
        run_dir, "train*/cli_arguments.json"
    )
    if cli_args_path is not None:
        results.cli_arguments = read_json(cli_args_path)
    results.run_status = read_json(run_dir / 'desktop-run.json') if (run_dir / 'desktop-run.json').exists() else None

    return results


def metric_rows(metrics: Optional[dict], prefix: str = '') -> list[tuple[str, str]]:
    """Flatten every model's metrics without inventing absent values or units."""
    rows = []
    for key, value in (metrics or {}).items():
        label = f'{prefix} / {key}' if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(metric_rows(value, label))
        elif isinstance(value, float):
            rows.append((label, f'{value:.6g}' if math.isfinite(value) else 'Unavailable'))
        else:
            rows.append((label, 'Unavailable' if value is None else str(value)))
    return rows


def loss_series(frame: Optional[pd.DataFrame], *columns: str) -> list[list[float]]:
    if frame is None or frame.empty:
        return [[], []]
    column = next((name for name in columns if name in frame.columns), None)
    if column is None:
        return [[], []]
    epochs = pd.to_numeric(frame['epoch'], errors='coerce') if 'epoch' in frame else pd.Series(range(1, len(frame) + 1), index=frame.index)
    values = pd.to_numeric(frame[column], errors='coerce')
    valid = epochs.map(lambda v: math.isfinite(v)) & values.map(lambda v: math.isfinite(v))
    return [epochs[valid].astype(float).tolist(), values[valid].astype(float).tolist()]


def export_report(results: RunResults) -> Path:
    import csv
    destination = results.run_dir / 'metrics-summary.csv'
    with destination.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('Metric', 'Value'))
        writer.writerows(metric_rows(results.metrics))
    return destination


# ---------------------------------------------------------------------------
# Test/Inference: running an already-completed run's checkpoint on new data
# via infer.py, the prediction-only sibling of deeptune.py's one-call CLI
# (see inference/ for the actual per-modality prediction code, and
# runner.start_infer_run for how the GUI launches it as a subprocess the
# same way runner.start_run launches deeptune.py).
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Everything the Test page needs to run infer.py against one picked
    run. Most fields are carried over from that run's own recorded
    cli_arguments.json (see infer_config_from_run) rather than re-entered."""

    run_dir: Optional[Path] = None
    modality: str = "images"
    checkpoint_path: Optional[Path] = None
    data: Optional[Path] = None  # new data: a folder (images/video) or a CSV/XLSX/parquet file
    out: Optional[Path] = None
    batch_size: int = 16
    label_mapping_path: Optional[Path] = None

    model_version: str = "resnet18"
    model_str: str = "BERT"  # text modality only: "BERT" | "gpt2"
    num_classes: int = 2
    mode: str = "cls"
    added_layers: int = 2
    embed_size: int = 1000
    freeze_backbone: bool = False
    use_peft: bool = False
    num_frames: int = 8
    pooling: str = "mean"
    target: str = "labels"
    finetuning_mode: bool = False
    time_idx_column: str = "labels"
    max_encoder_length: int = 30
    group_ids: list[str] = field(default_factory=list)
    history_df_paths: list[Path] = field(default_factory=list)


def find_history_paths(run_dir: Path) -> list[Path]:
    """The train/validation parquet splits a completed run produced,
    supplying DeepAR inference with the history window it needs to make a
    prediction at all -- see inference/timeseries.py."""
    run_dir = Path(run_dir)
    paths = []
    for name in ('train_split.parquet', 'val_split.parquet'):
        found = _first_glob(run_dir, f'data_splits_*/{name}')
        if found is not None:
            paths.append(found)
    return paths


def infer_config_from_run(results: RunResults, data: Path, out: Path, batch_size: int = 16) -> InferenceConfig:
    """Builds an InferenceConfig for testing `results`'s trained model
    against `data`, pulling every model-build setting from the run's own
    cli_arguments.json (written by DeepTuneVisionOptions.save_args at
    training time) so the user doesn't have to re-enter them -- deeptune.py
    itself sets args.mode to the already-resolved effective mode (e.g.
    GANDALF regression vs. classification, timeseries always 'reg') before
    that file is saved, so `mode` below needs no further resolution here."""
    cli_args = results.cli_arguments or {}
    modality = cli_args.get('modality', 'images')
    model_version = cli_args.get('model_version') or 'resnet18'
    cfg = InferenceConfig(
        run_dir=results.run_dir,
        modality=modality,
        checkpoint_path=results.checkpoint_path,
        data=data,
        out=out,
        batch_size=batch_size,
        model_version=model_version,
        model_str='gpt2' if model_version == 'gpt2' else 'BERT',
        num_classes=int(cli_args.get('num_classes') or 2),
        mode=cli_args.get('mode') or 'cls',
        added_layers=int(cli_args.get('added_layers') or 2),
        embed_size=int(cli_args.get('embed_size') or 1000),
        freeze_backbone=bool(cli_args.get('freeze_backbone', False)),
        use_peft=bool(cli_args.get('use_peft', False)),
        num_frames=int(cli_args.get('num_frames') or 8),
        pooling=cli_args.get('pooling') or 'mean',
        target=cli_args.get('target') or 'labels',
        finetuning_mode=bool(cli_args.get('finetuning_mode', False)),
        time_idx_column=cli_args.get('time_idx_column') or 'labels',
        max_encoder_length=int(cli_args.get('max_encoder_length') or 30),
        group_ids=list(cli_args.get('group_ids') or []) if isinstance(cli_args.get('group_ids'), list) else [],
    )
    label_mapping_path = _first_glob(results.run_dir, 'data_splits_*/label_mapping.json') if results.run_dir else None
    cfg.label_mapping_path = label_mapping_path
    if modality == 'timeseries':
        cfg.history_df_paths = find_history_paths(results.run_dir)
    return cfg


def build_infer_argv(cfg: InferenceConfig) -> list[str]:
    """Translate an InferenceConfig into infer.py's CLI argument list."""
    if cfg.checkpoint_path is None or cfg.data is None or cfg.out is None:
        raise ValueError('A trained run, new data, and an output folder are all required.')
    argv = [
        '--modality', cfg.modality,
        '--checkpoint', str(cfg.checkpoint_path),
        '--data', str(cfg.data),
        '--out', str(cfg.out),
        '--batch_size', str(cfg.batch_size),
        '--mode', cfg.mode,
    ]
    if cfg.label_mapping_path is not None:
        argv += ['--label_mapping', str(cfg.label_mapping_path)]
    if cfg.modality in ('images', 'video'):
        argv += [
            '--model_version', cfg.model_version,
            '--num_classes', str(cfg.num_classes),
            '--added_layers', str(cfg.added_layers),
            '--embed_size', str(cfg.embed_size),
        ]
        if cfg.freeze_backbone:
            argv.append('--freeze-backbone')
        if cfg.use_peft:
            argv.append('--use-peft')
        if cfg.modality == 'video':
            argv += ['--num_frames', str(cfg.num_frames), '--pooling', cfg.pooling]
    if cfg.modality == 'text':
        argv += ['--model_str', cfg.model_str]
    if cfg.modality == 'tabular':
        argv += ['--target', cfg.target, '--model_version', cfg.model_version]
        if cfg.model_version == 'tabpfn' and cfg.finetuning_mode:
            argv.append('--finetuning-mode')
    if cfg.modality == 'timeseries':
        argv += ['--target', cfg.target, '--time_idx_column', cfg.time_idx_column,
                 '--max_encoder_length', str(cfg.max_encoder_length)]
        if cfg.group_ids:
            argv += ['--group_ids', *cfg.group_ids]
        if cfg.history_df_paths:
            argv += ['--history_df', *[str(p) for p in cfg.history_df_paths]]
    return argv


def validate_infer_config(cfg: InferenceConfig) -> list[str]:
    """Cheap, GUI-side sanity checks -- infer.py itself is the source of
    truth and will still validate everything again."""
    issues: list[str] = []
    if cfg.run_dir is None or cfg.checkpoint_path is None:
        issues.append('Choose a completed run with a saved checkpoint.')
    if cfg.data is None or not str(cfg.data).strip():
        issues.append('Choose new data to test against.')
    elif not Path(cfg.data).exists():
        issues.append(f'Data path does not exist: {cfg.data}')
    if cfg.out is None or not str(cfg.out).strip():
        issues.append('Choose an output folder for predictions.')
    if cfg.modality == 'timeseries' and not cfg.history_df_paths:
        issues.append("This run's train/validation splits could not be found under its folder; "
                      "DeepAR inference needs them to supply history.")
    return issues


@dataclass
class InferenceResults:
    out_dir: Path
    predictions: Optional[pd.DataFrame] = None
    metrics: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


def read_inference_results(out_dir: Path) -> InferenceResults:
    """Best-effort read of what one infer.py run left behind."""
    out_dir = Path(out_dir)
    results = InferenceResults(out_dir=out_dir)
    predictions_path = out_dir / 'predictions.csv'
    if predictions_path.exists():
        try:
            results.predictions = pd.read_csv(predictions_path)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            results.warnings.append(f'predictions.csv: {exc}')
    metrics_path = out_dir / 'inference_metrics.json'
    if metrics_path.exists():
        try:
            results.metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as exc:
            results.warnings.append(f'inference_metrics.json: {exc}')
    return results
