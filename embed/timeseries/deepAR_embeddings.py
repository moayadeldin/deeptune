from datasets.timeseries_data import prepare_timeseries
from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting

import pandas as pd

from src.timeseries.deepAR import deepAR
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID

import numpy as np
import torch
from pytorch_forecasting.models.deepar import DeepAR
import time
from utils import save_process_times
from pathlib import Path

def embed(
        eval_df: Path,
        out: Path,
        batch_size: int,
        timeindex_column: str,
        target_column: list,
        model_weights: Path,
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        group_ids = None,
        time_varying_known_categoricals: list = [],
        time_varying_unknown_categoricals: list = [],
        static_categoricals: list = [],
        time_varying_known_reals: list = [],
        time_varying_unknown_reals: list = [],
        static_reals: list = [],
        model_str='deepAR',
        args: DeepTuneVisionOptions = None,
        history_df_paths: list[Path] | None = None,
):
    
    started = time.time()
    original = pd.read_parquet(eval_df)
    if original.empty:
        raise ValueError('The embedding dataset is empty.')
    current = original.copy()
    current['__embedding_row'] = np.arange(len(current))
    history = []
    for path in history_df_paths or []:
        previous = pd.read_parquet(path)
        previous['__embedding_row'] = -1
        history.append(previous)
    combined = pd.concat([*history, current], ignore_index=True)
    frame, groups = prepare_timeseries(combined, timeindex_column, target_column, group_ids)
    rows_to_embed = frame[frame['__embedding_row'] >= 0].sort_values('__embedding_row')
    checkpoint = Path(model_weights)
    checkpoint = checkpoint if checkpoint.is_file() else next(checkpoint.glob('*.ckpt'))
    model = DeepAR.load_from_checkpoint(checkpoint, map_location='cpu')
    model.eval()
    params = model.dataset_parameters
    # Reuse training encoders/scalers. Fitting new ones on holdout data changes
    # the representation and can leak holdout distribution into the model.
    dataset = TimeSeriesDataSet.from_parameters(
        params, frame, predict=False, stop_randomization=True,
        min_encoder_length=2,
        min_prediction_idx=int(rows_to_embed['time_idx'].min()),
    )
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    embeddings = {}
    with torch.no_grad():
        for x, _ in loader:
            hidden = model.encode(x)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            vectors = hidden[-1].detach().cpu().numpy()
            index = dataset.x_to_index(x)
            for batch_index, row in index.iterrows():
                group = tuple(str(row[column]) for column in groups)
                length = int(x['decoder_lengths'][batch_index])
                for step in x['decoder_time_idx'][batch_index, :length].tolist():
                    embeddings[(*group, int(step))] = vectors[batch_index]
    matrix, padded = [], []
    for _, row in rows_to_embed.iterrows():
        key = (*tuple(str(row[column]) for column in groups), int(row['time_idx']))
        value = embeddings.get(key)
        padded.append(value is None)
        matrix.append(np.zeros(model.rnn.hidden_size, dtype=np.float32) if value is None else value)
    output = pd.DataFrame(np.stack(matrix), columns=[f'emb_{index}' for index in range(model.rnn.hidden_size)])
    output[target_column] = original[target_column].to_numpy()
    output['is_padded'] = padded
    for group in group_ids or []:
        output[group] = original[group].to_numpy()
    directory = out / f'embed_output_{model_str}_{UNIQUE_ID}'
    directory.mkdir(parents=True, exist_ok=True)
    output.to_parquet(directory / f'{model_str}_embeddings.parquet', index=False)
    if args is not None:
        args.save_args(directory)
    save_process_times(epoch_times=1, total_duration=time.time() - started, outdir=directory, process='embedding')
    print(f'Saved {len(output)} embeddings ({sum(padded)} rows without sufficient history).')
    return directory, output.shape


def main():

    args = DeepTuneVisionOptions(RunType.TIMESERIES)

    DF_PATH = args.df
    OUT = args.out
    BATCH_SIZE = args.batch_size
    TIMEINDEX_COLUMN = args.time_idx_column
    TARGET_COLUMN = args.target_column
    MAX_ENCODER_LENGTH = args.max_encoder_length
    MAX_PREDICTION_LENGTH = args.max_prediction_length
    TIME_VARYING_KNOWN_CATEGORICALS = args.time_varying_known_categoricals
    TIME_VARYING_UNKNOWN_CATEGORICALS = args.time_varying_unknown_categoricals
    STATIC_CATEGORICALS = args.static_categoricals
    TIME_VARYING_KNOWN_REALS = args.time_varying_known_reals
    TIME_VARYING_UNKNOWN_REALS = args.time_varying_unknown_reals
    STATIC_REALS = args.static_reals
    MODEL_WEIGHTS = args.model_weights
    MODEL_STR = 'deepAR'

    embed(
        eval_df=DF_PATH,
        out=OUT,
        batch_size=BATCH_SIZE,
        timeindex_column=TIMEINDEX_COLUMN,
        target_column=TARGET_COLUMN,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        static_reals=STATIC_REALS,
        model_weights=MODEL_WEIGHTS,
        model_str=MODEL_STR,
        args=args
    )

if __name__ == "__main__":
    main()