from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting
import pandas as pd
from src.timeseries.deepar import deepAR # our wrapper class
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID
import numpy as np
from helpers import save_timeseries_prediction_to_json
from pytorch_forecasting.models.deepar import DeepAR
import time 
from utils import save_process_times


def main():
    
    args = DeepTuneVisionOptions(RunType.TIMESERIES)
    TRAIN_DF_PATH = args.train_df
    VAL_DF_PATH = args.val_df
    EVAL_DF_PATH = args.eval_df
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
    
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    eval_df = pd.read_parquet(EVAL_DF_PATH)
    
    TEST_OUTPUT_DIR = (OUT / f"test_deepAR_output_{UNIQUE_ID}")
    
    train_df["__split"] = "train"
    val_df["__split"] = "val"
    eval_df["__split"] = "eval"
    
    df = pd.concat([train_df, val_df, eval_df],ignore_index=True)

    
    """
    
    We assume we work with a single time series.
    
    """
    
    df['group'] = '0'
    
    df[TIMEINDEX_COLUMN] = pd.to_datetime(df[TIMEINDEX_COLUMN])
    df = df.sort_values(TIMEINDEX_COLUMN)
    
    time_col = df[TIMEINDEX_COLUMN]
    df["time_idx"] = ((time_col - time_col.min()).dt.total_seconds() // 3600).astype(int)
    
    GROUP_IDS = ['group'] if args.group_ids is None else args.group_ids
    
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.float64)
    
    train_df = df[df["__split"] == "train"].copy()
    val_df = df[df["__split"] == "val"].copy()
    eval_df = df[df["__split"] == "eval"].copy()
    
    model = DeepAR.load_from_checkpoint(MODEL_WEIGHTS)
    model.eval()
    
    hist_for_test = (
    pd.concat([train_df, val_df], ignore_index=True)
      .sort_values(["group", "time_idx"])
      .groupby("group", as_index=False)
      .tail(MAX_ENCODER_LENGTH)
    )

    test_plus_hist = (
        pd.concat([hist_for_test, eval_df], ignore_index=True)
        .sort_values(["group", "time_idx"])
    )
    
    training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx = "time_idx",
    target=TARGET_COLUMN,
    max_prediction_length = MAX_PREDICTION_LENGTH,
    max_encoder_length = MAX_ENCODER_LENGTH,
    time_varying_known_categoricals = TIME_VARYING_KNOWN_CATEGORICALS,
    time_varying_unknown_categoricals = TIME_VARYING_UNKNOWN_CATEGORICALS,
    static_categoricals = STATIC_CATEGORICALS,
    time_varying_known_reals = TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals=(TIME_VARYING_UNKNOWN_REALS) + [TARGET_COLUMN],
    static_reals = STATIC_REALS,
    group_ids = GROUP_IDS,
    allow_missing_timesteps=True,
    target_normalizer=pytorch_forecasting.data.encoders.TorchNormalizer()
    
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_plus_hist,
        predict=True,
        stop_randomization=True
    )

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    start_time = time.time()
    
    pred = model.predict(
        test_loader,
        mode="prediction",
        return_x=True,
        return_index=True,
        return_decoder_lengths=True
    )
    
    print(f"Model's prediction of the target {TARGET_COLUMN} in the evaluation/test set is {pred.output.squeeze().item():.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    save_timeseries_prediction_to_json(pred, TEST_OUTPUT_DIR)
    args.save_args(TEST_OUTPUT_DIR)
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="evaluation")
    
if __name__ == "__main__":
    main()
    