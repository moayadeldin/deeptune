from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting
import pandas as pd
from src.timeseries.deepar import deepAR
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
from helpers import save_timeseries_prediction_to_json
from pytorch_forecasting.metrics import MAE
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from lightning.pytorch import Trainer
from options import UNIQUE_ID

def main():
    
    
    args = DeepTuneVisionOptions(RunType.TIMESERIES)
    TRAIN_DF_PATH = args.train_df
    VAL_DF_PATH = args.val_df
    NUM_EPOCHS = args.num_epochs
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
    
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    
    train_df["__split"] = "train"
    val_df["__split"] = "val"
    
    df = pd.concat([train_df, val_df],ignore_index=True)
    
    logger = TensorBoardLogger(
    save_dir=OUT,
    name=UNIQUE_ID,
    version=""
)

    
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
    
    GRADIENT_CLIP_VAL = 1e-1
        
    # IMPORTANT: TimeSeries models needs the last "max_encoder_length" timesteps to predict the next "max_prediction_length". Hence, we must get these timesteps from the training set to initialize the encoder in validation
    hist = train_df.sort_values(["group","time_idx"]) \
                .groupby("group", as_index=False) \
                .tail(MAX_ENCODER_LENGTH)

    val_plus_hist = pd.concat([hist, val_df], ignore_index=True) \
                    .sort_values(["group","time_idx"])

    
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
    
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_plus_hist,
        predict=True,
        stop_randomization=True,

    )
    
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, batch_sampler='synchronized'
    )
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0, batch_sampler='synchronized'
    )
    
        
    checkpoint_cb = ModelCheckpoint(
    dirpath=f"{OUT/UNIQUE_ID}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,              
    filename="deepar-{epoch:02d}-{val_loss:.3f}",
    )
    
    trainer = Trainer(accelerator='cpu', gradient_clip_val = GRADIENT_CLIP_VAL)
    
    net_lr = deepAR(
        training_dataset,
        learning_rate = 3e-2,
        hidden_size=30,
        rnn_layers=2,
        loss=NormalDistributionLoss(),
        optimizer="Adam"
    )
    
    res = Tuner(trainer).lr_find(
        net_lr,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        min_lr=1e-5,
        max_lr=1e0,
        early_stop_threshold=100,
    )
    
    trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator="cpu",
    logger=logger,
    enable_model_summary=True,
    gradient_clip_val=GRADIENT_CLIP_VAL,
    limit_train_batches=50,
    enable_checkpointing=True,
    callbacks=[checkpoint_cb]
    )
    
    
    net = deepAR(
        training_dataset,
        learning_rate=res.suggestion(),
        log_interval=10,
        log_val_interval=1,
        hidden_size=30,
        rnn_layers=2,
        optimizer="Adam",
        loss=NormalDistributionLoss(),
    )
    
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    best_model_path = checkpoint_cb.best_model_path
    print('BEST MODEL PATH', best_model_path)
    print(checkpoint_cb.best_model_score)
    best_model = deepAR.load_from_checkpoint(best_model_path)
    
    predictions = best_model.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    MAE()(predictions.output, predictions.y)
    
    print(f"The Mean Absolute Error of your predictions are: {predictions.output}.")
    
    pred = best_model.predict(
    val_dataloader,
    return_index=True,
    return_decoder_lengths=True,
    return_x=True,
    mode="prediction",
    trainer_kwargs=dict(accelerator="cpu"),
    )
    
    save_timeseries_prediction_to_json(pred, f"{OUT/UNIQUE_ID}")

if __name__ == "__main__":
    main()