from datasets.timeseries_data import prepare_timeseries
from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting
import pandas as pd
from src.timeseries.deepAR import deepAR
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
from helpers import save_timeseries_prediction_to_json
from pytorch_forecasting.metrics import MAE
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch import Trainer
import time 
from pathlib import Path
from utils import save_process_times

class _LossHistory(Callback):
    def __init__(self, destination):
        self.destination = destination
        self.rows = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        def number(*keys):
            value = next((metrics[key] for key in keys if key in metrics), None)
            return float(value.detach().cpu()) if value is not None else None
        self.rows.append({'epoch': trainer.current_epoch + 1,
                          'epoch_loss': number('train_loss_epoch', 'train_loss'),
                          'val_loss': number('val_loss')})
        self.destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.destination / 'training_log.csv', index=False)


def train(
        train_df: Path,
        val_df: Path,
        num_epochs: int,
        out: Path,
        batch_size: int,
        timeindex_column: str,
        target_column: str,
        args: DeepTuneVisionOptions,
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        time_varying_known_categoricals: list = [],
        time_varying_unknown_categoricals: list = [],
        static_categoricals: list = [],
        time_varying_known_reals: list = [],
        group_ids = None,
        time_varying_unknown_reals: list = [],
        static_reals: list = None,
        model_str='DeepAR',
        learning_rate: float = 1e-3,

):
    
    train_df = pd.read_parquet(train_df)
    val_df = pd.read_parquet(val_df)
    
    
    train_df["__split"] = "train"
    val_df["__split"] = "val"
    
    df = pd.concat([train_df, val_df],ignore_index=True)
    
    TRAINVAL_OUTPUT_DIR = (out / f"trainval_{model_str}_output_{UNIQUE_ID}")
    
    logger = TensorBoardLogger(
    save_dir=out,
    name=UNIQUE_ID,
    version=""
)

    
    """
    We assume we work with a single time series.
    """
    
    df, GROUP_IDS = prepare_timeseries(df, timeindex_column, target_column, group_ids)

    train_df = df[df["__split"] == "train"].copy()
    val_df = df[df["__split"] == "val"].copy()
    
    GRADIENT_CLIP_VAL = 1e-1
        
    # IMPORTANT: TimeSeries models needs the last "max_encoder_length" timesteps to predict the next "max_prediction_length". Hence, we must get these timesteps from the training set to initialize the encoder in validation
    hist = train_df.sort_values([*GROUP_IDS,"time_idx"]) \
                .groupby(GROUP_IDS, as_index=False) \
                .tail(max_encoder_length)

    val_plus_hist = pd.concat([hist, val_df], ignore_index=True) \
                    .sort_values([*GROUP_IDS,"time_idx"])

    
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx = "time_idx",
        target=target_column,
        max_prediction_length = max_prediction_length,
        max_encoder_length = max_encoder_length,
        time_varying_known_categoricals = time_varying_known_categoricals,
        time_varying_unknown_categoricals = time_varying_unknown_categoricals,
        static_categoricals = static_categoricals,
        time_varying_known_reals = time_varying_known_reals,
        time_varying_unknown_reals=(time_varying_unknown_reals) + [target_column],
        static_reals = static_reals,
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
        train=True, batch_size=batch_size, num_workers=0, batch_sampler='synchronized'
    )
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler='synchronized'
    )
    
        
    checkpoint_cb = ModelCheckpoint(
    dirpath=TRAINVAL_OUTPUT_DIR,
    monitor="val_loss",
    mode="min",
    save_top_k=1,              
    filename="deepar-{epoch:02d}-{val_loss:.3f}",
    )
    
    total_time = 0
    start_time = time.time()
    
    trainer = Trainer(
    max_epochs=num_epochs,
    accelerator="cpu",
    logger=logger,
    enable_model_summary=True,
    gradient_clip_val=GRADIENT_CLIP_VAL,
    enable_checkpointing=True,
    callbacks=[checkpoint_cb, _LossHistory(TRAINVAL_OUTPUT_DIR)]
    )
    
    
    net = deepAR(
        training_dataset,
        learning_rate=learning_rate,
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
    
    pred = best_model.predict(
    val_dataloader,
    return_index=True,
    return_decoder_lengths=True,
    return_x=True,
    mode="prediction",
    trainer_kwargs=dict(accelerator="cpu"),
    )
    
    print(f"Validation predictions: {tuple(pred.output.shape)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    save_timeseries_prediction_to_json(pred, TRAINVAL_OUTPUT_DIR)
    args.save_args(TRAINVAL_OUTPUT_DIR)
    save_process_times(epoch_times=num_epochs, total_duration=total_time, outdir=TRAINVAL_OUTPUT_DIR, process="training")

    return TRAINVAL_OUTPUT_DIR

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
    GROUP_IDS = args.group_ids

    train(
        train_df=TRAIN_DF_PATH,
        val_df=VAL_DF_PATH,
        num_epochs=NUM_EPOCHS,
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
        args=args,
        group_ids=GROUP_IDS,
        model_str='DeepAR',
    )
    

    
if __name__ == "__main__":
    main()