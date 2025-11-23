import time
from src.tabular.gandalf import GANDALF
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import TabularModel
import numpy as np
import warnings
import options
import os
from helpers import PerformanceLogger,PerformanceLoggerCallback
from pathlib import Path
from utils import save_process_times
from options import UNIQUE_ID
from cli import DeepTuneVisionOptions
from utils import RunType,set_seed
import pandas as pd


os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' # disabling weights-only loading error


def main():


    args = DeepTuneVisionOptions(RunType.GANDALF)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    type = args.type
    OUT = args.out
    MODEL_STR = 'GANDALF'
    BATCH_SIZE=args.batch_size
    LEARNING_RATE=args.learning_rate
    NUM_EPOCHS = args.num_epochs


    # GANDALF Specific Args
    TARGET = args.tabular_target_column
    CONTINUOUS_COLS = args.continuous_cols
    CATEGORICAL_COLS = args.categorical_cols
    GFLU_STAGES = args.gflu_stages

    train(
        train_df=TRAIN_PATH,
        val_df=VAL_PATH,
        type=type,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        gflu_stages=GFLU_STAGES,
        target=TARGET,
        continuous_cols=CONTINUOUS_COLS,
        categorical_cols=CATEGORICAL_COLS,
        out=OUT,
        args=args,
        model_str=MODEL_STR
    )


def train(
        train_df: Path,
        val_df: Path,
        out: Path,
        type:str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        gflu_stages:int,
        target:str,
        continuous_cols: str,
        categorical_cols: str,
        model_str: str,
        args: DeepTuneVisionOptions,

):

    train_dataset = pd.read_parquet(train_df)
    val_dataset = pd.read_parquet(val_df)

    TRAINVAL_OUTPUT_DIR = (out / f"trainval_output_{model_str}_{UNIQUE_ID}") 

    performance_logger = PerformanceLogger(TRAINVAL_OUTPUT_DIR)

    categorical_cols = categorical_cols or []
    continuous_cols = continuous_cols or []

    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )

    optimizer_config = OptimizerConfig()

    start_time = time.time()
    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=num_epochs,
        progress_bar='None'
    )

    model_config = GANDALF(
        data_config=data_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        task=type,
        gflu_stages=gflu_stages,
        learning_rate=learning_rate,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        verbose=True,
    )

    callback = PerformanceLoggerCallback(
        performance_logger=performance_logger,
    )    
    
    tabular_model.fit(train=train_dataset, validation=val_dataset,callbacks=[callback])
    end_time = time.time()
    total_time = end_time - start_time
    tabular_model.save_model(TRAINVAL_OUTPUT_DIR/"GANDALF_model")
    print(f"Model saved to {TRAINVAL_OUTPUT_DIR}")
    performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
    args.save_args(TRAINVAL_OUTPUT_DIR)
    save_process_times(epoch_times="For GANDALF we only track total time", total_duration=total_time, outdir=TRAINVAL_OUTPUT_DIR, process="training")

    return TRAINVAL_OUTPUT_DIR

if __name__ == "__main__":


    main()