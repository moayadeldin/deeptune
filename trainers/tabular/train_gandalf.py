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
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from embed.vision.custom_embed_siglip_handler import embed_with_siglip
from cli import DeepTuneVisionOptions
from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP, RunType,set_seed
import pandas as pd


os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' # disabling weights-only loading error


def main():


    args = DeepTuneVisionOptions(RunType.GANDALF)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    DATA_DIR: Path = args.input_dir
    TYPE = args.type
    OUT = args.out
    MODEL_STR = 'GANDALF'
    DATA_DIR = args.input_dir
    BATCH_SIZE=args.batch_size
    LEARNING_RATE=args.learning_rate
    NUM_EPOCHS = args.num_epochs
    FIXED_SEED = args.fixed_seed


    # GANDALF Specific Args
    TARGET = args.tabular_target_column
    CONTINUOUS_COLS = args.continuous_cols
    CATEGORICAL_COLS = args.categorical_cols
    GFLU_STAGES = args.gflu_stages

    

    
    if FIXED_SEED:
        set_seed(FIXED_SEED)

    
    TRAIN_DATASET_PATH = TRAIN_PATH or ( DATA_DIR / "train_split.parquet" )
    VAL_DATASET_PATH = VAL_PATH or ( DATA_DIR / "val_split.parquet" )

    train_dataset = pd.read_parquet(TRAIN_DATASET_PATH)
    val_dataset = pd.read_parquet(VAL_DATASET_PATH)

    # We do some preprocessing for both datasets filling the missing numeric columns with zero and fill missing columns with place holder of unknown.

    num_cols_train = train_dataset.select_dtypes(include=["number"]).columns
    train_dataset[num_cols_train] = train_dataset[num_cols_train].fillna(0)

    num_cols_val = val_dataset.select_dtypes(include=["number"]).columns
    val_dataset[num_cols_val] = val_dataset[num_cols_val].fillna(0)

    train_str_cols = train_dataset.select_dtypes(include=["object", "category"]).columns
    train_dataset[train_str_cols] = train_dataset[train_str_cols].fillna("unknown")

    TRAINVAL_OUTPUT_DIR = (OUT / f"trainval_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/trainval_output_{MODEL_STR}_{TYPE}_{UNIQUE_ID}")

    performance_logger = PerformanceLogger(TRAINVAL_OUTPUT_DIR)


    data_config = DataConfig(
        target=TARGET,
        continuous_cols=CONTINUOUS_COLS,
        categorical_cols=CATEGORICAL_COLS,
    )

    optimizer_config = OptimizerConfig()

    start_time = time.time()
    trainer_config = TrainerConfig(
        batch_size=BATCH_SIZE,
        max_epochs=NUM_EPOCHS,
    )

    model_config = GANDALF(
        data_config=data_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        task=TYPE,
        gflu_stages=GFLU_STAGES,
        learning_rate=LEARNING_RATE,
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
    save_process_times(epoch_times="For GANDALF we only track total time", total_duration=total_time, outdir=TRAINVAL_OUTPUT_DIR, process="training")
    tabular_model.save_model(TRAINVAL_OUTPUT_DIR/"GANDALF_model")
    print(f"Model saved to {TRAINVAL_OUTPUT_DIR}")
    performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
    args.save_args(TRAINVAL_OUTPUT_DIR)


if __name__ == "__main__":


    main()