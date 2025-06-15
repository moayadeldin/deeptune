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
from utilities import fixed_seed,get_args,split_save_load_dataset,save_cli_args,PerformanceLogger,PerformanceLoggerCallback

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' # disabling weights-only loading error

DEVICE = options.DEVICE
parser = options.parser
args = get_args()

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
FIXED_SEED = args.fixed_seed
TYPE = args.type


# GANDALF Specific Args
TARGET = args.tabular_target_column
CONTINUOUS_COLS = args.continuous_cols
CATEGORICAL_COLS = args.categorical_cols
GFLU_STAGES = args.gflu_stages

# load the dataset with appropriate paths
TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
VAL_DATASET_PATH = options.VAL_DATASET_PATH
TEST_DATASET_PATH = options.TEST_DATASET_PATH
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR

# If we want to apply fixed seed or randomly initialize the weights and dataset.
if FIXED_SEED:
    SEED=42
    fixed_seed(SEED)
else:
    SEED = np.random.randint(low=0, high=1000)
    fixed_seed(SEED)
    warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)


train_data, val_data, _ = split_save_load_dataset(
    
    mode='train',
    type='tabular',
    input_dir= INPUT_DIR,
    train_size = TRAIN_SIZE,
    val_size = VAL_SIZE,
    test_size = TEST_SIZE,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
    seed=SEED,
    batch_size=BATCH_SIZE,
    tokenizer=None
)    


performance_logger = PerformanceLogger(TRAINVAL_OUTPUT_DIR)


data_config = DataConfig(
    target=TARGET,
    continuous_cols=CONTINUOUS_COLS,
    categorical_cols=CATEGORICAL_COLS,
)

optimizer_config = OptimizerConfig()

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


if __name__ == "__main__":


    callback = PerformanceLoggerCallback(
        performance_logger=performance_logger,
    )    

    tabular_model.fit(train=train_data, validation=val_data,callbacks=[callback])
    tabular_model.save_model(TRAINVAL_OUTPUT_DIR/"GANDALF_model")
    print(f"Model saved to {TRAINVAL_OUTPUT_DIR}")
    save_cli_args(args, TRAINVAL_OUTPUT_DIR, mode='tabular train')
    performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")