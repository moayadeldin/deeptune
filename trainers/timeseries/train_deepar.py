from pytorch_forecasting import TimeSeriesDataset
import pandas as pd

from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID

def main():
    
    args = DeepTuneVisionOptions(RunType.OTHER)
    INPUT_PATH = args.input_dir
    OUT = args.out
    TIMEINDEX_COLUMN = args.time_idx_column
    
    df = pd.read_parquet(INPUT_PATH)

    training_dataset = TimeSeriesDataset(
        df[lambda x: x.time_idx.astype(int)],
        time_idx = TIMEINDEX_COLUMN,
        
    )