""""

A code snippet that is dedicated to get the 40% test set intersection between Deeptune's embeddings (as a Parquet file) and df-analyze (as a CSV file) splits.

"""

import pandas as pd 
import numpy as np
from cli import DeepTuneVisionOptions
from utils import RunType
from pathlib import Path
from options import UNIQUE_ID

def main():

    args = DeepTuneVisionOptions(RunType.OTHER)
    PATH_DF_PARQUET = args.df_parquet_path
    PATH_DF_CSV = args.df_csv_path
    OUT = args.out
    
    df_parquet = pd.read_parquet(PATH_DF_PARQUET)
    df_csv = pd.read_csv(PATH_DF_CSV)
    
    common_cols = list(set(df_parquet) & set(df_csv.columns))
    
    df_parquet = df_parquet.astype(np.float32)
    df_csv = df_csv.astype(np.float32)
    
    result = df_parquet.merge(df_csv, on=common_cols, how='inner')
    
    print(f'There are {len(result)} records in intersection between both dataframes!')
    
    result.to_parquet(OUT / f"intersection_{UNIQUE_ID}.parquet") if OUT else Path(f"intersection_{UNIQUE_ID}.parquet")
    
if __name__ == "__main__":

    main()