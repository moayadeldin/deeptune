""""

A code snippet that is dedicated to get the 40% test set intersection between Deeptune's embeddings (as a Parquet file) and df-analyze (as a CSV file) splits.

"""

import pandas as pd 
import numpy as np
from cli import DeepTuneVisionOptions
from utils import RunType
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from options import UNIQUE_ID

def main():

    parser = make_parser()
    args = parser.parse_args()
    
    PATH_DF_PARQUET: Path = args.path_df_parquet
    PATH_DF_CSV: Path = args.path_df_csv
    OUT: Path = args.out
    
    df_parquet = pd.read_parquet(PATH_DF_PARQUET)
    df_csv = pd.read_csv(PATH_DF_CSV)
    
    common_cols = list(set(df_parquet) & set(df_csv.columns))
    
    df_parquet = df_parquet.astype(np.float32)
    df_csv = df_csv.astype(np.float32)
    
    result = df_parquet.merge(df_csv, on=common_cols, how='inner')
    
    print(f'There are {len(result)} records in intersection between both dataframes!')
    
    result.to_parquet(OUT / f"intersection_{UNIQUE_ID}.parquet") if OUT else Path(f"intersection_{UNIQUE_ID}.parquet")
    
    
def make_parser():
    
    parser = ArgumentParser(description='Obtain the intersection between a parquet file and csv file, mainly implemented to extract the 40 percent holdout dataset used by df-analyze from Deeptune embeddings representations fed.')
    
    parser.add_argument(
        '--df_parquet_path',
        type=Path,
        required=True,
        help=""
    )
    
    parser.add_argument(
        '--df_csv_path',
        type=Path,
        required=True,
        help=""
    )
    
    parser.add_argument('--out',
        type=Path,
        required=True,
        help="")


if __name__ == "__main__":

    main()