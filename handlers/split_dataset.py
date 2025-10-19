import numpy as np
import pandas as pd

from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils import save_cli_args


UNIQUE_ID = datetime.now().strftime("%Y%m%d_%H%M")


def main():

    parser = make_parser()
    args = parser.parse_args()

    DF_PATH: Path = args.df

    TRAIN_SIZE = args.train_size
    VAL_SIZE = args.val_size
    TEST_SIZE = args.test_size

    if size:=(TRAIN_SIZE + VAL_SIZE + TEST_SIZE) != 1:
        raise AssertionError(f"The sum of the requested split proportions is {size}, but it must be equal to 1.")

    FIXED_SEED = args.fixed_seed
    DISABLE_NUMERICAL_ENCODING = args.disable_numerical_encoding
    SEED: int = 42 if FIXED_SEED else np.random.randint(low=0, high=1_000)

    OUT_DIR: Path = args.out
    split_dir = OUT_DIR / f"data_splits_{UNIQUE_ID}"

    train_dataset_path = split_dir / f"train_split.parquet"
    val_dataset_path = split_dir / f"val_split.parquet"
    test_dataset_path = split_dir / f"test_split.parquet"
    
    df = pd.read_parquet(DF_PATH)

    ### NOTE THAT THE LABELS FOR DEEPTUNE MUST BE NUMERICALLY ENCODED ###

    if not DISABLE_NUMERICAL_ENCODING:
        CLASS_NAMES = df['labels']
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        df['labels'] = le.transform(df['labels'])


    # df = df[:10]
    split_dir.mkdir(parents=True, exist_ok=True)

    train_data: DataFrame
    val_data: DataFrame
    test_data: DataFrame

    train_data, temp_data = train_test_split(df, train_size=TRAIN_SIZE, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), random_state=SEED)
    
    test_data["indices"] = test_data.index

    for part in (train_data,val_data,test_data):
        part.reset_index(drop=True, inplace=True)

    train_data.to_parquet(train_dataset_path, index=False)
    val_data.to_parquet(val_dataset_path, index=False)
    test_data.to_parquet(test_dataset_path, index=False)

    print(f"Dataset splits saved to {split_dir}")

    save_cli_args(args, split_dir)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Split parquet file into Train, Val, and Test splits. Allows the same splits to be used for training several deep learners. The splits maintain all features.", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--df",
        type=Path,
        required=True,
        help=""
    )
    parser.add_argument(
        '--train_size',
        type=float,
        required=True,
        help='Mention the split ratio of the Train Dataset'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        required=True,
        help='Mention the split ratio of the Val Dataset'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        required=True,
        help='Mention the split ratio of the Test Dataset'
    )
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="Use fixed seed for randomisation of data splits. If omitted, a random seed is used."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Location and name of the directory to save the three dataset splits."
    )
    parser.add_argument(
        "--disable-numerical-encoding",
        action="store_true",
        help="Disable the automatic numerical encoding of the labels' column."
    )

    return parser


if __name__ == "__main__":
    main()