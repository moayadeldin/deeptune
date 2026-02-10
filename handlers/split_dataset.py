import numpy as np
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from options import UNIQUE_ID
import json
from sklearn.model_selection import StratifiedGroupKFold


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
    DISABLE_TARGET_COLUMN_RENAMING = args.disable_target_column_renaming
    TARGET_COLUMN = args.target
    MODALITY = args.modality
    GROUPER = args.grouper
        
    OUT_DIR: Path = args.out

    train_path, val_path, test_path = split_dataset(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        df_path=DF_PATH,
        out_dir=OUT_DIR,
        fixed_seed=FIXED_SEED,
        disable_numerical_encoding=DISABLE_NUMERICAL_ENCODING,
        target_column=TARGET_COLUMN,
        disable_target_column_renaming=DISABLE_TARGET_COLUMN_RENAMING,
        modality=MODALITY,
        grouper=GROUPER,
    )

    print(f"Dataset splits saved to:\n Train:{train_path}\n Validation: {val_path}\n Test: {test_path}")

def split_dataset(train_size: float, val_size: float, test_size:float, df_path: Path, out_dir: Path, fixed_seed: bool, disable_numerical_encoding: bool, target_column:str,modality:str, disable_target_column_renaming: bool=False, grouper: str=None):
    split_dir = out_dir / f"data_splits_{UNIQUE_ID}"
    train_dataset_path = split_dir / f"train_split.parquet"
    val_dataset_path = split_dir / f"val_split.parquet"
    test_dataset_path = split_dir / f"test_split.parquet"
    
    split_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(df_path)

    df = enable_label_numerical_encoding(
        df=df,
        split_dir=split_dir,
        target_column=target_column if target_column is not None else 'labels',
        disable_numerical_encoding=disable_numerical_encoding,
        modality=modality,
        disable_target_column_renaming=disable_target_column_renaming)


    if fixed_seed:
        SEED: int = 42
    else:
        SEED = np.random.randint(low=0, high=1_000)

        warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits. You are safe to neglect this warning if you are using Deeptune for purposes other than training or generating data splits', category=UserWarning)
        warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

    if grouper is not None:
        if target_column is None:
            raise ValueError("When using a grouper for stratified splitting, the target column must be specified.")
        
        # Group the train+val and test splits using StratifiedGroupKFold
        n_test = int(round(1 / test_size))
        sgkf_test = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=SEED)


        y=df[target_column].to_numpy() if target_column in df.columns else df['labels'].to_numpy()

        trainval_idx, test_idx = next(sgkf_test.split(df,y, groups=df[grouper]))

        df_trainval,y_trainval, groups_trainval = df.iloc[trainval_idx], y[trainval_idx], df.iloc[trainval_idx][grouper]
        X_test = df.iloc[test_idx]
        
        # Now split train+val into train and val using StratifiedGroupKFold

        val_frac_of_trainval = val_size / (train_size + val_size)
        n_val = int(round(1 / val_frac_of_trainval))

        sgkf_val =StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED)
        train_idx, val_idx = next(sgkf_val.split(df_trainval,y_trainval, groups=groups_trainval))

        X_train = df_trainval.iloc[train_idx]
        X_val = df_trainval.iloc[val_idx]

        train_data = X_train.reset_index(drop=True)
        val_data = X_val.reset_index(drop=True)
        test_data = X_test.reset_index(drop=True)

        df_test_indices = pd.DataFrame({'Test Set Indices in the Original Dataframe': test_data.index})
    
        df_test_indices.to_csv(f'{split_dir}/test_indices.csv')
        
        train_data.to_parquet(train_dataset_path, index=False)
        val_data.to_parquet(val_dataset_path, index=False)
        test_data.to_parquet(test_dataset_path, index=False)

        return train_dataset_path, val_dataset_path, test_dataset_path

    # for convenience and as part of the preprocessing, deeptune will rename the target column of prediction to labels

    train_data, temp_data = train_test_split(df, train_size=train_size, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=SEED)
    df_test_indices = pd.DataFrame({'Test Set Indices in the Original Dataframe': test_data.index})
    
    df_test_indices.to_csv(f'{split_dir}/test_indices.csv')

    for part in (train_data,val_data,test_data):
        part.reset_index(drop=True, inplace=True)

    train_data.to_parquet(train_dataset_path, index=False)
    val_data.to_parquet(val_dataset_path, index=False)
    test_data.to_parquet(test_dataset_path, index=False)

    return train_dataset_path, val_dataset_path, test_dataset_path


def enable_label_numerical_encoding(
        df: DataFrame,
        split_dir:Path,
        target_column: str,
        disable_numerical_encoding: bool,
        modality: str,
        disable_target_column_renaming: bool):
    

    if not disable_target_column_renaming:

        df = df.rename(columns={target_column:'labels'})

    ### NOTE THAT THE LABELS FOR DEEPTUNE MUST BE NUMERICALLY ENCODED ###

    if not disable_numerical_encoding and 'labels' in df.columns:
        CLASS_NAMES = df['labels']
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        df['labels'] = le.transform(df['labels'])

        if modality != 'timeseries':
            class_to_int = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}

            with open(split_dir / 'label_mapping.json', 'w') as f:

                json.dump(class_to_int, f, indent=4)

    return df


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

    parser.add_argument(
        "--target",
        type=str,
        required=False,
        help="Specify the name of your target column. Default is 'labels'.",
    )

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=['timeseries','tabular','text','images'],
        help="Specify the modality of your dataset. This is used to determine whether label encoding mapping files are needed."
    )

    parser.add_argument(
        "--disable-target-column-renaming",
        action="store_true",
        help="Disable the automatic renaming of the target column to 'labels'. By default, Deeptune renames the target column to 'labels' for consistency across modalities."
    )

    parser.add_argument(
        "--grouper",
        type=str,
        required=False,
        help="Specify the name of the column to be used as grouper for StratifiedGroupKFold splitting.",
    )
    

    return parser


if __name__ == "__main__":
    main()
