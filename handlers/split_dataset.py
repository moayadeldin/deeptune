import numpy as np
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import warnings
from options import UNIQUE_ID
import json
from sklearn.model_selection import StratifiedGroupKFold
from handlers.split_planning import allocate_split_counts, classification_split_issues


def main():

    parser = make_parser()
    args = parser.parse_args()

    DF_PATH: Path = args.df

    TRAIN_SIZE = args.train_size
    VAL_SIZE = args.val_size
    TEST_SIZE = args.test_size

    if not np.isclose(size := TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1):
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
        mode=args.mode,
        time_idx_column=args.time_idx_column,
    )

    print(f"Dataset splits saved to:\n Train:{train_path}\n Validation: {val_path}\n Test: {test_path}")

def split_dataset(train_size: float, val_size: float, test_size:float, df_path: Path, out_dir: Path, fixed_seed: bool, disable_numerical_encoding: bool, target_column:str,modality:str, disable_target_column_renaming: bool=False, grouper: str=None, mode: str='cls', time_idx_column: str=None):
    proportions = np.asarray([train_size, val_size, test_size], dtype=float)
    if not np.isfinite(proportions).all() or (proportions <= 0).any() or not np.isclose(proportions.sum(), 1):
        raise ValueError('Train, validation, and test proportions must be positive and sum to 1.')
    if mode not in ('cls', 'reg'):
        raise ValueError('Mode must be cls or reg.')
    out_dir = Path(out_dir)
    split_dir = out_dir / f"data_splits_{UNIQUE_ID}"
    train_dataset_path = split_dir / f"train_split.parquet"
    val_dataset_path = split_dir / f"val_split.parquet"
    test_dataset_path = split_dir / f"test_split.parquet"
    
    df = pd.read_parquet(df_path)
    target_column = target_column or 'labels'
    if target_column not in df.columns:
        raise ValueError(f'Target column {target_column!r} was not found in the dataset.')
    if target_column != 'labels' and 'labels' in df.columns and not disable_target_column_renaming:
        raise ValueError('The dataset already contains labels; choose labels as the target or rename that column.')
    if len(df) < 3 or df[target_column].isna().any():
        raise ValueError('The dataset needs at least three rows and no missing target values.')
    if grouper and (grouper not in df.columns or df[grouper].isna().any()):
        raise ValueError(f'Grouper column {grouper!r} must exist and contain no missing values.')
    if grouper == target_column:
        raise ValueError('The grouper must be different from the target column.')
    classification = mode == 'cls' and modality != 'timeseries'
    if classification:
        issues = classification_split_issues(df[target_column].value_counts().to_dict(),
                                             unit='clips' if modality == 'video' else 'samples')
        if issues:
            raise ValueError('\n'.join(issues))
    if not classification:
        values = pd.to_numeric(df[target_column], errors='coerce')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError('Regression and forecasting targets must be finite numeric values.')
        df[target_column] = values
    split_dir.mkdir(parents=True, exist_ok=True)

    df = enable_label_numerical_encoding(
        df=df,
        split_dir=split_dir,
        target_column=target_column if target_column is not None else 'labels',
        disable_numerical_encoding=disable_numerical_encoding or not classification,
        modality=modality,
        disable_target_column_renaming=disable_target_column_renaming)
    
    if fixed_seed:
        SEED: int = 42
    else:
        SEED = np.random.randint(low=0, high=1_000)

        warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits. You are safe to neglect this warning if you are using Deeptune for purposes other than training or generating data splits', category=UserWarning)
        warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

    label_column = target_column if disable_target_column_renaming else 'labels'
    if modality == 'timeseries':
        if not time_idx_column or time_idx_column == target_column or time_idx_column not in df.columns:
            raise ValueError('Forecasting requires a separate time index column present in the dataset.')
        if df[time_idx_column].isna().any():
            raise ValueError('The time index cannot contain missing values.')
        times = sorted(df[time_idx_column].unique())
        first, second = round(len(times) * train_size), round(len(times) * (train_size + val_size))
        if not 0 < first < second < len(times):
            raise ValueError('Not enough distinct time steps for the requested chronological splits.')
        ordered = df.sort_values(time_idx_column, kind='stable')
        train_data = ordered[ordered[time_idx_column] < times[first]]
        val_data = ordered[(ordered[time_idx_column] >= times[first]) & (ordered[time_idx_column] < times[second])]
        test_data = ordered[ordered[time_idx_column] >= times[second]]
        pd.DataFrame({'Test Set Indices in the Original Dataframe': test_data.index}).to_csv(split_dir / 'test_indices.csv', index=False)
        for part, destination in zip((train_data, val_data, test_data), (train_dataset_path, val_dataset_path, test_dataset_path)):
            part.reset_index(drop=True).to_parquet(destination, index=False)
        return train_dataset_path, val_dataset_path, test_dataset_path

    if grouper:
        if target_column is None:
            raise ValueError("When using a grouper for stratified splitting, the target column must be specified.")
        
        # Group the train+val and test splits using StratifiedGroupKFold
        n_test = int(round(1 / test_size))
        n_test = max(2, min(n_test, df[grouper].nunique()))
        if df[grouper].nunique() < 3:
            raise ValueError('Group splitting needs at least three distinct groups.')
        sgkf_test = (StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=SEED)
                     if classification else GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED))


        y = df[label_column].to_numpy()

        trainval_idx, test_idx = next(sgkf_test.split(df,y, groups=df[grouper]))

        df_trainval,y_trainval, groups_trainval = df.iloc[trainval_idx], y[trainval_idx], df.iloc[trainval_idx][grouper]
        X_test = df.iloc[test_idx]
        
        # Now split train+val into train and val using StratifiedGroupKFold

        val_frac_of_trainval = val_size / (train_size + val_size)
        n_val = int(round(1 / val_frac_of_trainval))

        n_val = max(2, min(n_val, groups_trainval.nunique()))
        if groups_trainval.nunique() < 2:
            raise ValueError('Not enough training groups remain for a separate validation set.')
        sgkf_val = (StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED)
                    if classification else GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=SEED))
        train_idx, val_idx = next(sgkf_val.split(df_trainval,y_trainval, groups=groups_trainval))

        X_train = df_trainval.iloc[train_idx]
        X_val = df_trainval.iloc[val_idx]

        train_data = X_train.reset_index(drop=True)
        val_data = X_val.reset_index(drop=True)
        test_data = X_test.reset_index(drop=True)

        df_test_indices = pd.DataFrame({'Test Set Indices in the Original Dataframe': X_test.index})
    
        df_test_indices.to_csv(f'{split_dir}/test_indices.csv')
        
        train_data.to_parquet(train_dataset_path, index=False)
        val_data.to_parquet(val_dataset_path, index=False)
        test_data.to_parquet(test_dataset_path, index=False)

        return train_dataset_path, val_dataset_path, test_dataset_path

    # for convenience and as part of the preprocessing, deeptune will rename the target column of prediction to labels

    if classification:
        # Allocate within each class so rounding never removes a class from
        # validation/test, even with only three clips per class.
        rng = np.random.RandomState(SEED)
        partitions = [[], [], []]
        for positions in df.groupby(label_column, sort=True).indices.values():
            positions = rng.permutation(positions)
            counts = allocate_split_counts(len(positions), proportions)
            first, second = counts[0], counts[0] + counts[1]
            for destination, subset in zip(partitions, (positions[:first], positions[first:second], positions[second:])):
                destination.extend(subset)
        train_data, val_data, test_data = [df.iloc[rng.permutation(indices)].copy() for indices in partitions]
    else:
        train_count, val_count, test_count = allocate_split_counts(len(df), proportions)
        train_data, temp_data = train_test_split(df, train_size=train_count, random_state=SEED)
        val_data, test_data = train_test_split(temp_data, train_size=val_count, test_size=test_count, random_state=SEED)
    actual = dict(zip(('train', 'validation', 'test'), map(len, (train_data, val_data, test_data))))
    (split_dir / 'split_summary.json').write_text(json.dumps({
        'requested_ratios': dict(zip(actual, proportions.tolist())), 'sample_counts': actual,
        'strategy': 'per_class' if classification else 'random',
    }, indent=2), encoding='utf-8')
    print('Dataset split: ' + ', '.join(f'{name}={count}' for name, count in actual.items()) +
          '. Small datasets use adjusted proportions to keep every split populated.')
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

    label_column = target_column if disable_target_column_renaming else 'labels'
    if not disable_numerical_encoding and modality != 'timeseries' and label_column in df.columns:
        CLASS_NAMES = df[label_column]
        le = LabelEncoder()
        le.fit(CLASS_NAMES)
        df[label_column] = le.transform(df[label_column])

        if modality != 'timeseries':
            class_to_int = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}

            with open(split_dir / 'label_mapping.json', 'w') as f:

                json.dump(class_to_int, f, indent=4)

    return df


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Split parquet file into Train, Val, and Test splits. Allows the same splits to be used for training several deep learners. The splits maintain all features.", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mode', choices=['cls', 'reg'], default='cls', help='Preserve numeric targets with reg.')
    parser.add_argument('--time_idx_column', help='Time index for chronological forecasting splits.')

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
        choices=['timeseries','tabular','text','images','video'],
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
