import torch.nn as nn
import options
import pandas as pd
from torch.utils.data import DataLoader
from datasets.tabular_datasets import TabularDataset
from tqdm import tqdm
import numpy as np
from pytorch_tabular import TabularModel
import torch
import time

from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE
from utils import RunType, save_process_times
import os

def embed(
        eval_df: Path,
        out: Path,
        model_weights: Path,
        cont_cols: str,
        cat_cols: str,
        batch_size: int,
        target:str,
        args: DeepTuneVisionOptions,
        grouper,
        model_str='GANDALF',
        device=options.DEVICE

):
    
    continuous_cols = cont_cols or []
    categorical_cols = cat_cols or []
    
    EMBED_OUTPUT = (out / f"embed_output_{model_str}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{model_str}_embeddings.parquet"

    df = pd.read_parquet(eval_df)


    if df.empty:
        raise ValueError('The embedding dataset is empty.')
    model = TabularModel.load_model(os.path.join(model_weights, 'GANDALF_model'))
    model.model = model.model.to(device)
    model.model.eval()
    extracted_embeddings = []
    start_time = time.time()

    def capture(module, inputs, output):
        extracted_embeddings.append(output.detach().cpu().numpy())

    # predict() applies the fitted training encoders and normalization, keeping
    # category IDs and continuous scales identical to training.
    handle = model.model.backbone.register_forward_hook(capture)
    try:
        model.predict(df, progress_bar=None)
    finally:
        handle.remove()
    embeddings = np.vstack(extracted_embeddings)
    if len(embeddings) != len(df):
        raise RuntimeError('Embedding output did not preserve the input row count.')
    combined_df = pd.DataFrame(embeddings)
    combined_df['label'] = df[target].to_numpy()

    if grouper is not None and grouper in df.columns:
        combined_df[grouper] = df[grouper]

    combined_df.to_parquet(EMBED_FILE,index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    args.save_args(EMBED_OUTPUT)
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")
    print(f'The embeddings file is saved in {EMBED_OUTPUT}')

    
    return EMBED_OUTPUT, combined_df.shape
    


def main():

    args = DeepTuneVisionOptions(RunType.GANDALF)

    TEST_PATH = args.df
    OUT = args.out
    MODEL_STR = 'GANDALF',
    args=args
    BATCH_SIZE = args.batch_size
    MODEL_WEIGHTS = args.model_weights
    CONTINUOUS_COLS = args.continuous_cols
    CATEGORICAL_COLS = args.categorical_cols
    DEVICE = options.DEVICE
    GROUPER = args.grouper
    TARGET_COLUMN = args.tabular_target_column

    embed(
        eval_df=TEST_PATH,
        out=OUT,
        model_weights=MODEL_WEIGHTS,
        args=args,
        cont_cols=CONTINUOUS_COLS,
        cat_cols=CATEGORICAL_COLS,
        batch_size=BATCH_SIZE,
        target=TARGET_COLUMN,
        model_str=MODEL_STR,
        device=DEVICE,
        grouper=GROUPER
    )

if __name__ == "__main__":
    
    main()