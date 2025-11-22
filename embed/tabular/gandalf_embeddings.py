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
        model_str='GANDALF',
        device=options.DEVICE

):
    
    continuous_cols = cont_cols or []
    categorical_cols = cat_cols or []
    
    EMBED_OUTPUT = (out / f"embed_output_{model_str}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{model_str}_embeddings.parquet"

    df = pd.read_parquet(eval_df)


    dataset = TabularDataset(df, cont_cols=continuous_cols, cat_cols=categorical_cols, label_col=target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TabularModel.load_model(os.path.join(model_weights, 'GANDALF_model'))
    model.model = model.model.to(device)
    extracted_embeddings = []
    extracted_labels = []
    
    start_time = time.time()

    for (x_cont, x_cat), labels in tqdm(data_loader):
        new_labels = labels.numpy().tolist()
        extracted_labels += new_labels

        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)

        with torch.no_grad():
            
            embedded_input = model.model.embedding_layer({"continuous": x_cont, "categorical": x_cat})

            embeddings = model.model.backbone(embedded_input)

        extracted_embeddings.append(embeddings.detach().cpu().numpy())

    extracted_embeddings = np.vstack(extracted_embeddings)

    embeddings_df = pd.DataFrame(embeddings.cpu().numpy())
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet(EMBED_FILE,index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    args.save_args(EMBED_OUTPUT)
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")

    return out, combined_df.shape
    


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
        device=DEVICE
    )

if __name__ == "__main__":
    
    main()