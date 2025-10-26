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
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls,RunType,set_seed,save_process_times
from datasets.text_datasets import TextDataset
def main():

    args = DeepTuneVisionOptions(RunType.GANDALF)

    TEST_PATH = args.df
    OUT = args.out
    MODEL_STR = 'GANDALF'
    BATCH_SIZE = args.batch_size
    MODEL_WEIGHTS = args.model_weights
    CONTINUOUS_COLS = args.continuous_cols
    CATEGORICAL_COLS = args.categorical_cols
    DEVICE = options.DEVICE
    TARGET_COLUMN = args.tabular_target_column
    TEST_OUTPUT_DIR = (OUT / f"test_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/test_output_{MODEL_STR}_{UNIQUE_ID}")

    df = pd.read_parquet(TEST_PATH)

    dataset = TabularDataset(df, CONTINUOUS_COLS, CATEGORICAL_COLS, label_col=TARGET_COLUMN)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TabularModel.load_model(MODEL_WEIGHTS)
    model.model = model.model.to(DEVICE)
    extracted_embeddings = []
    extracted_labels = []
    
    start_time = time.time()

    for (x_cont, x_cat), labels in tqdm(data_loader):
        new_labels = labels.numpy().tolist()
        extracted_labels += new_labels

        x_cont = x_cont.to(DEVICE)
        x_cat = x_cat.to(DEVICE)

        with torch.no_grad():
            
            embedded_input = model.model.embedding_layer({"continuous": x_cont, "categorical": x_cat})

            embeddings = model.model.backbone(embedded_input)

        extracted_embeddings.append(embeddings.detach().cpu().numpy())

    extracted_embeddings = np.vstack(extracted_embeddings)

    embeddings_df = pd.DataFrame(embeddings.cpu().numpy())
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet(TEST_OUTPUT_DIR,index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="embedding")

if __name__ == "__main__":
    
    main()