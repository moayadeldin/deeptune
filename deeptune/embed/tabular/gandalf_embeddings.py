import torch.nn as nn
import options
from deeptune.utilities import transformations,get_args
import pandas as pd
from torch.utils.data import DataLoader
from deeptune.datasets.tabular_datasets import TabularDataset
from tqdm import tqdm
import numpy as np
from pytorch_tabular import TabularModel
import torch
from deeptune.options import DEEPTUNE_RESULTS

DEVICE = options.DEVICE
parser = options.parser
args = get_args()

INPUT_DIR = args.input_dir
BATCH_SIZE = args.batch_size
MODEL_PATH = args.model_weights
CONTINUOUS_COLS = args.continuous_cols
CATEGORICAL_COLS = args.categorical_cols
DEVICE = options.DEVICE
TYPE = args.type
TARGET_COLUMN = args.tabular_target_column
TEST_OUTPUT = DEEPTUNE_RESULTS / f"test_set_gandalf_embeddings_{TYPE}.parquet"


df = pd.read_parquet(INPUT_DIR)
dataset = TabularDataset(df, CONTINUOUS_COLS, CATEGORICAL_COLS, label_col=TARGET_COLUMN)
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

model = TabularModel.load_model(MODEL_PATH)
model.model = model.model.to(DEVICE)


def extractEmbeddings():
    extracted_embeddings = []
    extracted_labels = []

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
    return extracted_embeddings, extracted_labels

if __name__ == "__main__":

    # run the function
    embeddings, labels = extractEmbeddings()

    print(f"The shape of the embeddings matrix in the dataset is {embeddings.shape}")
    
    # convert the embeddings to pd.DataFrame and merge the column of labels then return it

    embeddings_df = pd.DataFrame(embeddings)
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet(TEST_OUTPUT,index=False)