from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, label_col=None):
        self.label_col = label_col

        if label_col in cont_cols:
            cont_cols = [col for col in cont_cols if col != label_col]
        if label_col in cat_cols:
            cat_cols = [col for col in cat_cols if col != label_col]

        self.cont_data = torch.tensor(df[cont_cols].values, dtype=torch.float32)

        self.encoder = OrdinalEncoder(dtype=np.int64)
        self.cat_data = torch.tensor(
            self.encoder.fit_transform(df[cat_cols]), dtype=torch.long
        )

        self.labels = (
            torch.tensor(df[label_col].values, dtype=torch.long)
            if label_col is not None
            else None
        )

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (self.cont_data[idx], self.cat_data[idx]), self.labels[idx]
        return self.cont_data[idx], self.cat_data[idx]
