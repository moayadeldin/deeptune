import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torch


class TextDataset(Dataset):
    
    def __init__(self, parquet_file, tokenizer, max_length=512):
        
        self.data = pd.read_parquet(parquet_file)
        
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        
    def __len__(self):
        
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        
        return encoding, torch.tensor(label, dtype=torch.long)
    