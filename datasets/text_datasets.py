import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torch


class TextDataset(Dataset):
  
    """
    
    Loads the texts and labels from parquet file we feed to the model.
    
    Attributes:
    
        data (pd.DataFrame): The DataFrame containing image byte data and labels.
        image_bytes (texts): The images in bytes format
        labels (np.ndarray): The labels of corresponding image.
        tokenizer (AutoTokenizer.from_pretrained): Tokenizer from HuggingFace for Multilingual BERT.
        max_length (int) : Maximum length of the input sequence.

    """
    
    def __init__(self, parquet_file, tokenizer, max_length=512):
        
        self.data = pd.read_parquet(parquet_file)
        
        self.texts = self.data['text'].tolist()
        self.labels = self.data['labels'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        
    def __len__(self):
        
        """
        Returns the number of samples in dataset.
        """
        
        return len(self.texts)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        
        """
        Loads and processes the image and label at a given index idx.
        """
        
        text = row['text']
        label = row['labels']
        extras = row.drop(labels=['text','labels'],errors='ignore').to_dict()
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if extras:
            return encoding, torch.tensor(label, dtype=torch.long), extras
        else:
            return encoding, torch.tensor(label, dtype=torch.long)
    