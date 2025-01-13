import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torch
import torchvision
from utilities import transformations

class ParquetImageDataset(Dataset):

    def __init__(self, parquet_file, transform=None):
        """Args:
            parquet_file (string): Path to the parquet file containing image bytes and labels.
            transform (callable, optional): Transformations to be applied.
        """
        self.data = pd.read_parquet(parquet_file)

        if transform is None:
            self.transform = transformations
        else:
            self.transform = transform

        # Extract images (bytes) and labels from the parquet file
        self.image_bytes = self.data['images'].tolist()
        self.labels = self.data['labels'].tolist()

    def __len__(self):
        return len(self.image_bytes)

    def __getitem__(self, idx):
        # Convert image bytes back to PIL Image
        img_bytes = self.image_bytes[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label,dtype=torch.long)
