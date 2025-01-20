import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torch
import torchvision

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_file, transform=None, processor=None):
        """
        Args:
            parquet_file (string): Path to the parquet file containing image bytes and labels.
            transform (callable, optional): Transformations to be applied.
            processor (callable, optional): SIGLIP processor for image processing.
        """
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform
        self.processor = processor
        self.image_bytes = self.data['images'].tolist()
        self.labels = self.data['labels'].tolist()
        
    def __len__(self):
        return len(self.image_bytes)
    
    def __getitem__(self, idx):
        img_bytes = self.image_bytes[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[idx]

        if self.processor is not None:
            # SIGLIP processing path
            processed = self.processor(
                images=img,
                return_tensors="pt",
                padding=True
            )
            # Extract pixel_values and remove batch dimension
            pixel_values = processed["pixel_values"].squeeze(0)
            # print('HI THEREEEEEEEE', pixel_values)
            return pixel_values, torch.tensor(label, dtype=torch.long)
        else:
            # Standard processing path
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)