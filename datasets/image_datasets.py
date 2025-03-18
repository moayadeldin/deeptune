import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
import torch

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_file, transform=None, processor=None):
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform
        self.processor = processor
        self.image_bytes = self.data['images'].values
        self.labels = self.data['labels'].values

    def __len__(self):
        return len(self.image_bytes)

    def __getitem__(self, idx):
        img_bytes = self.image_bytes[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[idx]

        if self.processor is not None:
            processed = self.processor(
                images=img,
                return_tensors="pt",
                padding=True
            )
            # Remove the extra batch dimension added by the processor
            pixel_values = processed["pixel_values"].squeeze(0)  # Important: squeeze here
            return pixel_values, torch.tensor(label, dtype=torch.long)
        else:
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        
    
    
