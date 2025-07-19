import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from PIL import Image
import io
import torch

class ParquetImageDataset(Dataset):
    """
    Loads the images and labels from parquet file we feed to the model.
    
    Attributes:
    
        data (pd.DataFrame): The DataFrame containing image byte data and labels.
        transform: Transformations we apply to the image datasets (we use the same from ImageNet in utilities.py)
        processor (torchvision.transforms): Processor function needed for HuggingFace Siglip
        image_bytes (np.ndarray): The images in bytes format
        labels (np.ndarray): The labels of corresponding image.
        
    """
    def __init__(self, df: DataFrame, transform=None, processor=None):
        self.data = df
        self.transform = transform
        self.processor = processor
        self.image_bytes = self.data['images'].values
        self.has_labels = 'labels' in df.columns
        self.labels = self.data['labels'].values if self.has_labels else None
    
    @classmethod 
    def from_parquet(cls, parquet_file, transform=None, processor=None) -> "ParquetImageDataset":
        df = pd.read_parquet(parquet_file)
        return cls(df, transform, processor)

    def __len__(self):
        
        """
        Returns the number of samples in dataset.
        """
        return len(self.image_bytes)

    def __getitem__(self, idx):
        """
        Loads and processes the image and label at a given index idx.
        """
        img_bytes = self.image_bytes[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long) if self.has_labels else None

        if self.processor is not None:
            processed = self.processor(
                images=img,
                return_tensors="pt",
                # padding=True
            )
            # Remove the extra batch dimension added by the processor
            pixel_values = processed["pixel_values"].squeeze(0)  # Important: squeeze here
            return pixel_values, label
        else:
            if self.transform:
                img = self.transform(img)
            return img, label
        
    
    
