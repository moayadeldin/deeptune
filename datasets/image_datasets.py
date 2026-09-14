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
        coords_col (str): Name of the column holding the [x,y] center coordinates.
        coords_normalized (bool) : Whether the coordinates are normalized to [0,1] or in pixel values. If false, they are treated as pixel values and will be normalized to [0,1] in the __getitem__ method.


        
    """
    def __init__(self, df: DataFrame, transform=None, processor=None,image_col="images", label_cols="labels", coords_col="coords", coords_normalized=False):
        self.data = df
        self.transform = transform
        self.processor = processor
        self.image_bytes = self.data['images'].values
        self.coords_col = 'coords' if 'coords' in df.columns else None
        self.coords_normalized = coords_normalized
        self.has_labels = 'labels' in df.columns
        self.labels = self.data['labels'].values if self.has_labels else None
    
    @classmethod 
    def from_parquet(cls, parquet_file, transform=None, processor=None, image_col="images", label_cols="labels", coords_col="coords",coords_normalized=False) -> "ParquetImageDataset":
        df = pd.read_parquet(parquet_file)
        return cls(df, transform, processor, image_col, label_cols, coords_col, coords_normalized)

    def __len__(self):
        
        """
        Returns the number of samples in dataset.
        """
        return len(self.image_bytes)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        """
        Loads and processes the image and label at a given index idx.
        """
        img_bytes = row['images']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = torch.tensor(row['labels'],dtype=torch.long) if self.has_labels else None

        orig_w, orig_h = img.size

        coords = None

        if self.coords_col and self.coords_col in row:
            coords = row[self.coords_col]
            if not self.coords_normalized:
                # Normalize the coordinates to [0,1] based on original image size
                coords = (coords[0] / orig_w, coords[1] / orig_h)
            coords = torch.tensor(coords, dtype=torch.float32)

        drop_cols = ['images', 'labels']

        if self.coords_col:
            drop_cols.append(self.coords_col)

        extras = row.drop(labels=drop_cols, errors='ignore').to_dict()

        # Helper to construct the output

        def build_output(image):

            output = [image]

            if coords is not None:
                output.append(coords)
            if label is not None:
                output.append(label)
            if extras:
                output.append(extras)
            return tuple(output)
            
        if self.processor is not None:
            processed = self.processor(
                images=img,
                return_tensors="pt",
                # padding=True
            )
            # Remove the extra batch dimension added by the processor
            pixel_values = processed["pixel_values"].squeeze(0)  # Important: squeeze here
            return build_output(pixel_values)
        else:
            if self.transform:
                img = self.transform(img)
            
            return build_output(img)
        
    
    
