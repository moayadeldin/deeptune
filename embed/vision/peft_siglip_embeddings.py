from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from datasets.image_datasets import ParquetImageDataset
from src.vision.siglip_peft import load_peft_siglip_for_image_classification_offline

parser = ArgumentParser(description="Extract Embeddings used Peft model")
parser.add_argument('--input_df_path', type=str, required=True, help='Dataset path of test set. Must be a .parquet file with only two columns, "image" and "target".')
parser.add_argument('--out', type=Path, required=True, help='Destination file name (.parquet).')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DF_PATH = args.input_df_path
OUTPUT = args.out

def run_embeddings():
    model, processor = load_peft_siglip_for_image_classification_offline()
    image_dataset = ParquetImageDataset(INPUT_DF_PATH, processor=processor)
    
    # Create dataloader
    loader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    device = torch.device(DEVICE)
    df = get_vision_embeddings(model, loader, device)
    
    df.to_parquet(OUTPUT)
    print(f"Saved image embeddings to {OUTPUT}.")

def get_vision_embeddings(model, loader, device):
    model = model.to(device)
    model.eval()
    print("Starting image embedding...")
    print(f"Model architecture: {type(model).__name__}")
    print(f"Using device: {device}")
   
    all_embeddings = []
    all_labels = []
   
    with torch.no_grad():
        for pixel_values, label in tqdm(
            loader,
            total=len(loader),
            desc="Embedding images"
        ):
            pixel_values = pixel_values.to(device)
            
            output = model.vision_model(pixel_values=pixel_values)
            image_embeddings = output.pooler_output
            
            all_embeddings.append(image_embeddings.cpu())
            print(label)
            all_labels.append(label.item())
   
    embeddings = torch.cat(all_embeddings)
    
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed["target"] = all_labels
    df_embed.columns = df_embed.columns.astype(str)
   
    return df_embed

if __name__ == "__main__":
    run_embeddings()