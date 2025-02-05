from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame
import torch
import pandas as pd
from torch.utils.data import DataLoader
from datasets.text_datasets import TextDataset
from utilities import avg_pool
from src.nlp.E5_roberta import load_nlp_intfloat_ml_model_offline

parser = ArgumentParser(description="Extract Embeddings using E5 Multilingual model based on Roberta")
parser.add_argument('--input_df_path', type=str, required=True, help='Dataset path of test set. Must be a .parquet file with only two columns, "image" and "target".')
parser.add_argument('--out', type=Path, required=True, help='Destination file name (.parquet).')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DF_PATH = args.input_df_path
OUTPUT = args.out

model, tokenizer = load_nlp_intfloat_ml_model_offline()

text_dataset = TextDataset(INPUT_DF_PATH, tokenizer=tokenizer)

# Create dataloader
loader = DataLoader(
    text_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


def run_embeddings():
    
    
    device = torch.device(DEVICE)
    model.to(device)
    df = get_nlp_embeddings(model, loader, device)
    
    df.to_parquet(OUTPUT)
    print(f"Saved text embeddings to {OUTPUT}.")
    

def get_nlp_embeddings(model, loader, device):
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        model.eval()
        
        print("Starting Text embedding..")
        print(f"Model architecture: {type(model).__name__}")
        print(f"Using device: {device}")
        
        for batch_dict, labels in tqdm(
            loader,
            total=len(loader),
            desc="Embedding Text"
        ):
            
            batch_dict = {key: val.to(device) for key, val in batch_dict.items()}
            labels = labels.to(device)
            
            outputs = model(**batch_dict)
            embeddings = avg_pool(outputs.last_hidden_state, attention_mask=batch_dict["attention_mask"])
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    
    
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed['target'] = labels.numpy()
    
    
    return df_embed
    
    
if __name__ == "__main__":
    
    run_embeddings()