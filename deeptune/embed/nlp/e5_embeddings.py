from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame
import torch
import pandas as pd
from torch.utils.data import DataLoader
from deeptune.datasets.text_datasets import TextDataset
from deeptune.utilities import avg_pool
from deeptune.src.nlp.E5_roberta import load_nlp_intfloat_ml_model_offline
from deeptune.options import DEEPTUNE_RESULTS

"""
Note that as RoBERTa is built for embeddings extraction, We consider at DeepTune that it is currently the best practice to support the extraction of embeddings from the E5 model only from further modifications, if you want to modify the architecture you may refer to the MultiLingualBERT model.
"""


# Initialize the needed variables either from the CLI user sents or from the device.

parser = ArgumentParser(description="Extract Embeddings using E5 Multilingual model based on Roberta")
OUTPUT = DEEPTUNE_RESULTS / f"test_set_pretrained_XLMRoBERTa_embeddings.parquet"
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DF_PATH = args.input_dir
OUTPUT = args.output

# load the model, the tokenizer and the dataset.

model, tokenizer = load_nlp_intfloat_ml_model_offline()

text_dataset = TextDataset.from_parquet(INPUT_DF_PATH, tokenizer=tokenizer)

loader = DataLoader(
    text_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# this is the wrapping function that calls the main function to extract the embeddings from the model.
def run_embeddings():
    
    
    device = torch.device(DEVICE)
    model.to(device)
    df = get_nlp_embeddings(model, loader, device)
    
    df.to_parquet(OUTPUT)
    print(f"Saved text embeddings to {OUTPUT}.")
    

def get_nlp_embeddings(model, loader, device):
    
    """
    Extract embeddings from the Multilingual Adjusted BERT
    
    Args:
    
        model (HuggingFace Model): The HuggingFace Multinlingual BERT model.
        
        loader (torch.utils.data.DataLoader): The test dataloader.
        
        device (torch.device): The device to run the model on (CPU/GPU).
    """
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        model.eval()
        
        print("Starting Text embedding..")
        print(f"Using device: {device}")
        
        # Here for each batch from the dataset, we do the forward pass and return the results
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
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    
    
    # Create the dataframe stroing the embeddings and the labels.
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed['target'] = labels.numpy()
    
    
    return df_embed
    
    
if __name__ == "__main__":
    
    run_embeddings()