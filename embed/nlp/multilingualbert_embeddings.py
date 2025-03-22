from src.nlp.multilingual_bert import CustomMultilingualBERT
from datasets.text_datasets import TextDataset
from utilities import transformations
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import options
import pandas as pd
from pathlib import Path
from utilities import load_finetunedbert_model


"""
Please Note that that extracting embeddings from MultiLingualBERT is only supported through the finetuned or PEFT version. If you want to use original pre-tranied model please refer to the XLM RoBERTa in DeepTune.
"""

# Initialize the needed variables either from the CLI user sents or from the device.

DEVICE = options.DEVICE
parser = options.parser
args = parser.parse_args()


USE_CASE = args.use_case
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL_PATH = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FREEZE_BACKBONE = args.freeze_backbone
MODE = args.mode
INPUT_DF_PATH = args.input_dir
ADJUSTED_BERT_MODEL_DIR = args.adjusted_bert_dir

# Check which USE_CASE is used and based on this choose the model to get loaded. For example, if finetuned was the USE_CASE then the class call would be from the transfer-learning without PEFT version.

if USE_CASE == 'finetuned':
    model = CustomMultilingualBERT(NUM_CLASSES,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
    OUTPUT = f"deeptune_results/test_set_finetuned_MultilingualBERT_embeddings.parquet"
    args.use_case = 'finetuned-MultiLingualBERT'

elif USE_CASE == 'peft':
    pass
else:
    raise ValueError('There is no third option other than ["finetuned", "peft"]')

# load the model, the tokenizer and the dataset.
model,tokenizer = load_finetunedbert_model(ADJUSTED_BERT_MODEL_DIR)
model = model.to(DEVICE)

text_dataset = TextDataset(INPUT_DF_PATH, tokenizer=tokenizer)

data_loader = torch.utils.data.DataLoader(
    text_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# this is the wrapping function that calls the main function to extract the embeddings from the model.
def run_embeddings():
    
    device = torch.device(DEVICE)
    model.to(device)
    df = get_nlp_embeddings(model, data_loader, device)
    
    df.to_parquet(OUTPUT)
    print(f"Saved text embeddings to {OUTPUT}.")

def get_nlp_embeddings(model,loader,device):
    
    """
    Extract embeddings from the Multilingual Adjusted BERT
    
    Args:
    
        model (HuggingFace Model): The HuggingFace Multinlingual BERT model.
        
        loader (torch.utils.data.DataLoader): The test dataloader.
        
        device (torch.device): The device to run the model on (CPU/GPU).
    """
    
    all_embeddings=[]
    all_labels=[]
    
    with torch.no_grad():
        model.eval()
        
        for batch_dict,labels in tqdm(loader,total=len(loader),desc="Embedding Text"):
            
            print("Starting Text embedding..")
            print(f"Using device: {device}")
            
            batch_dict = {key: val.to(device) for key, val in batch_dict.items()}
            labels = labels.to(device)
        
            # Here we check if the added layers is 2 then we want to extract the embeddings from the intermediate additional layer. otherwise we extract the embeddings from the last layer directly.
            if ADDED_LAYERS ==2:
                
                    bert_outputs = model.bert(
                        input_ids=batch_dict["input_ids"],
                        attention_mask=batch_dict["attention_mask"],
                        token_type_ids=batch_dict.get("token_type_ids", None)
                    )
                    
                    
                    # Here we decide to extract the [CLS] token embedding, the first token's hidden state
                    embeddings = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
                    
                    embeddings = model.additional(embeddings)
                    
            else:
                
                    bert_outputs = model.bert(
                        input_ids=batch_dict["input_ids"],
                        attention_mask=batch_dict["attention_mask"],
                        token_type_ids=batch_dict.get("token_type_ids", None)
                    )
                    
                    
                    # Here we decide to extract the [CLS] token embedding, the first token's hidden state
                    embeddings = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
                    
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    
    # Create the dataframe stroing the embeddings and the labels.
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = pd.DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed['target'] = labels.numpy()
    
    return df_embed
            
if __name__ == "__main__":
    
    run_embeddings()
                