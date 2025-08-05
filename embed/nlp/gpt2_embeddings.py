import torch 
from helpers import load_finetuned_gpt2
from src.nlp.gpt2 import download_gpt2_model, load_gpt2_model_offline,AdjustedGPT2Model
from argparse import ArgumentParser
import options
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from embed.vision.custom_embed_siglip_handler import embed_with_siglip
from cli import DeepTuneVisionOptions
from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP, RunType

def main():

    args = DeepTuneVisionOptions(RunType.EMBED)

    DF_PATH = args.df
    MODE = args.mode
    OUT = args.out

    MODEL_STR = 'GPT2'

    MODEL_PATH = args.model_weights
    # USE_CASE = args.use_case.value

    BATCH_SIZE = args.batch_size
    EMBED_OUTPUT = (OUT / f"embed_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/embed_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{MODEL_STR}_{MODE}_embeddings.parquet"

    # load the model, the tokenizer and the dataset.
    gpt_model,_ = load_gpt2_model_offline()
    _,tokenizer = load_finetuned_gpt2(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(DF_PATH)

    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    adjusted_model = AdjustedGPT2Model(gpt_model=gpt_model).to(DEVICE)
        
    device = torch.device(DEVICE)
    adjusted_model.to(device)
    
    df.to_parquet(EMBED_FILE, index=False)
    print(f"Saved text embeddings to {OUT}")

    all_embeddings = []
    all_labels = []

    adjusted_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]

            inputs = tokenizer(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # get sentence-level embeddings directly
            outputs = adjusted_model(**inputs)
            mapped_embeddings = outputs

            all_embeddings.append(mapped_embeddings.cpu())
            all_labels.extend(batch_labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    embeddings_df = pd.DataFrame(all_embeddings.numpy())
    embeddings_df['label'] = all_labels


# def getting_mean_embeds_without_padding_tokens(inputs, outputs):
    
#     """
#     This function is mainly to mask out the padding tokens in the sequence so that the embeddings don't contain the noise of padding tokens before averaging, making them more accurate
#     """
    
#     attention_mask = inputs['attention_mask']
    
#     last_hidden_state = outputs.last_hidden_state # (B,seq_len,hidden_size)
    
#     # expand attention mask for matching dimensions
#     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()) # (B, seq_len)
    
#     # zero out the embeddings where attention mask is zero.
    
#     last_hidden_state = last_hidden_state * attention_mask_expanded
    
#     # sum along the sequence dimension to get the numerator for the mean
    
#     summed = last_hidden_state.sum(dim=1)
    
#     # count how many non-pad tokens per sentence
#     counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
    
#     # get the mean of embeddings by dividing summed vectors by token counts
    
#     mean_embeddings = summed / counts
    
#     return mean_embeddings

if __name__ == "__main__":
    
    main()




    


