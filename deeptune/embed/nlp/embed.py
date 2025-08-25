import pandas as pd
import torch

from argparse import ArgumentError
from pandas import DataFrame
from pathlib import Path
from transformers import BertTokenizer, GPT2Tokenizer
from typing import Type
from tqdm import tqdm

from deeptune.datasets.text_datasets import TextDataset
from deeptune.options import DEEPTUNE_RESULTS, DEVICE, UNIQUE_ID
from deeptune.utilities import get_args

from deeptune.src.nlp.gpt2 import AdjustedGPT2Model, load_tuned_gpt2, load_gpt2_tokenizer_offline
from deeptune.src.nlp.multilingual_bert import CustomMultilingualBERT, load_bert_tokenizer_offline, load_tuned_bert, \
    CustomMultilingualPeftBERT
# from deeptune_beta.src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT

from deeptune.utils import UseCase


def main():
    args = get_args()

    USE_CASE = UseCase.from_string(args.use_case)
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = args.num_classes
    MODEL_PATH = args.model_weights
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    # FREEZE_BACKBONE = args.freeze_backbone
    MODE = args.mode
    INPUT_DF_PATH = args.input_dir

    ADJUSTED_BERT_MODEL_DIR = args.adjusted_bert_dir
    ADJUSTED_GPT2_PATH = args.adjusted_gpt2_dir

    model_name: str
    if ADJUSTED_BERT_MODEL_DIR is not None:
        model_name = "MultilingualBERT"
    elif ADJUSTED_GPT2_PATH is not None:
        model_name = "GPT2"
    else:
        raise ArgumentError("Valid model path not specified.")
    
    MODEL_PATH = ADJUSTED_BERT_MODEL_DIR or ADJUSTED_GPT2_PATH

    df = pd.read_parquet(INPUT_DF_PATH)

    embed_df = embed_nlp_dataset(
        df=df,
        mode=MODE,
        num_classes=NUM_CLASSES,
        model_version=model_name,
        model_architecture=model_name,
        model_path=MODEL_PATH,
        use_case=USE_CASE,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
    )
    OUTPUT = DEEPTUNE_RESULTS / f"{USE_CASE.value}_{model_name}_embeddings_{UNIQUE_ID}.parquet"
    embed_df.to_parquet(OUTPUT)
    print(f"Saved text embeddings to {OUTPUT}.")


def embed_nlp_dataset(
    df: DataFrame,
    mode: str,
    num_classes: int,
    model_version: str,
    model_architecture: str,
    model_path: Path,
    use_case: UseCase,
    added_layers: int,
    embed_size: int,
    batch_size: int,
):
    model, tokenizer = load_nlp_model_tokenizer(model_version, model_path)
    text_dataset = TextDataset(df=df, tokenizer=tokenizer)
    
    data_loader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    embedded_df = get_nlp_embeddings(model, data_loader, DEVICE)

    if not text_dataset.has_labels:
        embedded_df = embedded_df.drop(columns=["target"])

    return embedded_df


def load_nlp_model_tokenizer(
    model_version: str, 
    model_path: Path,
) -> tuple[
    CustomMultilingualBERT | AdjustedGPT2Model, 
    BertTokenizer | GPT2Tokenizer]:
    if model_version.lower() == "multilingualbert":
        model = load_tuned_bert(model_path)
        tokenizer = load_bert_tokenizer_offline()

    elif model_version.lower() == "gpt2":
        model = load_tuned_gpt2(model_path)
        tokenizer = load_gpt2_tokenizer_offline()
    
    return model, tokenizer


def run_embeddings(model, data_loader, out) -> None:
    df = get_nlp_embeddings(model, data_loader, DEVICE)
    df.to_parquet(out)
    print(f"Saved text embeddings to {out}.")


def get_nlp_embeddings(model, loader, device) -> DataFrame:
    model.to(device)
    model.eval()

    all_embeddings=[]
    all_labels=[]
    
    print(f"Using device: {device}")
    print("Starting Text embedding..")

    with torch.no_grad():
        for batch_dict, labels in tqdm(loader, total=len(loader), desc="Embedding Text"):
            
            batch_dict = {key: val.to(device) for key, val in batch_dict.items()}
            labels = labels.to(device)
        
            if isinstance(model, (CustomMultilingualBERT, CustomMultilingualPeftBERT)):
                bert_outputs = model.bert(
                    input_ids=batch_dict["input_ids"],
                    attention_mask=batch_dict["attention_mask"],
                    token_type_ids=batch_dict.get("token_type_ids", None)
                )
                # Here we decide to extract the [CLS] token embedding, the first token's hidden state
                embeddings = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)

                # If the added layers is 2 extract the embeddings from the intermediate additional layer,
                # otherwise extract from the last layer directly.
                if hasattr(model, "added_layers") and  model.added_layers == 2:
                    embeddings = model.additional(embeddings)
                        
            elif isinstance(model, AdjustedGPT2Model):
                embeddings = model.get_embeddings(**batch_dict)

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
    main()