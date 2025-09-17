from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from datasets.text_datasets import TextDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import options
import pandas as pd
from pathlib import Path
from helpers import load_finetunedbert_model
import time
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from embed.vision.custom_embed_siglip_handler import embed_with_siglip
from cli import DeepTuneVisionOptions
from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP, RunType, save_process_times


"""
Please Note that that extracting embeddings from MultiLingualBERT is only supported through the finetuned or PEFT version. If you want to use original pre-tranied model please refer to the XLM RoBERTa in DeepTune.
"""

def main():
     
    args = DeepTuneVisionOptions(RunType.EMBED)

    DF_PATH = args.df
    MODE = args.mode
    OUT = args.out
    USE_CASE = args.use_case.value
    NUM_CLASSES = args.num_classes
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    MODEL_WEIGHTS = args.model_weights

    MODEL_STR = 'PEFT-BERT' if USE_CASE == 'peft' else 'BERT'

    MODEL_WEIGHTS = args.model_weights
    # USE_CASE = args.use_case.value

    BATCH_SIZE = args.batch_size
    EMBED_OUTPUT = (OUT / f"embed_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/embed_output_{MODEL_STR}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{MODEL_STR}_{USE_CASE}_embeddings.parquet"
    print(USE_CASE)

    if USE_CASE == 'finetuned':
        model = CustomMultilingualBERT(NUM_CLASSES,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
        args.use_case = 'finetuned-MultiLingualBERT'
        pass
    elif USE_CASE == 'peft':
        model = CustomMultilingualPeftBERT(NUM_CLASSES,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
        args.use_case = 'finetuned-MultiLingualBERT'
    else:
        raise ValueError('There is no third option other than ["finetuned", "peft"]')

    start_time = time.time()
    
    # load the model, the tokenizer and the dataset.
    _,tokenizer = load_finetunedbert_model(MODEL_WEIGHTS)
    model = model.to(DEVICE)

    df = pd.read_parquet(DF_PATH)

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    others = df.drop(columns=['text','label'],errors='ignore')


    all_embeddings=[]
    all_labels=[]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), total=(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE, desc="Embedding Text"):
            batch_texts  = texts[i:i + BATCH_SIZE]
            batch_labels = labels[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Maintain inner BERT logic: get [CLS] then optionally pass through additional layer(s)
            bert_outputs = model.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids", None)
            )
            embeddings = bert_outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

            if ADDED_LAYERS == 2:
                embeddings = model.additional(embeddings)

            all_embeddings.append(embeddings.cpu())
            all_labels.extend(batch_labels)

    # Concatenate and build DataFrame
    all_embeddings = torch.cat(all_embeddings, dim=0)
    p = all_embeddings.shape[1]
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = pd.DataFrame(all_embeddings.numpy(), columns=cols)
    df_embed["label"] = np.array(all_labels)
    df_embed = df_embed.reset_index(drop=True)
    others = others.reset_index(drop=True)
    df_embed = pd.concat([df_embed, others], axis=1)

    df_embed.to_parquet(EMBED_FILE, index=False)
    print(f"Saved text embeddings to {OUT}")


    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")
    
    return df_embed
            
if __name__ == "__main__":
    
    main()
                