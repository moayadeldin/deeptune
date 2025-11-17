from src.nlp.multilingual_bert import CustomMultilingualBERT,load_nlp_bert_ml_model_offline
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
from cli import DeepTuneVisionOptions
from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP, RunType, save_process_times


def embed(df_path, out, model_weights, batch_size, use_case,num_classes,added_layers,embed_size,freeze_backbone=False):

    if use_case == 'peft':
        MODEL_STR = 'PEFT_BERT'
    elif use_case == 'finetuned':
        MODEL_STR = 'FINETUNED_BERT'
    else:
        MODEL_STR = 'PRETRAINED_BERT'

    # use_case = args.use_case.value

    EMBED_OUTPUT = (out / f"embed_output_{MODEL_STR}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{MODEL_STR}_{use_case}_embeddings.parquet"

    if use_case == 'finetuned':
        model = CustomMultilingualBERT(num_classes=num_classes,added_layers=added_layers, embedding_layer=embed_size,freeze_backbone=freeze_backbone)
        pass
    elif use_case == 'peft':
        model = CustomMultilingualPeftBERT(num_classes=num_classes,added_layers=added_layers, embedding_layer=embed_size,freeze_backbone=freeze_backbone)

    elif use_case == 'pretrained':
        model = CustomMultilingualBERT(num_classes=num_classes,added_layers=added_layers, embedding_layer=embed_size,freeze_backbone=freeze_backbone,pretrained=True)
    else:
        raise ValueError('There is no fourth option other than ["finetuned", "peft", "pretrained"]')

    start_time = time.time()
    
    # load the model, the tokenizer and the dataset.
    if use_case == 'pretrained':
        _, tokenizer = load_nlp_bert_ml_model_offline()
    else:
        _,tokenizer = load_finetunedbert_model(model_weights)
    model = model.to(DEVICE)

    df = pd.read_parquet(df_path)

    texts = df['text'].tolist()
    labels = df['labels'].tolist()

    others = df.drop(columns=['text','labels'],errors='ignore')


    all_embeddings=[]
    all_labels=[]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="Embedding Text"):
            batch_texts  = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

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

            if added_layers == 2:
                embeddings = model.additional(embeddings)

            all_embeddings.append(embeddings.cpu())
            all_labels.extend(batch_labels)

    # Concatenate and build DataFrame
    all_embeddings = torch.cat(all_embeddings, dim=0)
    p = all_embeddings.shape[1]
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = pd.DataFrame(all_embeddings.numpy(), columns=cols)
    df_embed["labels"] = np.array(all_labels)
    df_embed = df_embed.reset_index(drop=True)
    others = others.reset_index(drop=True)
    df_embed = pd.concat([df_embed, others], axis=1)


    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")

    df_embed.to_parquet(EMBED_FILE, index=False)
    print(f'The embeddings file is saved in {EMBED_OUTPUT}')
    
    return out,df_embed.shape


def main():
     
    args = DeepTuneVisionOptions(RunType.EMBED)

    DF_PATH = args.df
    OUT = args.out
    USE_CASE = args.use_case.value
    NUM_CLASSES = args.num_classes
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    MODEL_WEIGHTS = args.model_weights
    BATCH_SIZE = args.batch_size

    embed_output_path = embed(
        df_path=DF_PATH,
        out=OUT,
        use_case=USE_CASE,
        num_classes=NUM_CLASSES,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        freeze_backbone=FREEZE_BACKBONE ,
        model_weights=MODEL_WEIGHTS,
        batch_size=BATCH_SIZE
    )

    # print(f'The embeddings file is saved in {embed_output_path}')

            
if __name__ == "__main__":
    
    main()
                