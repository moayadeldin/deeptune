
import torch
from datasets.text_datasets import TextDataset
from src.nlp.multilingual_bert import load_nlp_bert_ml_model_offline
import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score
from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from helpers import load_finetunedbert_model
import options
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import logging
import options
import time
from utils import save_process_times
from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls,RunType,set_seed
from datasets.text_datasets import TextDataset

def main():

    args = DeepTuneVisionOptions(RunType.EVAL)

    TEST_PATH = args.eval_df
    OUT = args.out
    USE_PEFT = args.use_peft
    MODEL_STR = 'PEFT-BERT' if USE_PEFT else 'BERT'

    BATCH_SIZE = args.batch_size
    
    MODEL_WEIGHTS = args.model_weights
    FREEZE_BACKBONE = args.freeze_backbone
    NUM_CLASSES = args.num_classes
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size

    TEST_OUTPUT_DIR = (OUT / f"test_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/test_output_{MODEL_STR}_{UNIQUE_ID}")

    if USE_PEFT:
        model = CustomMultilingualPeftBERT(NUM_CLASSES,ADDED_LAYERS,EMBED_SIZE,FREEZE_BACKBONE)
    else:
        model = CustomMultilingualBERT(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)

    _,tokenizer = load_finetunedbert_model(MODEL_WEIGHTS)

    model.to(device=DEVICE)

    test_dataset = TextDataset(parquet_file=TEST_PATH, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    logger = logging.getLogger()

    # initialize metrics
    test_accuracy = 0.0
    test_loss = 0.0
    total, correct = 0, 0

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    all_labels = []
    all_predictions = []
    all_probs = []

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for _, (encoding, labels) in test_pbar:
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            token_type_ids = encoding.get('token_type_ids', None)
            
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            
            labels = labels.to(DEVICE)
           
            # Apply forward pass and accumulate loss
            outputs = model(input_ids, attention_mask, token_type_ids)  
   
            loss = criterion(outputs, labels)
           
            test_loss += loss.item()
    
            # Calculate accuracy
            probs = torch.softmax(outputs, 1)
            _, predicted = torch.max(probs, 1)
            
            total += labels.size(0)
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            
            # Store classification outputs
            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
          
    # Add the accuracy and loss to the metrics dictionary  
    test_loss = test_loss / len(test_loader)
    metrics_dict = {"loss": test_loss}

    test_accuracy = (correct / total) * 100
    metrics_dict["accuracy"] = test_accuracy
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate classification report and AUROC (if applicable)
    report = classification_report(y_true=all_labels, y_pred=all_predictions, output_dict=True)
    metrics_dict.update(report)
    
    try:
        metrics_dict["auroc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        metrics_dict["auroc"] = "AUROC not applicable for this setup"
    
    print(test_accuracy, test_loss)
    logger.info(f"Test accuracy: {test_accuracy}%")
    print(metrics_dict)
    args.save_args(TEST_OUTPUT_DIR)

    with open(TEST_OUTPUT_DIR / "full_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="evaluation")

    
if __name__ == "__main__":
    
    main()