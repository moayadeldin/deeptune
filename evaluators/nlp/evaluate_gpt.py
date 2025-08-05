
import torch
from datasets.text_datasets import TextDataset
import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score
from src.nlp.gpt2 import AdjustedGPT2Model,load_gpt2_model_offline
from helpers import load_finetuned_gpt2
import options
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import logging
import options

from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls,RunType,set_seed
from datasets.text_datasets import TextDataset

def main():

    args = DeepTuneVisionOptions(RunType.EVAL)

    TEST_PATH = args.eval_df
    OUT = args.out
    MODEL_STR = 'GPT2'
    MODE = args.mode
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft

    BATCH_SIZE = args.batch_size
    
    MODEL_WEIGHTS = args.model_weights
    FREEZE_BACKBONE = args.freeze_backbone

    TEST_OUTPUT_DIR = (OUT / f"test_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/test_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")

    if USE_PEFT:
        pass
    else:
        gpt2_model,tokenizer = load_gpt2_model_offline()
        model =AdjustedGPT2Model(gpt_model=gpt2_model, freeze_backbone=FREEZE_BACKBONE)
    
    model,tokenizer = load_finetuned_gpt2(MODEL_WEIGHTS)

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

    model.eval()
    with torch.no_grad():
        for _, (encoding, labels) in test_pbar:
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total

    all_probs = np.concatenate(all_probs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics_dict = {
        "loss": test_loss,
        "accuracy": test_accuracy
    }
    
    report = classification_report(y_true=all_labels, y_pred=all_predictions, output_dict=True)
    metrics_dict.update(report)

    try:
        metrics_dict["auroc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        metrics_dict["auroc"] = "AUROC not applicable for this setup"

    print(test_accuracy, test_loss)
    logger.info(f"Test accuracy: {test_accuracy:.2f}%")
    print(metrics_dict)
    args.save_args(TEST_OUTPUT_DIR)

    
if __name__ == "__main__":
    main()
