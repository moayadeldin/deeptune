
import torch
from datasets.text_datasets import TextDataset
from sklearn.metrics import classification_report, roc_auc_score
from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from helpers import load_finetunedbert_model
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import logging
import time
from utils import save_process_times
from cli import DeepTuneVisionOptions
from options import UNIQUE_ID, DEVICE
from utils import RunType
from datasets.text_datasets import TextDataset

def evaluate(eval_df, out, model_weights, num_classes,added_layers,embed_size, batch_size, freeze_backbone, args, use_peft, model_str):


    if use_peft:
        model = CustomMultilingualPeftBERT(num_classes=num_classes,added_layers=added_layers,embedding_layer=embed_size,freeze_backbone=freeze_backbone)
    else:
        model = CustomMultilingualBERT(num_classes=num_classes, added_layers=added_layers, embedding_layer=embed_size,freeze_backbone=freeze_backbone)

    _,tokenizer = load_finetunedbert_model(model_weights)

    TEST_OUTPUT_DIR = (out / f"test_output_{model_str}_{UNIQUE_ID}")

    model.to(device=DEVICE)

    test_dataset = TextDataset(parquet_file=eval_df, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
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
        for _, (encoding, labels, *_) in test_pbar:
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

    logger.info(f"The test accuracy is: {test_accuracy}, while the test loss is: {test_loss}")
    # print(metrics_dict)
    args.save_args(TEST_OUTPUT_DIR)

    with open(TEST_OUTPUT_DIR / "full_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="evaluation")

    return metrics_dict


def main():

    args = DeepTuneVisionOptions(RunType.EVAL)

    EVAL_PATH = args.eval_df
    OUT = args.out
    USE_PEFT = args.use_peft
    MODEL_STR = 'PEFT-BERT' if USE_PEFT else 'BERT'

    BATCH_SIZE = args.batch_size
    
    MODEL_WEIGHTS = args.model_weights
    FREEZE_BACKBONE = args.freeze_backbone
    NUM_CLASSES = args.num_classes
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size

    evaluate(
        eval_df=EVAL_PATH,
        out=OUT,
        model_weights=MODEL_WEIGHTS,
        freeze_backbone=FREEZE_BACKBONE,
        added_layers=ADDED_LAYERS,
        num_classes=NUM_CLASSES,
        embed_size=EMBED_SIZE,
        use_peft=False,
        args=args,
        batch_size=BATCH_SIZE,
        model_str=MODEL_STR
    )
    
    
if __name__ == "__main__":
    
    main()