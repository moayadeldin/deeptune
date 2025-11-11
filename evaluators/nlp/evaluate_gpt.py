
import torch
from datasets.text_datasets import TextDataset
from sklearn.metrics import classification_report, roc_auc_score
from src.nlp.gpt2 import AdjustedGPT2Model,load_gpt2_model_offline
from helpers import load_finetuned_gpt2
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import logging
import json
import time
from cli import DeepTuneVisionOptions
from options import UNIQUE_ID, DEVICE
from utils import RunType
from utils import save_process_times
from datasets.text_datasets import TextDataset

def evaluate(eval_df, out, model_weights, batch_size, freeze_backbone, args, model_str, use_peft=False):

    if use_peft:
        raise ValueError("PEFT is not supported for GPT2 yet.")
    else:
        gpt2_model,tokenizer = load_gpt2_model_offline()
        model =AdjustedGPT2Model(gpt_model=gpt2_model, freeze_backbone=freeze_backbone)
    
    model,tokenizer = load_finetuned_gpt2(model_weights)

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
    
        for _, (encoding, labels, _) in test_pbar:
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

    print(f"The test accuracy is: {test_accuracy}, while the test loss is: {test_loss}")
    logger.info(f"Test accuracy: {test_accuracy:.2f}%")
    print(metrics_dict)
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
    MODEL_STR = 'GPT2'
    FREEZE_BACKBONE = args.freeze_backbone

    BATCH_SIZE = args.batch_size
    
    MODEL_WEIGHTS = args.model_weights
    FREEZE_BACKBONE = args.freeze_backbone

    evaluate(
        eval_df=EVAL_PATH,
        out=OUT,
        model_weights=MODEL_WEIGHTS,
        model_str=MODEL_STR,
        freeze_backbone=FREEZE_BACKBONE,
        use_peft=False,
        args=args,
        batch_size=BATCH_SIZE,
    )



    
if __name__ == "__main__":
    main()
