
import torch
from deeptune.datasets.text_datasets import TextDataset
import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score
from deeptune.utilities import save_test_metrics
from deeptune.src.nlp.gpt2 import AdjustedGPT2Model,load_gpt2_model_offline
from deeptune.utilities import save_cli_args,load_finetunedbert_model,get_args,load_finetuned_gpt2
import options
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import logging
import options

# Initialize the needed variables either from the CLI user sents or from the device.

parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = get_args()

TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
USE_PEFT = args.use_peft
ADJUSTED_GPT2_MODEL_DIR = args.adjusted_gpt2_dir
FREEZE_BACKBONE = args.freeze_backbone

if USE_PEFT:
    pass
else:
    gpt2_model,_ = load_gpt2_model_offline()
    model =AdjustedGPT2Model(gpt_model=gpt2_model, freeze_backbone=FREEZE_BACKBONE)
    args.model = 'Adjusted GPT-2 model.'
    
# Load the test dataset from its path
df = pd.read_parquet(TEST_DATASET_PATH)

model,tokenizer = load_finetuned_gpt2(ADJUSTED_GPT2_MODEL_DIR)
model = model.to(DEVICE)

# Define the loss function, load the dataset
criterion = nn.CrossEntropyLoss()
logger = logging.getLogger()

test_dataset = TextDataset.from_parquet(parquet_file=TEST_DATASET_PATH, tokenizer=tokenizer)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


def test():
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
        for batch_idx, (encoding, labels) in test_pbar:
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
    save_test_metrics(test_accuracy=test_accuracy, output_dir=TEST_OUTPUT_DIR)

    print(metrics_dict)

    
if __name__ == "__main__":
    test()
    save_cli_args(args, TEST_OUTPUT_DIR, mode='test')
    print('Test results saved successfully!')
