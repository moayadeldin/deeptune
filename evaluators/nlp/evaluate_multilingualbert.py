
import torch
from datasets.text_datasets import TextDataset
import pandas as pd 
from sklearn.metrics import classification_report, roc_auc_score
from utilities import save_test_metrics
from src.nlp.multilingual_bert import CustomMultilingualBERT
from utilities import save_cli_args,load_finetunedbert_model
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
args = parser.parse_args()

TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_PEFT = args.use_peft
MODEL_WEIGHTS = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
MODE = args.mode
ADJUSTED_BERT_MODEL_DIR = args.adjusted_bert_dir
FREEZE_BACKBONE = args.freeze_backbone

if USE_PEFT:
    pass
else:
    model = CustomMultilingualBERT(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
    args.model = 'Multilingual BERT'
    
# Load the test dataset from its path
df = pd.read_parquet(TEST_DATASET_PATH)

# Load the model and tokenizer
model,tokenizer = load_finetunedbert_model(ADJUSTED_BERT_MODEL_DIR)
model = model.to(DEVICE)

# Define the loss function, load the dataset
criterion = nn.CrossEntropyLoss()
logger = logging.getLogger()

test_dataset = TextDataset(parquet_file=TEST_DATASET_PATH, tokenizer=tokenizer)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


def test():
    
    # initialize the metrics for validation
    test_accuracy=0.0
    test_loss=0.0
    total,correct=0,0
    
    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        
    all_labels = []
    all_predictions=[]
    all_probs=[]
    
    with torch.no_grad():
        
        for _, (text, labels) in test_pbar:
            labels = labels.to(DEVICE)
            # Initialize the tokenizer
            encoding = tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
            )
            # move the input to GPU 
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            token_type_ids = encoding.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            
            # Apply forward pass and accumulate loss
            outputs = model(input_ids, attention_mask, token_type_ids)   
    
            loss = criterion(outputs,labels)
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
    save_test_metrics(test_accuracy=test_accuracy, output_dir=TEST_OUTPUT_DIR)

    print(metrics_dict)
    
if __name__ == "__main__":
    
    # Call the test function with saving CLI
    
    test()
    
    save_cli_args(args, TEST_OUTPUT_DIR, mode='test')
    
    print('Test results saved successfully!')