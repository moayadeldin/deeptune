import importlib
from src.vision.siglip_peft import load_peft_siglip_for_image_classification_offline
from src.vision.siglip import load_siglip_for_image_classification_offline
from utilities import save_training_metrics
from utilities import save_cli_args
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pyarrow.parquet as pq
import torch
from datasets.image_datasets import ParquetImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
from pathlib import Path
import pandas as pd 
import logging
import numpy as np
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).parent.parent

TEST_OUTPUT_DIR = ROOT / 'output_directory_test'

parser = argparse.ArgumentParser(description="Test the Trained Model on Your The test set.")


parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to test your model.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing test data.')

args = parser.parse_args()

TEST_DATASET_PATH = args.input_dir
BATCH_SIZE= args.batch_size
USE_PEFT = args.use_peft

if USE_PEFT:
    
    MODEL,PROCESSOR = load_peft_siglip_for_image_classification_offline()
    args.model = 'PEFT-SIGLIP'
    
else:
    
    MODEL,PROCESSOR = load_siglip_for_image_classification_offline()
    args.model = 'SIGLIP'
    
    
args.num_epochs = 'None'
args.learning_rate = 'None'
args.num_classes = 'None'

# WE load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
df = pd.read_parquet(TEST_DATASET_PATH)

test_dataset = ParquetImageDataset(parquet_file=TEST_DATASET_PATH, processor=PROCESSOR)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class TestTrainer:
    
    def __init__(self, model, batch_size):
        
        self.model = model
        self.batch_size = batch_size
        
        self.model.to(DEVICE)
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.criterion = nn.CrossEntropyLoss()
        
    def test(self):
        
        test_accuracy=0.0
        test_loss=0.0
        total, correct= 0,0
        
        test_pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        )
        
        all_labels=[]
        all_predictions=[]
        all_probs=[]
        
        with torch.no_grad():
            for _, (image_pixels, labels) in test_pbar:
                
                self.model.eval()
                
                image_pixels,labels = image_pixels.to(DEVICE), labels.to(DEVICE)

                outputs = self.model.vision_model(pixel_values=image_pixels)
                
                logits = outputs.pooler_output  
                
                probs = torch.softmax(logits, 1)
                
                _, predicted = torch.max(probs,1)

                loss = self.criterion(logits, labels)
                
                test_loss += loss.item()
                
                # Store all probabilities, predictions, and labels to the CPU memory
                all_probs.append(probs.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                                
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
                
        all_probs = np.concatenate(all_probs, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        test_accuracy = correct / total
        test_loss = test_loss / len(test_loader)
        
        
        metrics_dict = {}
        
        metrics_dict['accuracy'] = test_accuracy
        metrics_dict['loss'] = test_loss
        
        report = classification_report(
            y_true = all_labels,
            y_pred = all_predictions,
            output_dict=True
        )
        
        print('Metrics Dictionary is being computed..')
        
        metrics_dict.update(report)
        
        metrics_dict['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)
        
        try:
            metrics_dict['auroc'] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            metrics_dict['auroc'] = "AUROC not applicable for this setup"
            
            

        print(test_accuracy,test_loss)
        save_training_metrics(test_accuracy=test_accuracy,output_dir=TEST_OUTPUT_DIR)
        save_cli_args(args, TEST_OUTPUT_DIR)
        
        print(metrics_dict)
        return metrics_dict
    
if __name__ == "__main__":
    
    test_trainer = TestTrainer(MODEL,BATCH_SIZE)
    
    test_trainer.test()
        
    