from models.resnet18 import adjustedResNet
from models.resnet18_peft import adjustedPeftResNet
import importlib
from utilities import transformations
from utilities import save_training_metrics
from utilities import save_cli_args
import os
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import torch
from dataset import ParquetImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd 
import warnings
import logging
import argparse
from utilities import PerformanceLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).parent.parent

TEST_OUTPUT_DIR = ROOT / 'output_directory_test'

parser = argparse.ArgumentParser(description="Test the Trained Model on Your The test set.")

parser.add_argument('--model', choices=['peft-resnet18', 'resnet18'], help="Choose the Model you want to use.")
parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')
parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to test your model.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing test data.')
parser.add_argument('--model_weights', type=str, required=True, help='Directory containing trained model weights.')


### These are only kept for the CLI arguments function. ###
parser.add_argument('--num_epochs', type=str, required=False, default='None', help='The number of epochs you wan the model to run on.')
parser.add_argument('--learning_rate', type=str, required=False, default='None', help='Learning Rate to apply for fine-tuning.')



args = parser.parse_args()

TEST_DATASET_PATH = args.input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_PEFT = args.use_peft
MODEL_WEIGHTS = args.model_weights

if USE_PEFT:
    
    MODEL = adjustedPeftResNet(NUM_CLASSES)
    
else:
    MODEL = adjustedResNet(NUM_CLASSES)

# WE load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
df = pd.read_parquet(TEST_DATASET_PATH)

test_dataset = ParquetImageDataset(parquet_file=TEST_DATASET_PATH, transform=transformations)

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
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.criterion = nn.CrossEntropyLoss()
        
        
    def test(self, best_model_weights_path=None):
        
        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')
            
        self.model.to(DEVICE)
        
        test_accuracy=0.0
        test_loss=0.0
        total,correct = 0,0
        
        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        with torch.no_grad():

            for _, (input, labels) in test_pbar:

                self.model.eval()
                
                inputs, labels = input.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs,labels)
                
                test_loss += loss.item()
                
                total += labels.size(0)
                correct += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                
                
        test_accuracy = ( correct / total ) * 100
        test_loss = test_loss / len(test_loader)
        print(test_accuracy, test_loss)
        
        self.logger.info(f"Test accuracy: {(test_accuracy)}%")

        save_training_metrics(test_accuracy=test_accuracy,output_dir=TEST_OUTPUT_DIR)
        
        save_cli_args(args, TEST_OUTPUT_DIR)
  
  
if __name__ == "__main__":
    
    test_trainer = TestTrainer(model=MODEL, batch_size=BATCH_SIZE)
    
    test_trainer.test(best_model_weights_path=MODEL_WEIGHTS)