from src.vision.swin import adjustedSwin
from src.vision.swin_peft import adjustedPeftSwin
import importlib
from utilities import transformations
from utilities import save_training_metrics
from utilities import save_cli_args
import os
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import torch
from datasets.image_datasets import ParquetImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd 
import logging
import argparse
from utilities import PerformanceLogger
import warnings

##### SEED IS IMPORTANT TO ENSURE REPRODUCABILITY #####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")


# Arguments constructed for CLI

parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs you wan the model to run on.')
parser.add_argument('--added_layers', type=int, choices=[1,2], required=True, help='Specify the number of layers you want to add.')
parser.add_argument('--embed_size', type=int, required=False, help='This is optional in the case you want to work with Added Layers size of 2. Specify the size of the embeddings you would obtain through embedding layer.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to train your model.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--train_size', type=float, required=True, help='Mention the split ratio of the Train Dataset')
parser.add_argument('--val_size', type=float, required=True, help='Mention the split ratio of the Val Dataset')
parser.add_argument('--test_size', type=float, required=True, help='Mention the split ratio of the Test Dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing training data.')
parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')
parser.add_argument('--fixed-seed', action='store_true', help='Choose whether a seed is required or not.')
parser.add_argument('--freeze-backbone', action='store_true', help='Decide whether you want to freeze backbone or not.')


args = parser.parse_args()

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
CHECK_VAL_EVERY_N_EPOCH = 1
USE_PEFT = args.use_peft
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FIXED_SEED = args.fixed_seed
FREEZE_BACKBONE = args.freeze_backbone

if FIXED_SEED:
    seed=42
    args.fixed_seed = 42
else:
    seed = np.random.randint(low=0, high=1000)
    args.fixed_seed = seed
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

torch.manual_seed(seed)

def get_model():

    """Allows the user to choose from Adjusted Swin or PEFT-Swin versions.
    """

    if USE_PEFT:
        
        model = importlib.import_module('src.vision.swin_peft')
        args.model = 'PEFT-Swin'
        return model.adjustedPeftSwin

    else:
        model = importlib.import_module('src.vision.swin')
        args.model = 'Swin'
        return model.adjustedSwin
    
    
TRAIN_DATASET_PATH = Path(__file__).parent.parent / "train_split.parquet"
VAL_DATASET_PATH = Path(__file__).parent.parent / "val_split.parquet"
TEST_DATASET_PATH = Path(__file__).parent.parent / "test_split.parquet"

TRAINVAL_OUTPUT_DIR = Path(__file__).parent.parent / 'output_directory_trainval'


# WE load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
df = pd.read_parquet(INPUT_DIR)

df = df[:10]

train_data, temp_data = train_test_split(df, test_size=(1 - TRAIN_SIZE), random_state=seed)
val_data, test_data = train_test_split(temp_data, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), random_state=seed)

print('Number of Training Samples is', train_data.shape[0])
print('Number of Val Samples is', val_data.shape[0])
print('Number of Test Samples is', test_data.shape[0])

train_data.to_parquet(TRAIN_DATASET_PATH, index=False)
val_data.to_parquet(VAL_DATASET_PATH, index=False)
test_data.to_parquet(TEST_DATASET_PATH, index=False)

print("Data splits have been saved and overwritten if they existed.")

# The current datasets loaded as dataloaders
train_dataset = ParquetImageDataset(parquet_file=TRAIN_DATASET_PATH, transform=transformations)
val_dataset = ParquetImageDataset(parquet_file=VAL_DATASET_PATH, transform=transformations)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class Trainer:

    def __init__(self, model):
        
        self.model = model
        self.model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        
        self.performance_logger = PerformanceLogger(f'{TRAINVAL_OUTPUT_DIR}')

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

    def train(self,train_loader=train_loader):

        for epoch in range(NUM_EPOCHS):

            self.model.train()
            
            """
            The following arguments are as follows:
            
            - running_loss: indicating the loss during the epoch for train and val datasets respectively.
            - correct_predictions: number of predictions that equal true label 
            - total_predictions: number of predictions made in total
            """

            running_loss=0.0
            correct_predictions=0.0
            total_predictions=0
            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (inputs, labels) in train_pbar:
                
                # make sure the inputs and labels on GPU
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # null the gradient
                self.optimizer.zero_grad()

                # run the model
                outputs = self.model(inputs)

                # compute loss
                loss = self.criterion(outputs, labels)

                # backprop and apply optimizer
                loss.backward()
                self.optimizer.step()

                # accumulate loss
                running_loss += loss.item()
                
                
                # calculate accuracy
                correct_predictions += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                total_predictions += labels.size(0)

                # update tqdm progress bar
                train_pbar.set_postfix({"loss": round(running_loss / (i+1),5)})

        
            # now we compute the average loss and accuracy for each epoch
            epoch_accuracy = 100. * correct_predictions / total_predictions

            epoch_loss = running_loss / len(train_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            # Then we see how validation set works 
            val_loss, val_accuracy = self.validate()
            
            
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            
            # to save the metrics we got
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                
            )
            
            
            self.performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
            
        
    def validate(self):
        
        """
        The validation function is devoted for the validation set only, please consider the test function for test set.
        """
        
        val_accuracy=0.0
        val_loss=0.0
        total,correct=0,0
        
        
        val_pbar=tqdm(
            enumerate(val_loader),
            total=len(val_loader)
        )
        
        with torch.no_grad():
            
            for _, (input, labels) in val_pbar:
                
                self.model.eval()
                
                input, labels = input.to(DEVICE), labels.to(DEVICE)
                                
                output = self.model(input)
                
                loss = self.criterion(output, labels)
                
                val_loss += loss.item()
                
                total += labels.size(0)
                
                correct += torch.sum(torch.argmax(output,dim=1)==labels).item()
                
        val_accuracy = correct / total
        val_loss = val_loss / len(val_loader)
        return val_loss, val_accuracy
    
    
    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
            
if __name__ == "__main__":

    choosed_model = get_model()

    model = choosed_model(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE)

    model_trainer = Trainer(model=model)

    model_trainer.train()

    model_trainer.saveModel(path=f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth')
    
    save_cli_args(args, TRAINVAL_OUTPUT_DIR, mode='train')


