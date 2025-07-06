
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datasets.image_datasets import ParquetImageDataset
from src.vision.siglip import load_siglip_offline
from src.vision.siglip_peft import CustomSigLIPWithPeft, load_peft_siglip_offline
from utilities import PerformanceLogger,save_cli_args,get_args,split_save_load_dataset,fixed_seed
import warnings
from src.vision.siglip import load_custom_siglip_model
from tqdm import tqdm
from pathlib import Path
import sys
import logging
import options
import os
import json


parser = options.parser
args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments constructed for CLI

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
USE_PEFT = args.use_peft
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FIXED_SEED = args.fixed_seed
FREEZE_BACKBONE = args.freeze_backbone

# If we want to apply fixed seed or randomly initialize the weights and dataset.
if FIXED_SEED:
    SEED=42
    fixed_seed(SEED)
else:
    SEED = np.random.randint(low=0, high=1000)
    fixed_seed(SEED)
    warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)



if FIXED_SEED:
    seed=42
    args.fixed_seed = 42
else:
    seed = np.random.randint(low=0, high=1000)
    args.fixed_seed = seed

TRAINVAL_OUTPUT_DIR = Path(__file__).parent.parent / 'output_directory_trainval'

if USE_PEFT:
        
    model, processor = load_peft_siglip_offline(added_layers=ADDED_LAYERS,embedding_layer=EMBED_SIZE, freeze_backbone=FREEZE_BACKBONE,num_classes=NUM_CLASSES)
    args.model = 'PEFT-SIGLIP'
    
else:
    
    model = load_custom_siglip_model(
        added_layers=ADDED_LAYERS,
        embedding_dim=EMBED_SIZE,
        num_classes=NUM_CLASSES,
        freeze_backbone=FREEZE_BACKBONE
    )
    _, processor = load_siglip_offline()
    args.model = 'SIGLIP'
    


TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
VAL_DATASET_PATH = options.VAL_DATASET_PATH
TEST_DATASET_PATH = options.TEST_DATASET_PATH
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR


# We load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.


train_loader, val_loader = split_save_load_dataset(
    
    mode='train',
    type='image',
    input_dir= INPUT_DIR,
    train_size = TRAIN_SIZE,
    val_size = VAL_SIZE,
    test_size = TEST_SIZE,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
    seed=SEED,
    batch_size=BATCH_SIZE,
    tokenizer=processor,
    siglip=True
)           


class Trainer:
    def __init__(self, model=model):
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        self.performance_logger = PerformanceLogger(f'{TRAINVAL_OUTPUT_DIR}')

        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
    
    def train(self):

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            
            """
            The following arguments are as follows:
            
            - running_loss: indicating the loss during the epoch for train and val datasets respectively.
            - correct_predictions: number of predictions that equal true label 
            - total_predictions: number of predictions made in total
            """
            
            running_loss = 0.0
            correct_predictions = 0.0
            total_predictions = 0.0
            train_pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
            )

            for batch_idx, (pixel_values, labels) in train_pbar:
                # make sure the inputs and labels on GPU
                pixel_values = pixel_values.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # null gradient

                self.optimizer.zero_grad()
            
                logits = self.model({'pixel_values': pixel_values}) 
            
                # compute loss
                loss = self.criterion(logits, labels)
                
                # backprop and apply optimizer
                loss.backward()
                self.optimizer.step()
                
                
                # accumulate loss
                running_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(logits, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                    
                # update tqdm progress bar
                train_pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                })
                
            # now we compute average loss and accuracy for each epoch
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100. * correct_predictions / total_predictions
                    
            
            self.logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            save_cli_args(args, TRAINVAL_OUTPUT_DIR,mode='train')
            
            
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                
            )
            
            
        self.performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
        
        if USE_PEFT:
            
            torch.save(self.model.state_dict(), TRAINVAL_OUTPUT_DIR / "custom_siglip_model.pt")
            print(f"Siglip model saved to {os.path.join(TRAINVAL_OUTPUT_DIR, 'PEFT_SIGLIP_model')}")
            config = {
            "added_layers": ADDED_LAYERS,
            "embedding_dim": EMBED_SIZE,
            "num_classes": NUM_CLASSES
            }
        
        else:
            torch.save(self.model.state_dict(), TRAINVAL_OUTPUT_DIR / "custom_siglip_model.pt")
            print(f"Siglip model saved to {os.path.join(TRAINVAL_OUTPUT_DIR, 'SIGLIP_model')}")
            config = {
            "added_layers": ADDED_LAYERS,
            "embedding_dim": EMBED_SIZE,
            "num_classes": NUM_CLASSES
            }

            with open(TRAINVAL_OUTPUT_DIR / "custom_siglip_config.json", "w") as f:
                json.dump(config, f)
                    
    def validate(self):
        
        """
        The evaluation function is devoted for the validation set only, please consider the test function for test set.
        """
        
        val_accuracy=0.0
        val_loss=0.0
        total, correct= 0,0
        
        val_pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        )
        
        with torch.no_grad():
            for _, (image_pixels, labels) in val_pbar:
                
                self.model.eval()
                
                image_pixels,labels = image_pixels.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()

                logits = self.model({'pixel_values': image_pixels}) 

                loss = self.criterion(logits, labels)
                
                val_loss += loss.item()
                
                # calculate accuracy
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
        
        val_accuracy = correct / total
        val_loss = val_loss / len(val_loader)
        return val_loss, val_accuracy
        

if __name__ == '__main__':

        
    model_trainer = Trainer()
    model_trainer.train()