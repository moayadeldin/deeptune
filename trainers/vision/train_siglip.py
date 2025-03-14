
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datasets.image_datasets import ParquetImageDataset
from src.vision.siglip import load_siglip_offline
from src.vision.siglip_peft import SIGLIP_PEFT_TRAINED
from src.vision.siglip_peft import CustomSigLIPWithPeft, load_peft_siglip_offline
from src.vision.siglip import SIGLIP_TRAINED
from utilities import PerformanceLogger
from tqdm import tqdm
from pathlib import Path
import sys
import logging
from utilities import save_cli_args, save_training_metrics


parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments constructed for CLI

parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs you wan the model to run on.')
parser.add_argument('--added_layers', type=int, choices=[1,2], required=False, help='Specify the number of layers you want to add.')
parser.add_argument('--embed_size', type=int, required=False, help='Specify the size of the embeddings you would obtain through embedding layer.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to train your model.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--train_size', type=float, required=True, help='Mention the split ratio of the Train Dataset')
parser.add_argument('--val_size', type=float, required=True, help='Mention the split ratio of the Val Dataset')
parser.add_argument('--test_size', type=float, required=True, help='Mention the split ratio of the Test Dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing training data.')
parser.add_argument('--num_classes', type=int, required=False, help='Number of classes for your original dataset.')
parser.add_argument('--fixed-seed', action='store_true', help='Choose whether a seed is required or not.')
parser.add_argument('--freeze-backbone', action='store_true', help='Decide whether you want to freeze backbone or not.')
parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')

args = parser.parse_args()

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

if FIXED_SEED:
    seed=42
    args.fixed_seed = 42
else:
    seed = np.random.randint(low=0)
    args.fixed_seed = seed

TRAINVAL_OUTPUT_DIR = Path(__file__).parent.parent / 'output_directory_trainval'

if USE_PEFT:
        
    model, processor = load_peft_siglip_offline(added_layers=ADDED_LAYERS,embedding_layer=EMBED_SIZE, freeze_backbone=FREEZE_BACKBONE,num_classes=NUM_CLASSES)
    args.model = 'PEFT-SIGLIP'
    
else:
    
    model,processor = load_siglip_offline()
    args.model = 'SIGLIP'    
    


TRAIN_DATASET_PATH = Path(__file__).parent.parent / "train_split.parquet"
VAL_DATASET_PATH = Path(__file__).parent.parent / "val_split.parquet"
TEST_DATASET_PATH = Path(__file__).parent.parent / "test_split.parquet"


# We load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.

df = pd.read_parquet(INPUT_DIR)
df = df[:10]
train_data, temp_data = train_test_split(df, test_size=(1 - TRAIN_SIZE), random_state=42)

val_data, test_data = train_test_split(temp_data, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), random_state=42) # random_state is very important to produce the same dataset everytime.

train_data.to_parquet(TRAIN_DATASET_PATH, index=False)
val_data.to_parquet(VAL_DATASET_PATH, index=False)
test_data.to_parquet(TEST_DATASET_PATH, index=False)

print("Data splits have been saved and overwritten if they existed.")


# The current datasets loaded as dataloaders
train_dataset = ParquetImageDataset(parquet_file=TRAIN_DATASET_PATH, transform=None, processor=processor)
val_dataset = ParquetImageDataset(parquet_file=VAL_DATASET_PATH, transform=None, processor=processor)
test_dataset = ParquetImageDataset(parquet_file=TEST_DATASET_PATH, transform=None, processor=processor)


# Create DataLoaders  
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
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
                
                # Forward pass with pixel values argument very important
                if isinstance(self.model, CustomSigLIPWithPeft):
                    outputs = self.model.base_model.base_model.vision_model(pixel_values=pixel_values)
                    pooled_output = outputs.pooler_output
                    logits = self.model.fc_layers(pooled_output)  # Apply the classification head
                else:
                    outputs = self.model.vision_model(pixel_values=pixel_values)
                    logits = outputs.pooler_output
                
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
            
            
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                
            )
            
            
        self.performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
        
        if USE_PEFT:
        
            self.model = self.model.base_model.base_model.merge_and_unload()
            self.model.save_pretrained(SIGLIP_PEFT_TRAINED)
            print(f"Siglip model saved to {SIGLIP_PEFT_TRAINED}")
        
        else:
            self.model = self.model.merge_and_unload()
            self.model.save_pretrained(SIGLIP_TRAINED)
            print(f"Siglip model saved to {SIGLIP_TRAINED}")
                
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

                if isinstance(self.model, CustomSigLIPWithPeft):
                    outputs = self.model.base_model.base_model.vision_model(pixel_values=image_pixels)
                else:
                    outputs = self.model.vision_model(pixel_values=image_pixels)
                
                logits = outputs.pooler_output  

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