import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import ParquetImageDataset
from models.siglip import load_siglip_offline
from models.siglip import SIGLIP_PEFT_ADAPTER
from peft import LoraConfig, get_peft_model
from utilities import PerformanceLogger
from tqdm import tqdm
import sys
import logging
from utilities import save_cli_args, save_training_metrics

##### SEED IS IMPORTANT TO ENSURE REPRODUCABILITY #####
torch.manual_seed(42)
parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments constructed for CLI
parser.add_argument('--model', type=str, required=False, default='peft-siglip', help='Model type.')
parser.add_argument('--num_classes', type=str, required=False, default='None', help='Number of classes for your original dataset.')
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs you wan the model to run on.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to train your model.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--train_size', type=float, required=True, help='Mention the split ratio of the Train Dataset')
parser.add_argument('--val_size', type=float, required=True, help='Mention the split ratio of the Val Dataset')
parser.add_argument('--test_size', type=float, required=True, help='Mention the split ratio of the Test Dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing training data.')


args = parser.parse_args()

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size


# We load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
datasets_paths = ["train_split.parquet", "val_split.parquet", "test_split.parquet"]

df = pd.read_parquet(INPUT_DIR)
train_data, temp_data = train_test_split(df, test_size=(1 - TRAIN_SIZE), random_state=42)

val_data, test_data = train_test_split(temp_data, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), random_state=42) # random_state is very important to produce the same dataset everytime.

base_model, processor = load_siglip_offline()

for path in datasets_paths:

    if os.path.exists(path):
        continue
    else:
        train_data.to_parquet(datasets_paths[0])
        val_data.to_parquet(datasets_paths[1])
        test_data.to_parquet(datasets_paths[2])


# The current datasets loaded as dataloaders
train_dataset = ParquetImageDataset(parquet_file=datasets_paths[0], transform=None, processor=processor)
val_dataset = ParquetImageDataset(parquet_file=datasets_paths[1], transform=None, processor=processor)
test_dataset = ParquetImageDataset(parquet_file=datasets_paths[2], transform=None, processor=processor)


# Create DataLoaders  
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

# Get the PEFT model.

for param in base_model.parameters():
    param.requires_grad = False

peft_config = LoraConfig(
        inference_mode=False,  # Enable training
        r=16,                  # Low-rank dimension
        lora_alpha=32,         # Scaling factor
        lora_dropout=0.1,      # Dropout
        target_modules=[
            # "k_proj",
            "v_proj",
            "q_proj",
            # "out_proj",
        ]
    )

# Wrap the base model with the PEFT model
peft_model = get_peft_model(base_model, peft_config)
# print_trainable_parameters(peft_model)

num_samples = len(train_loader.dataset)
print(f"Number of samples: {num_samples}")


class Trainer:
    def __init__(self, model=peft_model):
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        self.performance_logger = PerformanceLogger('output_directory')

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
            
            val_loss, val_accuracy = self.evaluate()
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            
            
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                
            )
            
            
        self.performance_logger.save_to_csv("output_directory/training_log.csv")
            
    def evaluate(self):
        
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
    
    
    def test(self):
        
        test_accuracy=0.0
        test_loss=0.0
        total, correct= 0,0
        
        test_pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        )
        
        with torch.no_grad():
            for _, (image_pixels, labels) in test_pbar:
                
                self.model.eval()
                
                image_pixels,labels = image_pixels.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()

                outputs = self.model.vision_model(pixel_values=image_pixels)
                
                logits = outputs.pooler_output  

                loss = self.criterion(logits, labels)
                
                test_loss += loss.item()
                
                # calculate accuracy
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
        
        test_accuracy = correct / total
        test_loss = test_loss / len(test_loader)
        print(test_accuracy,test_loss)
        save_training_metrics(test_accuracy=test_accuracy,output_dir='output_directory')
        save_cli_args(args, 'output_directory')
        return test_loss, test_accuracy
        

if __name__ == '__main__':

        
    trainr = Trainer()
    trainr.train()
    trainr.test()