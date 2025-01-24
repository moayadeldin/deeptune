from models.resnet import adjustedResNet
from models.resnet_peft import adjustedPeftResNet
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
import pandas as pd 
import logging
import argparse
from utilities import PerformanceLogger

##### SEED IS IMPORTANT TO ENSURE REPRODUCABILITY #####
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")


def get_model(model_name):

    """Allows the user to choose from Adjusted ResNet18 or PEFT-ResNet18 versions.
    """

    if model_name == "resnet18":
        model = importlib.import_module('models.resnet')
        return model.adjustedResNet

    elif model_name == "peft-resnet18":

        model = importlib.import_module('models.resnet_peft')
        return model.adjustedPeftResNet

    else:
        raise ValueError('Please Use Either ResNet18 or PEFT-ResNet18')


# Arguments constructed for CLI

parser.add_argument('--model', choices=['peft-resnet18', 'resnet18'], help="Choose the Model you want to use.")
parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
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
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
CHECK_VAL_EVERY_N_EPOCH = 1

# WE load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
df = pd.read_parquet(INPUT_DIR)

train_data, temp_data = train_test_split(df, test_size=(1 - TRAIN_SIZE), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), random_state=42)
datasets_paths = ["train_split.parquet", "val_split.parquet", "test_split.parquet"]

for path in datasets_paths:

    if os.path.exists(path):
        continue
    else:
        train_data.to_parquet(datasets_paths[0])
        val_data.to_parquet(datasets_paths[1])
        test_data.to_parquet(datasets_paths[2])


# The current datasets loaded as dataloaders
train_dataset = ParquetImageDataset(parquet_file=datasets_paths[0], transform=transformations)
val_dataset = ParquetImageDataset(parquet_file=datasets_paths[1], transform=transformations)
test_dataset = ParquetImageDataset(parquet_file=datasets_paths[2], transform=transformations)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

class Trainer:

    def __init__(self, model):
        
        self.model = model
        self.model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        self.performance_logger = PerformanceLogger('output_directory')

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

    def train(self,train_loader=train_loader,val_loader=val_loader):

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
            epoch_loss = correct_predictions / total_predictions

            epoch_accuracy = running_loss / len(train_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            # Then we see how validation set works 
            val_loss, val_accuracy = self.evaluate()
            
            
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            
            # to save the metrics we got
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
    

    def test(self, best_model_weights_path=None):

        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')

        test_accuracy=0.0
        test_loss=0.0
        total,correct = 0,0
        
        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        with torch.no_grad():

            for _, (input, labels) in test_pbar:

                self.model.eval()
                
                inputs, labels = input.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs,labels)
                
                test_loss += loss.item()
                
                total += labels.size(0)
                correct += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                
                
        test_accuracy = ( correct / total ) * 100
        test_loss = test_loss / len(test_loader)
        print(test_accuracy, test_loss)
        
        self.logger.info(f"Test accuracy: {(test_accuracy)}%")

        save_training_metrics(test_accuracy=test_accuracy,output_dir='output_directory')
        
        save_cli_args(args, 'output_directory')
        
        

    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
            
if __name__ == "__main__":

    choosed_model = get_model(args.model)

    model = choosed_model(NUM_CLASSES)

    model_trainer = Trainer(model=model)

    model_trainer.train()

    model_trainer.saveModel(path='output_directory/model_weights.pth')

    model_trainer.test()


