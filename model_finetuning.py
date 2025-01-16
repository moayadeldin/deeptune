from models.resnet import adjustedResNet
from models.resnet_peft import adjustedPeftResNet
import importlib
from utilities import transformations
from utilities import save_training_metrics
from utilities import save_cli_args
from models.siglip import siglipModel
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

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")

# split ratios

train_ratio=0.7
val_to_test_ratio = 1/3

def get_model(model_name):

    """Allows the user to choose from Adjusted ResNet50 or PEFT-ResNet18 versions.
    """

    if model_name == "resnet50":
        model = importlib.import_module('models.resnet')
        return model.adjustedResNet

    elif model_name == "peft-resnet18":

        model = importlib.import_module('models.resnet_peft')
        return model.adjustedPeftResNet
    
    elif model_name == "peft-siglip":
        model = siglipModel()
        return model
    
    else:
        raise ValueError('Please Use Either ResNet18 or PEFT-ResNet18')



parser.add_argument('--model', choices=['resnet50', 'peft-resnet18'], help="Choose the Model you want to use.")
parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs you wan the model to run on.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to train your model.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing training data.')

args = parser.parse_args()

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs

CHECK_VAL_EVERY_N_EPOCH = 1


df = pd.read_parquet(INPUT_DIR)

train_data, remaining_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42)

# further split for the val and test

val_data, test_data = train_test_split(remaining_data, test_size = (1-val_to_test_ratio), random_state=42)


train_file = "train_split.parquet"
val_file = "val_split.parquet"
test_file = "test_split.parquet"

train_data.to_parquet(train_file)
val_data.to_parquet(val_file)
test_data.to_parquet(test_file)

_, processor = siglipModel()

train_dataset = ParquetImageDataset(parquet_file=train_file, transform=transformations,processor=processor)
val_dataset = ParquetImageDataset(parquet_file=val_file, transform=transformations,processor=processor)
test_dataset = ParquetImageDataset(parquet_file=test_file, transform=transformations,processor=processor)

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

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []


        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

    def train(self,train_loader=train_loader,val_loader=val_loader):
    
        best_val_acc=0.0 # to save the best fitting model on the validation set

        for epoch in range(NUM_EPOCHS):

            self.model.train()

            # loss tracking metrics

            running_loss=0.0
            running_vloss=0.0
            batch_loss=0.0
            running_acc=0.0
            running_val_acc = 0
            total_val_samples=0
            total_train_samples=0

            # for imgs, labels in train_loader:
            #     print(imgs, labels)
            #     break

            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (inputs, labels) in train_pbar:

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_acc += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                total_train_samples += labels.size(0)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_loss += loss.item()

                if i % 10 == 0: # calculate average loss across every 10 batches

                    batch_loss = batch_loss / 10
                    train_pbar.set_postfix({"loss": round(batch_loss,5)})
                    batch_loss = 0.0

        
            # now we compute the average loss and accuracy for each epoch
            train_accuracy_per_epoch = running_acc / total_train_samples
            self.train_accuracies.append((epoch, train_accuracy_per_epoch))

            avg_loss = running_loss / len(train_loader)
            self.train_losses.append((epoch, avg_loss))

            # evaluating the model after certain number of epochs
            if epoch % CHECK_VAL_EVERY_N_EPOCH == 0:

                self.model.eval()

                val_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

                with torch.no_grad():

                    for i, (input,labels) in val_pbar:

                        inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                        outputs = self.model(inputs)

                        loss = self.criterion(outputs, labels)

                        running_vloss+= loss.item()

                        # compute validation accuracy for this epochmodel_weights

                        running_val_acc += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                        total_val_samples += labels.size(0)

                val_accuracy_per_epoch = running_val_acc / total_val_samples
                self.val_accuracies.append((epoch, val_accuracy_per_epoch))

                avg_vloss = running_vloss / len(val_loader)
                self.val_losses.append((epoch, avg_vloss))

                self.logger.info(
                        f"[EPOCH {epoch + 1}] Training Loss= {avg_loss} Validation Loss={avg_vloss} | Training Accuracy={train_accuracy_per_epoch} Validation Accuracy={val_accuracy_per_epoch}"
                    )

    def test(self, test_loader=test_loader, best_model_weights_path=None):

        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')

        correct = 0 # we want to know how many images in the test set was predicted correctly (matched the label) so we keep adding the results of this with each batch running with specific input.
        total_test_samples=0

        self.model.eval()

        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        with torch.no_grad():

            for i, (input, labels) in test_pbar:

                inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(inputs)

                correct+=torch.sum(torch.argmax(outputs,dim=1)==labels).item()

                total_test_samples += labels.size(0)

        self.logger.info(f"Test accuracy: {(correct / total_test_samples) * 100}%")

        save_training_metrics(self.val_accuracies, ((correct / total_test_samples) * 100), 'output_directory')

        save_cli_args(args, 'output_directory')

    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
            
if __name__ == "__main__":

    choosed_model = get_model(args.model)

    model = choosed_model(NUM_CLASSES)

    model_trainer = Trainer(model=model)

    model_trainer.train()

    model_trainer.saveModel('model_weights.pth')

    model_trainer.test()


