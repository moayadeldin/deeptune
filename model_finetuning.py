from models.resnet import adjustedResNet
from utilities import transformations
import os
import torch
from dataset import ImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import logging
import argparse

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Fine-tune the model passing your Hyperparameters, train, val, and test directories.")

parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--num_epochs', type=int, required=True, help='The number of epochs you wan the model to run on.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size to train your model.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training data.')
parser.add_argument('--val_dir', type=str, required=True, help='Directory containing validation data.')
parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data.')

args = parser.parse_args()

TRAIN_DIR = args.train_dir
VAL_DIR = args.val_dir
TEST_DIR = args.test_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs

CHECK_VAL_EVERY_N_EPOCH = 1

train_dataset = ImageDataset(root_dir=TRAIN_DIR, transform=transformations)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_dataset = ImageDataset(root_dir=VAL_DIR, transform=transformations)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

test_dataset = ImageDataset(root_dir=TEST_DIR, transform=transformations)
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

            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (inputs, labels) in train_pbar:

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_acc += self.computeAccuracy(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_loss += loss.item()

                if i % 10 == 0: # calculate average loss across every 10 batches

                    batch_loss = batch_loss / 10
                    train_pbar.set_postfix({"loss": round(batch_loss,5)})
                    batch_loss = 0.0

        
            # now we compute the average loss and accuracy for each epoch
            train_accuracy_per_epoch = running_acc / len(train_loader)
            self.train_accuracies.append((epoch, train_accuracy_per_epoch.cpu()))

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

                        running_val_acc += self.computeAccuracy(outputs,labels)

                val_accuracy_per_epoch = running_val_acc / len(val_loader)
                self.val_accuracies.append((epoch, val_accuracy_per_epoch.cpu()))

                avg_vloss = running_vloss / len(val_loader)
                self.val_losses.append((epoch, avg_vloss))

                self.logger.info(
                        f"[EPOCH {epoch + 1}] Training Loss= {avg_loss} Validation Loss={avg_vloss} | Training Accuracy={train_accuracy_per_epoch} val={val_accuracy_per_epoch}"
                    )

    def test(self, test_loader=test_loader, best_model_weights_path=None):

        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')

        correct = 0 # we want to know how many images in the test set was predicted correctly (matched the label) so we keep adding the results of this with each batch running with specific input.

        self.model.eval()

        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        with torch.no_grad():

            for i, (input, labels) in test_pbar:

                inputs, labels = input.to(DEVICE), labels.to(DEVICE)

                outputs = self.model(inputs)

                correct+=self.computeAccuracy(outputs,labels)

        self.logger.info(f"Test accuracy: {(correct / len(test_loader)) * 100}%")

    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
            

    def computeAccuracy(self,outputs, labels):

        """Compute accuracy given outputs as logits.
        """

        preds = torch.argmax(outputs, dim=1)
        return torch.sum(preds == labels) / len(preds)

if __name__ == "__main__":

    model = adjustedResNet(NUM_CLASSES)

    model_trainer = Trainer(model=model)

    model_trainer.train()

    model_trainer.saveModel('model_weights.pth')

    model_trainer.test()


