import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import logging
from utilities import PerformanceLogger
import options

parser = options.parser
args = parser.parse_args()

DEVICE = options.DEVICE
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate

class Trainer:

    def __init__(self, model, train_loader, val_loader):
        
        self.model = model
        self.model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        
        self.performance_logger = PerformanceLogger(f'{TRAINVAL_OUTPUT_DIR}')

        # logging info
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

            running_loss=0.0
            correct_predictions=0.0
            total_predictions=0
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

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

            epoch_loss = running_loss / len(self.train_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(self.train_loader)}, Training Accuracy: {epoch_accuracy}"
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
            enumerate(self.val_loader),
            total=len(self.val_loader)
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
        val_loss = val_loss / len(self.val_loader)
        return val_loss, val_accuracy
    
    
    def saveModel(self, path):

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")