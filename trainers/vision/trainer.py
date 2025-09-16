from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import logging
from helpers import PerformanceLogger
from options import DEVICE, TRAINVAL_OUTPUT_DIR
import time
from utils import save_process_times
import functools
print = functools.partial(print, flush=True)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        mode: str,
        num_epochs: int,
        output_dir: Path = TRAINVAL_OUTPUT_DIR
    ):
        """
        Performs Training & Validation on the input image dataset.
        
        Args:
        
            model (PyTorch Model): The model we are loading from the src file, whether it is for transfer learning with PEFT Or without.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): The DataLoader for the validation set.
            learning_rate (float): The learning rate for the optimizer.
            mode (str): The mode of the model, either classification or regression.
            num_epochs (int): The number of epochs to train the model.
            output_dir (str): The directory to save the training logs and model checkpoints.
            
        Attributes:
        
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            performance_logger (PerformanceLogger): Logger instance for tracking training and validation progress.
            logger (logging.Logger): Logger instance for tracking training progress.
        """
        
        self.model = model
        self.model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        if self.mode == 'cls':    
            self.criterion = nn.CrossEntropyLoss()
        else: # then regression if not classification
            self.criterion = nn.MSELoss()
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if not params:
            raise ValueError("No trainable parameters found in the model!")
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
        self.performance_logger = PerformanceLogger(output_dir) if output_dir else None

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

    def train(self):
                    
        total_time = 0
        epoch_times = []
        for epoch in range(self.num_epochs):

            self.model.train()
            epoch_start = time.time()
            
            
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
                
                if isinstance(self.criterion, nn.MSELoss): # if regression then we must rehape the target tensor    
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    labels = labels.view(-1,1).float() # [batch_size,1]
                else:
                    inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
                    

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
                
                if self.mode == "cls":
                
                    # calculate accuracy
                    correct_predictions += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                    total_predictions += labels.size(0)

                    # now we compute the average loss and accuracy for each epoch
                    epoch_accuracy = 100. * correct_predictions / total_predictions
                    
                # update tqdm progress bar
                train_pbar.set_postfix({"loss": round(running_loss / (i+1),5)})
            # update training loss
            epoch_loss = running_loss / len(self.train_loader)
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            total_time += epoch_duration

            # record the time taken for the current epoch
            epoch_times.append({"epoch": epoch + 1, "duration_seconds": epoch_duration})
        
            if self.mode == 'cls':
                
                """
                Here in Classification we have both loss and accuracy as metrics to track, so we will log both of them.
                """
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {running_loss / len(self.train_loader)}, Training Accuracy: {epoch_accuracy}"
                )    
                # Then we see how validation set works 
                val_loss, val_accuracy = self.validate()
                self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            
            else:
                """
                Only loss is needed to be logged in regression mode.
                """
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {running_loss / len(self.train_loader)}"
                )    
    
                val_loss = self.validate()    
            
            self.logger.info(f"Validation loss: {val_loss}")
            if self.performance_logger:

                if self.mode == 'cls':
                    
                    self.performance_logger.log_epoch(
                        epoch = epoch+1,
                        epoch_loss=epoch_loss,
                        epoch_accuracy=epoch_accuracy,
                        val_loss=val_loss,
                        val_accuracy=val_accuracy,
                        
                    )
            
                else:
                    
                    self.performance_logger.log_epoch(
                        epoch = epoch+1,
                        epoch_loss=epoch_loss,
                        val_loss=val_loss,
                        epoch_accuracy='Regression: No Accuracy',
                        val_accuracy = 'Regression: No Accuracy'
                        
                    )    
                
                self.performance_logger.save_to_csv(f"{self.output_dir}/training_log.csv")
        # record the total training time at the end
        save_process_times(epoch_times, total_time, self.output_dir,"training")
        
    def validate(self):
        
        """
        The validation function is devoted for the validation set only, please consider the test function for test set.
        """
        # initialize the metrics for validation
        val_accuracy=0.0
        val_loss=0.0
        total,correct=0,0
        
        
        val_pbar=tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader)
        )
        
        with torch.no_grad():
            
            for _, (input, labels) in val_pbar:
                
                # set the model to evaluation mode
                self.model.eval()
                
                input, labels = input.to(DEVICE), labels.to(DEVICE)
                                
                # apply forward pass and accumulate loss                
                
                output = self.model(input)
                
                loss = self.criterion(output, labels)
                
                val_loss += loss.item()
                
                # calculate accuracy
                total += labels.size(0)
                
                correct += torch.sum(torch.argmax(output,dim=1)==labels).item()
        
        # if mode regression then no need to return accuracy or compute it
        
        if self.mode == 'cls':
            val_accuracy = correct / total
            val_loss = val_loss / len(self.val_loader)
            return val_loss, val_accuracy
        else:
            val_loss = val_loss / len(self.val_loader)
            return val_loss
            
    
    def saveModel(self, path):
        
        """
        We save the model to the path specified by the user.
        
        Args:
            path (str): The path to save the model to.
        """

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")