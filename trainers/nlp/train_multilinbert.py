from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from src.nlp.multilingual_bert import load_nlp_bert_ml_model_offline
import importlib
from helpers import save_finetunedbertmodel
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import options
import logging
from helpers import PerformanceLogger
import sys
import time
from utils import save_process_times

from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls,RunType,set_seed
from datasets.text_datasets import TextDataset

# Initialize the needed variables either from the CLI user sents or from the device.

def main():

    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    DATA_DIR: Path = args.input_dir
    MODE = args.mode
    OUT = args.out
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft
    MODEL_STR = 'PEFT-BERT' if USE_PEFT else 'BERT'
    FIXED_SEED = args.fixed_seed
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    ADDED_LAYERS = args.added_layers
    NUM_CLASSES = args.num_classes
    EMBED_SIZE = args.embed_size
    
    if FIXED_SEED:
        set_seed(FIXED_SEED)

    TRAIN_DATASET_PATH = TRAIN_PATH or ( DATA_DIR / "train_split.parquet" )
    VAL_DATASET_PATH = VAL_PATH or ( DATA_DIR / "val_split.parquet" )

    TRAINVAL_OUTPUT_DIR = (OUT / f"trainval_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/trainval_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")
    
    # load the tokenizer, the dataloaders

    choosed_model = get_model(added_layers=ADDED_LAYERS, use_peft=USE_PEFT, args=args)

    model = choosed_model(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE)

    _,tokenizer = load_nlp_bert_ml_model_offline()

    model.to(device=DEVICE)

    train_dataset = TextDataset(parquet_file=TRAIN_DATASET_PATH, tokenizer=tokenizer, max_length=512)
    val_dataset = TextDataset(parquet_file=VAL_DATASET_PATH, tokenizer=tokenizer, max_length=512)
            
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
    
    model_trainer = BERTrainer(model,tokenizer,LEARNING_RATE,NUM_EPOCHS,train_loader, val_loader,TRAINVAL_OUTPUT_DIR)

    model_trainer.train()

    model_config = {
        "num_classes":NUM_CLASSES,
        "added_layers":ADDED_LAYERS,
        "embedding_layer": EMBED_SIZE
    }

    save_finetunedbertmodel(model,tokenizer,output_dir=TRAINVAL_OUTPUT_DIR,model_config=model_config)
    args.save_args(TRAINVAL_OUTPUT_DIR)




# Here we construct the trainer of Multlingual BERT

class BERTrainer:
    def __init__(self,model,tokenizer, learning_rate,num_epochs,train_loader,val_loader, trainval_output_dir):
        
        """
        Performs Training & Validation on the input text dataset.
        
        Args:
        
            model (HuggingFace Model): The NLP BERT model we are loading from the src file.
            tokenizer (HuggingFace Tokenizer): The NLP BERT model we are loading from load_nlp_bert_ml_model_offline() function in utilities file.
            
        Attributes:
        
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            optimizer (torch.optim.Optimizer): Adam optimizer for updating the model weights during training.
            logger (logging.Logger): Logger instance for tracking training progress.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.trainval_output_dir = trainval_output_dir
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.tokenizer = tokenizer
        
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.performance_logger = PerformanceLogger(f'{self.trainval_output_dir}')
        
        

    def train(self):
        
        total_time = 0
        epoch_times = []
        
        for epoch in range(self.num_epochs):
            
            # set the model to training mode
            self.model.train()
            
            epoch_start = time.time()
            
            # initialize metrics tracking values for training
            running_loss = 0.0
            correct_predictions = 0.0
            total_predictions = 0.0
            
            # initlaize progress bar
            train_pbar = tqdm(
                self.train_loader,
                total = len(self.train_loader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            # iterate over the training dataset
            for batch_idx, (encoding,labels, *_) in enumerate(train_pbar):
                
                # move the input to GPU 
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                token_type_ids = encoding.get('token_type_ids',None)
                
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                
                labels = labels.to(DEVICE)
                
                # null the gradient
                self.optimizer.zero_grad()
                #run the model
                outputs = self.model(input_ids,attention_mask,token_type_ids)
                
                # compute the loss
                loss = self.criterion(outputs,labels)
                
                # backprop and apply optimizer
                loss.backward()
                self.optimizer.step()
                
                # accumulate loss
                running_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs,1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # update tqdm progress bar
                train_pbar.set_postfix({
                        'loss': running_loss / (batch_idx + 1)
                    })
                
                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = 100. * correct_predictions / total_predictions
                
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                total_time += epoch_duration

            # record the time taken for the current epoch
            epoch_times.append({"epoch": epoch + 1, "duration_seconds": epoch_duration})

            # update the logger
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {running_loss / len(self.train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            # apply validation after each epoch on training set
                
            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
                
            # update the performance logger
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy
                
            )
                
                
            self.performance_logger.save_to_csv(f"{self.trainval_output_dir}/training_log.csv")
    
        # record the total training time at the end
        save_process_times(epoch_times, total_time, self.trainval_output_dir,"training")
            
    def validate(self):
        
        
        # initialize the metrics for validation
        val_accuracy=0.0
        val_loss = 0.0
        total,correct=0,0
        
        
        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader)
        )
        
        with torch.no_grad():
            
            for _, (encoding,labels, *_) in val_pbar:
                
                # set the model to evaluation mode
                self.model.eval()
                
                # move the input to GPU
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                token_type_ids = encoding.get('token_type_ids',None)
                
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                
                labels = labels.to(DEVICE)
                
                # apply forward pass and accumulate loss
                
                outputs = self.model(input_ids,attention_mask,token_type_ids)
                
                loss = self.criterion(outputs,labels)
                
                val_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
        val_accuracy = correct / total
        val_loss = val_loss / len(self.val_loader)
        return val_loss, val_accuracy
                
def get_model(added_layers,use_peft,args):

    """
    Allows the user to choose from Adjusted ResNet18 or PEFT-ResNet18 versions.
    """
    if added_layers == 0:
        
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    if use_peft:
        
        model = importlib.import_module('src.nlp.multilingual_bert_peft')
        args.model = 'Multilingual BERT PEFT'
        return model.CustomMultilingualPeftBERT
    else:
        model = importlib.import_module('src.nlp.multilingual_bert')
        args.model = 'Multilingual BERT'
        return model.CustomMultilingualBERT
            
if __name__ == '__main__':

    main()
