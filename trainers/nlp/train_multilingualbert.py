from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from src.nlp.multilingual_bert import load_nlp_bert_ml_model_offline
import importlib
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
import os,json,shutil
from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE
from utils import RunType,set_seed
from datasets.text_datasets import TextDataset


def main():

    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
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

    train(
        train_df=TRAIN_PATH,
        val_df=VAL_PATH,
        out=OUT,
        freeze_backbone=FREEZE_BACKBONE,
        use_peft=USE_PEFT,
        fixed_seed=FIXED_SEED,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        added_layers=ADDED_LAYERS,
        num_classes=NUM_CLASSES,
        embed_size=EMBED_SIZE,
        model_str=MODEL_STR,
        args=args
        
    )


def train(
        train_df: Path,
        val_df: Path,
        out: Path,
        freeze_backbone: bool,
        use_peft: bool,
        fixed_seed: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        added_layers: int,
        num_classes: int,
        embed_size: int,
        model_str: str,
        args: DeepTuneVisionOptions,

):
    
        
    if fixed_seed:
        set_seed(fixed_seed)

    TRAIN_DATASET_PATH = train_df
    VAL_DATASET_PATH = val_df

    TRAINVAL_OUTPUT_DIR = (out / f"trainval_output_{model_str}_{UNIQUE_ID}")
    
    # load the tokenizer, the dataloaders

    choosed_model = get_model(added_layers=added_layers, use_peft=use_peft, args=args)

    model = choosed_model(num_classes, added_layers, embed_size, freeze_backbone)

    _,tokenizer = load_nlp_bert_ml_model_offline()

    model.to(device=DEVICE)

    train_dataset = TextDataset(parquet_file=TRAIN_DATASET_PATH, tokenizer=tokenizer, max_length=512)
    val_dataset = TextDataset(parquet_file=VAL_DATASET_PATH, tokenizer=tokenizer, max_length=512)
            
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    model_trainer = BERTrainer(model,tokenizer,learning_rate,num_epochs,train_loader, val_loader,TRAINVAL_OUTPUT_DIR)

    model_trainer.train()

    model_config = {
        "num_classes":num_classes,
        "added_layers":added_layers,
        "embedding_layer": embed_size
    }

    output_dir = model_trainer.save_tunedbertmodel(model,tokenizer,output_dir=TRAINVAL_OUTPUT_DIR,model_config=model_config)
    args.save_args(output_dir)

    return output_dir



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
        
        self.total_time = 0
        self.epoch_times = []
        
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
                self.total_time += epoch_duration

            # record the time taken for the current epoch
            self.epoch_times.append({"epoch": epoch + 1, "Total Duration of the Whole Program": epoch_duration})

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
    
    def save_tunedbertmodel(self,model,tokenizer,output_dir,model_config):
    
        """
        Save the BERT model after we finetune it.
        
        Args:
            model (CustomMultilingualBERT): The finetuned model.
            tokenizer (BertTokenizer): The tokenizer used for the model.
            output_dir (str): The path to save the model.
            model_config (dict): The configuration of different adjustments the user had to the model (e.g, the number of added layers, the embedding layer size, the number of classes in nue dataset).
        """
        
        try:
        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            dir_path = os.path.abspath(output_dir)

                
            # save model state dic
            torch.save(model.state_dict(), os.path.join(dir_path,'model_weights.pth'))
            print(f"Model weights saved to {os.path.join(dir_path, 'model_weights.pth')}")
            
            # save underlying BERT model
            model.bert.save_pretrained(os.path.join(dir_path, "bert_model"))
            print(f"BERT model saved to {os.path.join(dir_path, 'bert_model')}")
            
            # save tokenizer
            tokenizer.save_pretrained(os.path.join(dir_path, "tokenizer"))
            print(f"Tokenizer saved to {os.path.join(dir_path, 'tokenizer')}")
            
            # save model's layers
            with open(os.path.join(dir_path, "model_config.json"), "w") as f:
                json.dump(model_config, f)
            print(f"Model configuration saved to {os.path.join(dir_path, 'model_config.json')}")

                            
            self.performance_logger.save_to_csv(f"{dir_path}/training_log.csv")
        
            # record the total training time at the end
            save_process_times(self.epoch_times, self.total_time, dir_path,"training")

            return dir_path
        
        except Exception as e:
            print(f"There is an Error while saving model: {e}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

                
def get_model(added_layers,use_peft,args):

    """
    Allows the user to choose from Adjusted BERT or PEFT-Adjusted BERT versions.
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
