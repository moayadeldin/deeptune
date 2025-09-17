from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from src.nlp.gpt2 import load_gpt2_model_offline
from helpers import save_finetuned_gpt2
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import options
import logging
from helpers import PerformanceLogger
import sys
from src.nlp.gpt2 import AdjustedGPT2Model
import time
from cli import DeepTuneVisionOptions
from utils import save_process_times
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
    MODEL_STR = 'GPT2'
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft # GPT2 doesn't support PEFT YET
    FIXED_SEED = args.fixed_seed
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate

    if FIXED_SEED:
        set_seed(FIXED_SEED)

    TRAIN_DATASET_PATH = TRAIN_PATH or ( DATA_DIR / "train_split.parquet" )
    VAL_DATASET_PATH = VAL_PATH or ( DATA_DIR / "val_split.parquet" )

    TRAINVAL_OUTPUT_DIR = (OUT / f"trainval_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/trainval_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")


    gpt_model,tokenizer = load_gpt2_model_offline()
    tokenizer.pad_token = tokenizer.eos_token

    if USE_PEFT:
        pass
    else:
        adjusted_model = AdjustedGPT2Model(gpt_model=gpt_model,freeze_backbone=FREEZE_BACKBONE)
    

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
    
    
    model_trainer = GPTrainer(adjusted_model,tokenizer, LEARNING_RATE, TRAINVAL_OUTPUT_DIR,NUM_EPOCHS,train_loader,val_loader)

    model_trainer.train()

    save_finetuned_gpt2(adjusted_model,tokenizer,output_dir=TRAINVAL_OUTPUT_DIR)
    args.save_args(TRAINVAL_OUTPUT_DIR)
    
    


# Here we construct the trainer of Multlingual BERT

class GPTrainer:

    
    def __init__(self,model,tokenizer,learning_rate, outdir, num_epochs, train_loader,val_loader):
        
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
        
        
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.outdir = outdir
        
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.performance_logger = PerformanceLogger(f'{outdir}')
        
    def train(self):
        
        total_time = 0
        epoch_times = []
        
        for epoch in range(self.num_epochs):
            
            start_time = time.time()
            
            self.model.train()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )

            for batch_idx, (encoding, labels, _) in train_pbar:
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                labels = labels.to(DEVICE)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                train_pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct_predictions / total_predictions
                })

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100. * correct_predictions / total_predictions
            
                        
            epoch_end = time.time()
            epoch_duration = epoch_end - start_time
            total_time += epoch_duration

            # record the time taken for the current epoch
            epoch_times.append({"epoch": epoch + 1, "duration_seconds": epoch_duration})

            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

            self.performance_logger.log_epoch(
                epoch=epoch + 1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy
            )

            self.performance_logger.save_to_csv(f"{self.outdir}/training_log.csv")
        # record the total training time at the end
        save_process_times(epoch_times, total_time, self.outdir,"training")

    def validate(self):
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Validation"
        )

        with torch.no_grad():
            for _, (encoding, labels, _) in val_pbar:
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = correct / total

        return avg_val_loss, val_accuracy
    
if __name__ == '__main__':

    main()

                
                