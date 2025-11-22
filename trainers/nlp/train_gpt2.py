from src.nlp.gpt2 import load_gpt2_model_offline
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import logging
from helpers import PerformanceLogger
import sys
from src.nlp.gpt2 import AdjustedGPT2Model
import time
from cli import DeepTuneVisionOptions
from utils import save_process_times
import shutil
from pathlib import Path
from options import UNIQUE_ID, DEVICE
from utils import RunType,set_seed
import json, os
from datasets.text_datasets import TextDataset


def main():

    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    OUT = args.out
    MODEL_STR = 'GPT2'
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft # GPT2 doesn't support PEFT YET
    FIXED_SEED = args.fixed_seed
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate


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
        model_str: str,
        args: DeepTuneVisionOptions,

):
    if use_peft:
        raise ValueError("PEFT is not supported for GPT2 yet.")
    
    if fixed_seed:
        set_seed(fixed_seed)

    TRAIN_DATASET_PATH = train_df
    VAL_DATASET_PATH = val_df

    TRAINVAL_OUTPUT_DIR = (out / f"trainval_output_{model_str}_{UNIQUE_ID}")


    gpt_model,tokenizer = load_gpt2_model_offline()
    tokenizer.pad_token = tokenizer.eos_token

    if use_peft:
        pass
    else:
        adjusted_model = AdjustedGPT2Model(gpt_model=gpt_model,freeze_backbone=freeze_backbone)
    

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
    
    model_trainer = GPTrainer(adjusted_model,tokenizer, learning_rate, TRAINVAL_OUTPUT_DIR,num_epochs,train_loader,val_loader)

    model_trainer.train()

    output_dir = model_trainer.save_tunedgpt2model(model=adjusted_model,tokenizer=tokenizer,output_dir=TRAINVAL_OUTPUT_DIR, args=args)
    
    return output_dir


# Here we construct the trainer of Multlingual GPT2

class GPTrainer:

    
    def __init__(self,model,tokenizer,learning_rate, outdir, num_epochs, train_loader,val_loader):
        
        """
        Performs Training & Validation on the input text dataset.
        
        Args:
        
            model (HuggingFace Model): The NLP gpt2 model we are loading from the src file.
            tokenizer (HuggingFace Tokenizer): The NLP gpt2 model we are loading from load_nlp_gpt2_ml_model_offline() function in utilities file.
            
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
        
        self.total_time = 0
        self.epoch_times = []
        
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

            for batch_idx, (encoding, labels, *_) in train_pbar:
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
            self.total_time += epoch_duration

            # record the time taken for the current epoch
            self.epoch_times.append({"epoch": epoch + 1, "Duration": epoch_duration})

            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.3f}%")

            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss:.4f}, Accuracy: {val_accuracy:.3f}%")

            self.performance_logger.log_epoch(
                epoch=epoch + 1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy
            )

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
            for _, (encoding, labels, *_) in val_pbar:
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
        val_accuracy = 100. * ( correct / total )

        return avg_val_loss, val_accuracy
    
    def save_tunedgpt2model(self,model,tokenizer,output_dir,args,output_dim=1000):
    
        """
        Save the gpt2 model after we finetune it.
        
        Args:
            model (CustomMultilingualgpt2): The finetuned model.
            tokenizer (gpt2Tokenizer): The tokenizer used for the model.
            output_dir (str): The path to save the model.
        """
        
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            dir_path = os.path.abspath(output_dir)

            # Save model weights
            torch.save(model.state_dict(), os.path.join(dir_path, "model_weights.pth"))
            print(f"Saved model weights to {os.path.join(dir_path, 'model_weights.pth')}")

            # Save GPT-2 backbone separately
            model.gpt2.save_pretrained(os.path.join(dir_path, "gpt2_model"))
            tokenizer.save_pretrained(os.path.join(dir_path, "tokenizer"))
            print(f"Saved GPT-2 backbone and tokenizer to {dir_path}")

            # Save config for reproducibility
            config = {"output_dim": output_dim}
            with open(os.path.join(dir_path, "model_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            print(f"Saved model config to {os.path.join(dir_path)}")

            args.save_args(dir_path)           
                
            self.performance_logger.save_to_csv(f"{dir_path}/training_log.csv")

            save_process_times(self.epoch_times, self.total_time, dir_path,"training")

            return dir_path

        except Exception as e:

            print(f"There is an Error while saving model: {e}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
        
    
if __name__ == '__main__':

    main()

                
                