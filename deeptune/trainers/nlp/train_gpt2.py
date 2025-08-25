import importlib
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import logging
import sys

import deeptune.options as options
from deeptune.src.nlp.multilingual_bert import CustomMultilingualBERT
from deeptune.src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
from deeptune.src.nlp.multilingual_bert import load_nlp_bert_ml_model_offline
from deeptune.src.nlp.gpt2 import load_gpt2_model_offline
from deeptune.utilities import save_cli_args, fixed_seed,split_save_load_dataset,save_finetuned_gpt2
from deeptune.utilities import PerformanceLogger,get_args
from deeptune.src.nlp.gpt2 import AdjustedGPT2Model

# Initialize the needed variables either from the CLI user sents or from the device.

DEVICE = options.DEVICE
# parser = options.parser
args = get_args()

INPUT_DIR = args.input_dir
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
CHECK_VAL_EVERY_N_EPOCH = 1
USE_PEFT = args.use_peft
FIXED_SEED = args.fixed_seed
FREEZE_BACKBONE = args.freeze_backbone
    
TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
VAL_DATASET_PATH = options.VAL_DATASET_PATH
TEST_DATASET_PATH = options.TEST_DATASET_PATH
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR

# If we want to apply fixed seed or randomly initialize the weights and dataset.
if FIXED_SEED:
    SEED=42
    fixed_seed(SEED)
else:
    SEED = np.random.randint(low=0, high=1000)
    fixed_seed(SEED)
    warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)


gpt_model,tokenizer = load_gpt2_model_offline()
tokenizer.pad_token = tokenizer.eos_token
# Fetch whether the transfer-learning with PEFT version or transfer-learning without
def get_model():
    """
    Allows the user to choose from Adjusted Fine-tuned version of model or PEFT-tuned version.
    """

    if USE_PEFT:
        pass
    else:
        adjusted_model = AdjustedGPT2Model(gpt_model=gpt_model,freeze_backbone=FREEZE_BACKBONE)
        return adjusted_model


train_loader, val_loader = split_save_load_dataset(
    
    mode='train',
    type='text',
    input_dir= INPUT_DIR,
    train_size = TRAIN_SIZE,
    val_size = VAL_SIZE,
    test_size = TEST_SIZE,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
    seed=SEED,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)

from pathlib import Path
from torch.utils.data import DataLoader
# Here we construct the trainer of Multlingual BERT
class GPTrainer:  
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
        # self.tokenizer = tokenizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.performance_logger = PerformanceLogger(output_dir)
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()

            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )

            for batch_idx, (encoding, labels) in train_pbar:
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

            self.performance_logger.save_to_csv(f"{self.output_dir}/training_log.csv")

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
            for _, (encoding, labels) in val_pbar:
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
    # fetch the appropriate model
    model = get_model()
    
    # initialize trainer class
    model_trainer = GPTrainer(model,tokenizer)
    
    # start training
    model_trainer.train()
    
    # saving the model after training
    save_finetuned_gpt2(model,tokenizer,output_dir=TRAINVAL_OUTPUT_DIR)
    
    print('Finetuned GPT2 Model Saved.')

                
                