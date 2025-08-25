import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoTokenizer
from tqdm import tqdm

from deeptune.datasets.text_datasets import TextDataset
from deeptune.options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM, DEEPTUNE_RESULTS

from deeptune.cli import DeepTuneNLPOptions
from deeptune.utils import RunType

from deeptune.utilities import save_finetuned_gpt2, save_finetunedbertmodel, PerformanceLogger

from deeptune.src.nlp.gpt2 import AdjustedGPT2Model, load_gpt2_tokenizer_offline
from deeptune.src.nlp.multilingual_bert import CustomMultilingualBERT, load_bert_tokenizer_offline, \
    CustomMultilingualPeftBERT
# from deeptune_beta.src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT
# from deeptune_beta.trainers.nlp.train_gpt2 import GPTrainer
# from deeptune_beta.trainers.nlp.train_multilinbert import BERTrainer


def main():
    from time import time
    start = time()

    args = DeepTuneNLPOptions(RunType.TRAIN)
    DATA_DIR: Path = args.input_dir
    MODE = args.mode
    NUM_CLASSES = args.num_classes

    MODEL_VERSION = args.model_version
    MODEL_ARCHITECTURE = args.model_architecture
    MODEL_STR = args.model
    
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate

    if ADDED_LAYERS == 0:
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    TRAIN_DATASET_PATH = DATA_DIR / "train_split.parquet"
    VAL_DATASET_PATH = DATA_DIR / "val_split.parquet"

    TRAINVAL_OUTPUT_DIR = DEEPTUNE_RESULTS / f"train_output_{MODEL_STR}_{UNIQUE_ID}"
    TRAINVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model_tokenizer(
        model=MODEL_VERSION,
        num_classes=NUM_CLASSES,
        added_layers=ADDED_LAYERS,
        embedding_layer=EMBED_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        use_peft=USE_PEFT,
    )

    train_dataset = TextDataset.from_parquet(parquet_file=TRAIN_DATASET_PATH, tokenizer=tokenizer)
    val_dataset = TextDataset.from_parquet(parquet_file=VAL_DATASET_PATH, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )

    trainer_cls = GPTrainer if MODEL_VERSION.lower() == "gpt2" else BERTrainer
    trainer = trainer_cls(
        model,
        train_loader,
        val_loader,
        LEARNING_RATE,
        MODE,
        NUM_EPOCHS,
        TRAINVAL_OUTPUT_DIR,
    )
    trainer.train()
    
    print('Saving the model and arguments is under way!')
    
    # if MODEL_VERSION.lower() == "gpt2":
    #     save_finetuned_gpt2(model, tokenizer, TRAINVAL_OUTPUT_DIR)
    # elif MODEL_VERSION.lower() == "multilinbert":
    #     model_config = {
    #         "num_classes":NUM_CLASSES,
    #         "added_layers":ADDED_LAYERS,
    #         "embedding_layer": EMBED_SIZE,
    #     }
    #     save_finetunedbertmodel(model, tokenizer, TRAINVAL_OUTPUT_DIR, model_config=model_config)
    
    torch.save(model.state_dict(), TRAINVAL_OUTPUT_DIR / 'model_weights.pth')
    print(f"Model weights saved to {TRAINVAL_OUTPUT_DIR / 'model_weights.pth'}")

    args.save_args(TRAINVAL_OUTPUT_DIR)

    end = time()
    with open(TRAINVAL_OUTPUT_DIR / "train_time.txt", 'w') as f:
        f.write(f"{MODEL_STR} training duration:\n{round(end - start, 2)} seconds\n")


def get_model_tokenizer(
    model: str,
    num_classes: int,
    added_layers: int,
    embedding_layer: int,
    freeze_backbone: bool = False,
    use_peft: bool = False
) -> tuple[AdjustedGPT2Model | CustomMultilingualBERT, GPT2Tokenizer | AutoTokenizer]:
    if model.lower() == "gpt2":
        model = AdjustedGPT2Model(num_classes=num_classes, freeze_backbone=freeze_backbone, output_dim=embedding_layer)
        tokenizer = load_gpt2_tokenizer_offline()
    elif model.lower() == "multilinbert":
        if added_layers == 0:
            raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

        model_cls = CustomMultilingualPeftBERT if use_peft else CustomMultilingualBERT
        model = model_cls(num_classes, added_layers, embedding_layer, freeze_backbone)
        tokenizer = load_bert_tokenizer_offline()
    
    return model, tokenizer


class BERTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        mode: str,
        num_epochs: int,
        output_dir: Path,
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
            
            # set the model to training mode
            self.model.train()
            
            # initialize metrics tracking values for training
            running_loss = 0.0
            correct_predictions = 0.0
            total_predictions = 0.0
            
            # initlaize progress bar
            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader)
            )
            
            # iterate over the training dataset
            for batch_idx, (encoding,labels) in train_pbar:
                
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
            self.performance_logger.save_to_csv(f"{self.output_dir}/training_log.csv")
            
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
            
            for _, (encoding,labels) in val_pbar:
                
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
 

class GPTrainer:  
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        mode: str,
        num_epochs: int,
        output_dir: Path,
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


if __name__ == "__main__":
    main()