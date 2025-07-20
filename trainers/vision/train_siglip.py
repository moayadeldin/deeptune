import json
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.image_datasets import ParquetImageDataset
from options import DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from src.vision.siglip import (
    CustomSiglipModel,
    CustomSigLIPWithPeft,
    load_siglip_processor_offline,
    load_siglip_variant,
)
from utilities import PerformanceLogger, get_args, split_save_load_dataset, fixed_seed

from utils import save_cli_args, UseCase


def train_siglip(
    num_classes: int,
    added_layers: int,
    embed_size: int,
    train_dataset_path: Path,
    val_dataset_path: Path,
    outdir: Path,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    use_peft: bool,
    freeze_backbone: bool,
) -> None:
    use_case = UseCase.PEFT if use_peft else UseCase.FINETUNED
    model = load_siglip_variant(
        use_case=use_case,
        num_classes=num_classes,
        added_layers=added_layers,
        embed_size=embed_size,
        freeze_backbone=freeze_backbone,
        model_weights=None,
        device=device,
    )
    processor = load_siglip_processor_offline()

    train_dataset = ParquetImageDataset.from_parquet(parquet_file=train_dataset_path, processor=processor)
    val_dataset = ParquetImageDataset.from_parquet(parquet_file=val_dataset_path, processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,            # adjust based on CPU cores
        pin_memory=PIN_MEM,                 # speeds up CPU-GPU transfer
        persistent_workers=PERSIST_WORK     # for PyTorch >= 1.8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )

    outdir.mkdir(parents=True, exist_ok=True)

    model_trainer = SiglipTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        outdir=outdir,
        device=device,
        use_peft=use_peft,
    )
    model_trainer.train()


class SiglipTrainer:
    def __init__(
        self,
        model: CustomSiglipModel | CustomSigLIPWithPeft,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        outdir: Path,
        device: torch.device,
        use_peft: bool,
    ):
        self.model = model
        self.model.to(device)

        self.num_epochs = num_epochs
        self.outdir = outdir
        self.device = device
        self.use_peft = use_peft

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.scaler = GradScaler(device=device.type)
        
        self.performance_logger = PerformanceLogger(outdir)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
    
    def train(self):
        
        for epoch in range(self.num_epochs):
            self.model.train()
            """
            The following arguments are as follows:
            
            - running_loss: indicating the loss during the epoch for train and val datasets respectively.
            - correct_predictions: number of predictions that equal true label 
            - total_predictions: number of predictions made in total
            """
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
            )

            for batch_idx, (pixel_values, labels) in train_pbar:
                # make sure the inputs and labels on GPU
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)
                
                # null gradient
                self.optimizer.zero_grad()

                with autocast(device_type=self.device.type):
                    logits = self.model({'pixel_values': pixel_values}) 
                    loss = self.criterion(logits, labels)
                
                # backprop and apply optimizer
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # accumulate loss
                running_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(logits, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                    
                # update tqdm progress bar
                train_pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                })
                
            # now we compute average loss and accuracy for each epoch
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100. * correct_predictions / total_predictions
                    
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {running_loss / len(self.train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
            
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
            
        self.performance_logger.save_to_csv(f"{self.outdir}/training_log.csv")
        
        torch.save(self.model.state_dict(), self.outdir / "custom_siglip_model.pt")
        print(f"Siglip model saved to {os.path.join(self.outdir, f'{"PEFT_" if self.use_peft else ""}SIGLIP_model')}")

        config = {
            "added_layers": self.model.added_layers,
            "embedding_dim": self.model.embedding_dim,
            "num_classes": self.model.num_classes,
        }
        with open(self.outdir / "custom_siglip_config.json", "w") as f:
            json.dump(config, f)
                    
    def validate(self):
        """
        The evaluation function is devoted for the validation set only, please consider the test function for test set.
        """
        val_accuracy = 0.0
        val_loss = 0.0
        total, correct = 0, 0
        
        val_pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
        )
        self.model.eval()
        
        with torch.no_grad():
            for _, (pixel_values, labels) in val_pbar:
                
                
                pixel_values,labels = pixel_values.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()

                with autocast(device_type=self.device.type):
                    logits = self.model({'pixel_values': pixel_values}) 
                    loss = self.criterion(logits, labels)
                
                val_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        val_accuracy = correct / total
        val_loss = val_loss / len(self.val_loader)
        return val_loss, val_accuracy
        




if __name__ == '__main__':
    args = get_args()

    # Arguments constructed for CLI
    INPUT_DIR = args.input_dir
    BATCH_SIZE=args.batch_size
    LEARNING_RATE=args.learning_rate
    NUM_CLASSES = args.num_classes
    NUM_EPOCHS = args.num_epochs
    TRAIN_SIZE = args.train_size
    VAL_SIZE = args.val_size
    TEST_SIZE = args.test_size
    USE_PEFT = args.use_peft
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FIXED_SEED = args.fixed_seed
    FREEZE_BACKBONE = args.freeze_backbone

    # If we want to apply fixed seed or randomly initialize the weights and dataset.
    if FIXED_SEED:
        SEED=42
        fixed_seed(SEED)
    else:
        SEED = np.random.randint(low=0, high=1000)
        fixed_seed(SEED)
        warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
        warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

    if FIXED_SEED:
        seed=42
        args.fixed_seed = 42
    else:
        seed = np.random.randint(low=0, high=1000)
        args.fixed_seed = seed

    TRAINVAL_OUTPUT_DIR = Path(__file__).parent.parent / 'output_directory_trainval'

    if USE_PEFT:
        args.model = 'PEFT-SIGLIP'
    else:
        args.model = 'SIGLIP'

    use_case = UseCase.PEFT if USE_PEFT else UseCase.FINETUNED
    model = load_siglip_variant(
        use_case=use_case,
        num_classes=NUM_CLASSES,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        model_weights=None,
        device=DEVICE,
    )
    processor = load_siglip_processor_offline()
    
    import options
    TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
    VAL_DATASET_PATH = options.VAL_DATASET_PATH
    TEST_DATASET_PATH = options.TEST_DATASET_PATH
    TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR

    # We load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
    train_loader, val_loader = split_save_load_dataset(
        mode='train',
        type='image',
        input_dir= INPUT_DIR,
        train_size = TRAIN_SIZE,
        val_size = VAL_SIZE,
        test_size = TEST_SIZE,
        train_dataset_path=TRAIN_DATASET_PATH,
        val_dataset_path=VAL_DATASET_PATH,
        test_dataset_path=TEST_DATASET_PATH,
        seed=SEED,
        batch_size=BATCH_SIZE,
        tokenizer=processor,
        siglip=True
    )
        
    model_trainer = SiglipTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        outdir=TRAINVAL_OUTPUT_DIR,
        device=DEVICE,
        use_peft=USE_PEFT,
    )
    model_trainer.train()
    save_cli_args(args, TRAINVAL_OUTPUT_DIR)