from pathlib import Path
from torch.utils.data import DataLoader

from helpers import transformations
from trainers.vision.trainer import Trainer
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from datasets.image_datasets import ParquetImageDataset
from trainers.vision.custom_siglip_train import train_siglip
from cli import DeepTuneVisionOptions
from utils import get_model_cls, RunType,set_seed

def main():
    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_STR = args.model
    
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    FIXED_SEED = args.fixed_seed

    train(
        train_df=TRAIN_PATH,
        val_df=VAL_PATH,
        mode=MODE,
        num_classes=NUM_CLASSES,
        out=OUT,
        model_version=MODEL_VERSION,
        model_str=MODEL_STR,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        use_peft=USE_PEFT,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fixed_seed=FIXED_SEED,
        args=args
    )




def train(
        train_df: Path,
        val_df: Path,
        out: Path,
        freeze_backbone: bool,
        use_peft: bool,
        fixed_seed: int,
        mode:str,
        model_version:str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        added_layers: int,
        num_classes: int,
        embed_size: int,
        model_str: str,
        args: DeepTuneVisionOptions,

):
    
    MODEL_ARCHITECTURE = args.model_architecture
    if fixed_seed:
        set_seed(fixed_seed)

    if added_layers == 0:
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    TRAINVAL_OUTPUT_DIR = (out / f"trainval_output_{model_str}_{UNIQUE_ID}")

    if MODEL_ARCHITECTURE.lower() == "siglip" and model_version == "siglip":
        _ = train_siglip(
            num_classes=num_classes,
            added_layers=added_layers,
            embed_size=embed_size,
            train_dataset_path=train_df,
            val_dataset_path=val_df,
            outdir=TRAINVAL_OUTPUT_DIR,
            device=DEVICE,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            use_peft=use_peft,
            freeze_backbone=freeze_backbone,
        )
        args.save_args(TRAINVAL_OUTPUT_DIR)
        return TRAINVAL_OUTPUT_DIR

    train_dataset = ParquetImageDataset.from_parquet(parquet_file=train_df, transform=transformations)
    val_dataset = ParquetImageDataset.from_parquet(parquet_file=val_df, transform=transformations)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )

    chosen_model = get_model_cls(model_architecture=MODEL_ARCHITECTURE, use_peft=use_peft)
    model = chosen_model(num_classes, model_version, added_layers, embed_size, freeze_backbone, task_type=mode)

    TRAINVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader, learning_rate=learning_rate, mode=mode, num_epochs=num_epochs, output_dir=TRAINVAL_OUTPUT_DIR)
    
    print('The Trainer class is loaded successfully.')
    
    trainer.train()
    trainer.validate()
    
    print('Saving the model and arguments is under way!')

    output_dir = f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth'
    
    trainer.saveModel(path=output_dir)
    args.save_args(TRAINVAL_OUTPUT_DIR)

    return output_dir
    


if __name__ == "__main__":
    main()