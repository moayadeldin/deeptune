from pathlib import Path
from torch.utils.data import DataLoader

from helpers import transformations
from trainers.vision.trainer import Trainer
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from datasets.image_datasets import ParquetImageDataset

from cli import DeepTuneVisionOptions
from utils import get_model_cls, RunType,set_seed


def main():
    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    DATA_DIR: Path = args.input_dir
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

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
    FIXED_SEED = args.fixed_seed

    if FIXED_SEED:
        set_seed(FIXED_SEED)

    if ADDED_LAYERS == 0:
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    TRAIN_DATASET_PATH = TRAIN_PATH or ( DATA_DIR / "train_split.parquet" )
    VAL_DATASET_PATH = VAL_PATH or ( DATA_DIR / "val_split.parquet" )

    TRAINVAL_OUTPUT_DIR = (OUT / f"trainval_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/trainval_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")


    if MODEL_ARCHITECTURE.lower() == "siglip" and MODEL_VERSION == "siglip":
        from trainers.vision.custom_train_siglip_handler import train_siglip
        train_siglip(
            num_classes=NUM_CLASSES,
            added_layers=ADDED_LAYERS,
            embed_size=EMBED_SIZE,
            train_dataset_path=TRAIN_DATASET_PATH,
            val_dataset_path=VAL_DATASET_PATH,
            outdir=TRAINVAL_OUTPUT_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            use_peft=USE_PEFT,
            freeze_backbone=FREEZE_BACKBONE,
        )
        args.save_args(TRAINVAL_OUTPUT_DIR)
        return

    train_dataset = ParquetImageDataset.from_parquet(parquet_file=TRAIN_DATASET_PATH, transform=transformations)
    val_dataset = ParquetImageDataset.from_parquet(parquet_file=VAL_DATASET_PATH, transform=transformations)

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

    chosen_model = get_model_cls(MODEL_ARCHITECTURE, USE_PEFT)
    model = chosen_model(NUM_CLASSES, MODEL_VERSION, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE, task_type=MODE)

    TRAINVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader, learning_rate=LEARNING_RATE, mode=MODE, num_epochs=NUM_EPOCHS, output_dir=TRAINVAL_OUTPUT_DIR)
    
    print('The Trainer class is loaded successfully.')
    
    trainer.train()
    trainer.validate()
    
    print('Saving the model and arguments is under way!')
    
    trainer.saveModel(path=f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth')
    args.save_args(TRAINVAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()