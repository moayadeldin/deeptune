from pathlib import Path
from torch.utils.data import DataLoader

from datasets.image_datasets import ParquetImageDataset
from evaluators.vision.evaluator import TestTrainer
from helpers import transformations
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM

from cli import DeepTuneVisionOptions
from evaluators.vision.custom_evaluate_siglip_handler import evaluate_siglip
from utils import get_model_cls, RunType


def main() -> None:
    args = DeepTuneVisionOptions(RunType.EVAL)
    EVAL_DF_PATH: Path = args.eval_df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_ARCHITECTURE = args.model_architecture
    MODEL_STR = args.model

    MODEL_WEIGHTS = args.model_weights
    USE_PEFT = args.use_peft
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    
    BATCH_SIZE = args.batch_size
    
    EVAL_OUTPUT_DIR = (OUT / f"eval_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/eval_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_ARCHITECTURE == "siglip" and MODEL_VERSION == "siglip":
        evaluate_siglip(
            test_dataset_path=EVAL_DF_PATH,
            model_weights=MODEL_WEIGHTS,
            num_classes=NUM_CLASSES,
            added_layers=ADDED_LAYERS,
            embed_size=EMBED_SIZE,
            use_peft=USE_PEFT,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            outdir=EVAL_OUTPUT_DIR,
        )
        args.save_args(EVAL_OUTPUT_DIR)
        print('Test results saved successfully!')
        return

    adjusted_model_cls = get_model_cls(MODEL_ARCHITECTURE, use_peft=USE_PEFT)
    MODEL = adjusted_model_cls(NUM_CLASSES, MODEL_VERSION, ADDED_LAYERS, EMBED_SIZE, task_type=MODE)

    test_dataset = ParquetImageDataset.from_parquet(EVAL_DF_PATH, transform=transformations)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )
  
    test_trainer = TestTrainer(
        model=MODEL,
        batch_size=BATCH_SIZE,
        test_loader=test_loader,
        output_dir=EVAL_OUTPUT_DIR,
        device=DEVICE,
        mode=MODE
    )
    
    test_trainer.test(best_model_weights_path=MODEL_WEIGHTS)
    
    args.save_args(EVAL_OUTPUT_DIR)
    
    print('Test results saved successfully!')


if __name__ == "__main__":
    main()