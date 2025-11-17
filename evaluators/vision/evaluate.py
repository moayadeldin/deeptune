from pathlib import Path
from torch.utils.data import DataLoader

from datasets.image_datasets import ParquetImageDataset
from evaluators.vision.evaluator import TestTrainer
from helpers import transformations
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM

from cli import DeepTuneVisionOptions
# from evaluators.vision.custom_evaluate_siglip_handler import evaluate_siglip
from utils import get_model_cls, RunType

def main() -> None:
    args = DeepTuneVisionOptions(RunType.EVAL)
    EVAL_DF_PATH: Path = args.eval_df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_STR = args.model

    MODEL_WEIGHTS = args.model_weights
    USE_PEFT = args.use_peft
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    BATCH_SIZE = args.batch_size

    evaluate(
        eval_df=EVAL_DF_PATH,
        mode=MODE,
        num_classes=NUM_CLASSES,
        out=OUT,
        model_version=MODEL_VERSION,
        model_str=MODEL_STR,
        model_weights=MODEL_WEIGHTS,
        use_peft=USE_PEFT,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        args=args
    )



def evaluate(eval_df, out, model_weights, num_classes,model_version,mode,added_layers,embed_size, batch_size, freeze_backbone, args, use_peft, model_str):

        
    EVAL_OUTPUT_DIR = (out / f"eval_output_{model_str}_{UNIQUE_ID}")
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_ARCHITECTURE = args.model_architecture

    adjusted_model_cls = get_model_cls(MODEL_ARCHITECTURE, use_peft=use_peft)
    MODEL = adjusted_model_cls(num_classes, model_version, added_layers, embed_size, task_type=mode,freeze_backbone=freeze_backbone)

    test_dataset = ParquetImageDataset.from_parquet(eval_df, transform=transformations)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )
  
    test_trainer = TestTrainer(
        model=MODEL,
        batch_size=batch_size,
        test_loader=test_loader,
        output_dir=EVAL_OUTPUT_DIR,
        device=DEVICE,
        mode=mode
    )

    # if MODEL_ARCHITECTURE == "siglip" and MODEL_VERSION == "siglip":
    #     evaluate_siglip(
    #         test_dataset_path=EVAL_DF_PATH,
    #         model_weights=MODEL_WEIGHTS,
    #         num_classes=NUM_CLASSES,
    #         added_layers=ADDED_LAYERS,
    #         embed_size=EMBED_SIZE,
    #         use_peft=USE_PEFT,
    #         batch_size=BATCH_SIZE,
    #         device=DEVICE,
    #         outdir=EVAL_OUTPUT_DIR,
    #     )

    mw = Path(model_weights)

    if mw.suffix == ".pth":
        ckpt_path = mw
    else:
        ckpt_path = next(mw.glob("*.pth"))
    
    metrics_dict = test_trainer.test(best_model_weights_path=ckpt_path)
    
    args.save_args(EVAL_OUTPUT_DIR)
    
    print('Test results saved successfully!')

    return metrics_dict



if __name__ == "__main__":
    main()