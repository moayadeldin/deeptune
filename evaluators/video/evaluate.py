from pathlib import Path
from torch.utils.data import DataLoader

from datasets.video_datasets import ParquetVideoDataset, native_video_transform
from evaluators.vision.evaluator import TestTrainer
from helpers import transformations
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM

from cli import DeepTuneVisionOptions
from utils import RunType, build_video_model


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
    NUM_FRAMES = args.num_frames
    POOLING = args.pooling

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
        num_frames=NUM_FRAMES,
        pooling=POOLING,
        args=args
    )


def evaluate(eval_df, out, model_weights, num_classes, model_version, mode, added_layers, embed_size, batch_size, freeze_backbone, args, use_peft, model_str, num_frames=8, pooling="mean"):
    """
    Evaluates a trained DeepTune video classifier on a held-out parquet split.

    Mirrors evaluators/vision/evaluate.py's `evaluate`, reusing the exact same
    TestTrainer class as the images pipeline - the video models built by
    utils.build_video_model expose the same forward(x) -> [B, num_classes] API.
    """

    mw = Path(model_weights)

    EVAL_OUTPUT_DIR = (out / f"eval_output_{model_str}_{UNIQUE_ID}")
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args.save_args(EVAL_OUTPUT_DIR)

    ckpt_path = mw if mw.suffix == ".pth" else next(mw.glob("*.pth"))

    MODEL = build_video_model(
        model_version=model_version,
        num_classes=num_classes,
        added_layers=added_layers,
        embed_size=embed_size,
        freeze_backbone=freeze_backbone,
        mode=mode,
        use_peft=use_peft,
        pooling=pooling,
    )

    test_dataset = ParquetVideoDataset.from_parquet(eval_df, transform=transformations, num_frames=num_frames,
                                                   clip_transform=native_video_transform(model_version))

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

    metrics_dict = test_trainer.test(best_model_weights_path=ckpt_path)

    print('Test results saved successfully!')

    return metrics_dict


if __name__ == "__main__":
    main()
