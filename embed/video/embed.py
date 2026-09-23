import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from cli import DeepTuneVisionOptions
from datasets.video_datasets import ParquetVideoDataset, native_video_transform, collate_video_samples
from helpers import transformations
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import RunType, build_video_model, save_process_times


def embed(df_path, out, model_weights, batch_size, use_case, model_version, added_layers, embed_size, num_classes, args, mode, grouper, model_str, num_frames=8, pooling="mean"):
    """
    Extracts clip-level embeddings from a DeepTune video classifier.

    Unlike embed/vision/embed.py, no per-architecture layer surgery is needed
    here: both video model families (native spatio-temporal, and frame-sampling
    on a 2D backbone) already implement the same forward(x, extract_embed=True)
    convention used across DeepTune's adjusted*Net classes, so VideoEmbeddingModel
    below can call it directly regardless of which architecture was chosen.
    """

    EMBED_OUTPUT = out / f"embed_output_{model_str}_{UNIQUE_ID}"
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{model_str}_{mode}_embeddings.parquet"

    use_peft = use_case == "peft"

    model = build_video_model(
        model_version=model_version,
        num_classes=num_classes,
        added_layers=added_layers,
        embed_size=embed_size,
        freeze_backbone=False,
        mode=mode,
        use_peft=use_peft,
        pooling=pooling,
    )

    if use_case in ('peft', 'finetuned'):
        mw = Path(model_weights)
        ckpt_path = mw if mw.suffix == ".pth" else next(mw.glob("*.pth"))
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu'))

    start_time = time.time()

    embedding_model = VideoEmbeddingModel(
        model,
        model_version=model_version,
        added_layers=added_layers,
        embed_size=embed_size,
        num_frames=num_frames,
        device=DEVICE,
    )

    df = pd.read_parquet(df_path)

    combined_df = embedding_model(df, batch_size)

    if grouper is not None and grouper in df.columns:
        combined_df[grouper] = df[grouper]

    combined_df.to_parquet(EMBED_FILE, index=False)

    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")

    args.save_args(EMBED_OUTPUT)

    print(f'The embeddings file is saved in {EMBED_OUTPUT}')

    return EMBED_OUTPUT, combined_df.shape


def main():
    args = DeepTuneVisionOptions(RunType.EMBED)
    DF_PATH: Path = args.df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_STR = args.model

    USE_CASE = args.use_case.value
    ADDED_LAYERS = args.added_layers
    MODEL_WEIGHTS = args.model_weights
    EMBED_SIZE = args.embed_size

    BATCH_SIZE = args.batch_size
    GROUPER = args.grouper
    NUM_FRAMES = args.num_frames
    POOLING = args.pooling

    embed(
        df_path=DF_PATH,
        num_classes=NUM_CLASSES,
        model_version=MODEL_VERSION,
        model_weights=MODEL_WEIGHTS,
        use_case=USE_CASE,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
        out=OUT,
        mode=MODE,
        model_str=MODEL_STR,
        args=args,
        grouper=GROUPER,
        num_frames=NUM_FRAMES,
        pooling=POOLING,
    )


class VideoEmbeddingModel:
    """
    Extracts clip-level embeddings from a trained (or freshly initialised, for
    the "pretrained" use case) DeepTune video model - either a native
    spatio-temporal architecture or a frame-sampling model built on one of the
    existing 2D vision backbones.
    """

    def __init__(self, model, model_version, added_layers=2, embed_size=1000, num_frames=8, device=DEVICE):
        self.model = model
        self.model_version = model_version
        self.added_layers = added_layers
        self.embed_size = embed_size
        self.num_frames = num_frames
        self.device = device

    def __call__(self, df: DataFrame, batch_size: int = 2) -> DataFrame:
        if df.empty:
            raise ValueError('The embedding dataset is empty.')
        dataset = ParquetVideoDataset(df, transform=transformations, num_frames=self.num_frames,
                                       clip_transform=native_video_transform(self.model_version))
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEM,
            persistent_workers=PERSIST_WORK,
            collate_fn=collate_video_samples,
        )
        return self._extract_embeddings(data_loader)

    def _extract_embeddings(self, data_loader: DataLoader) -> DataFrame:

        self.model.to(self.device)

        embeddings, labels, extracted_other_cols = self._extract_embeddings_labels(data_loader)

        embeddings_df = pd.DataFrame(embeddings)

        if labels is None:
            return embeddings_df

        labels_df = pd.DataFrame(labels, columns=["label"])
        other_cols_df = pd.DataFrame(extracted_other_cols) if extracted_other_cols else None

        if other_cols_df is not None:
            combined_df = pd.concat([embeddings_df, labels_df, other_cols_df], axis=1)
        else:
            combined_df = pd.concat([embeddings_df, labels_df], axis=1)

        return combined_df

    def _extract_embeddings_labels(self, data_loader: DataLoader) -> tuple[np.ndarray, list, list]:
        """
        Runs every clip in `data_loader` through the model with
        extract_embed=True and collects the resulting embeddings alongside
        their labels and any extra (passthrough) columns.
        """
        extracted_labels = []
        extracted_embeddings = []
        extracted_other_cols = []

        self.model.eval()
        self.model = self.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(data_loader):
                data, labels, *extras = batch
                data = data.to(self.device)

                embeddings = self.model(data, extract_embed=True)

                extracted_embeddings.append(embeddings.detach().cpu().numpy())

                if labels is not None:
                    extracted_labels += labels.numpy().tolist()
                if extras:
                    # extras is a list of dicts, take the first dict element
                    dict_extras = extras[0]
                    batch_size = len(data)
                    for idx in range(batch_size):
                        sample_extras = {k: v[idx].item() if hasattr(v, 'item') else v[idx]
                                    for k, v in dict_extras.items()}
                        extracted_other_cols.append(sample_extras)

        extracted_embeddings = np.vstack(extracted_embeddings)
        labels_array = extracted_labels if len(extracted_labels) > 0 else None
        return extracted_embeddings, labels_array, extracted_other_cols


if __name__ == "__main__":

    main()
