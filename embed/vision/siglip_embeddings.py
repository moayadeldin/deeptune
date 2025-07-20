import torch

from pandas import DataFrame
from pathlib import Path
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.image_datasets import ParquetImageDataset
from options import DEVICE, UNIQUE_ID, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from src.vision.siglip import CustomSiglipModel, CustomSigLIPWithPeft, load_siglip_processor_offline, load_siglip_variant
from utilities import get_args

from utils import UseCase


def embed_with_siglip(
    dataset_path: Path,
    model_weights: Path,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    use_case: str,
    output: Path,
    device: torch.device
):
    use_case = UseCase.from_string(use_case)
    model = load_siglip_variant(
        use_case=use_case,
        num_classes=num_classes,
        added_layers=added_layers,
        embed_size=embed_size,
        freeze_backbone=False,  # no effect during inference
        model_weights=model_weights,
        device=device,
    )
    
    processor = load_siglip_processor_offline()

    image_dataset = ParquetImageDataset.from_parquet(dataset_path, processor=processor)
    
    loader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK
    )
    
    df = get_vision_embeddings(model, loader, device)
    
    df.to_parquet(output, index=False)
    print(f"Saved image embeddings to {output}.")


def get_vision_embeddings(
    model: CustomSiglipModel | CustomSigLIPWithPeft,
    loader: DataLoader,
    device: torch.device,
):
    model = model.to(device)
    model.eval()
    print("Starting image embedding...")
    print(f"Model architecture: {type(model).__name__}")
    print(f"Using device: {device}")
   
    all_embeddings = []
    all_labels = []
   
    with torch.no_grad():
        for pixel_values, label in tqdm(
            loader,
            total=len(loader),
            desc="Embedding images"
        ):
            pixel_values = pixel_values.to(device)

            with autocast(device_type=device.type):
                image_embeddings = model.get_image_embeddings({"pixel_values": pixel_values})            
            
            all_embeddings.append(image_embeddings.cpu())
            all_labels.append(label.item())
   
    embeddings = torch.cat(all_embeddings)
    
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed["target"] = all_labels
    df_embed.columns = df_embed.columns.astype(str)
   
    return df_embed


if __name__ == "__main__":
    args = get_args()

    TEST_DATASET_PATH = args.test_set_input_dir
    MODEL_WEIGHTS = args.model_weights
    NUM_CLASSES = args.num_classes
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    USE_CASE = args.use_case
    BATCH_SIZE= args.batch_size

    MODEL_VERSION = "siglip_vision"
    MODE = 'cls'

    RESULTS_DIR = Path(__file__).parent.parent.parent / "deeptune_results"

    EMBED_OUTPUT = RESULTS_DIR / f"embed_output_{USE_CASE}_{MODEL_VERSION}_{MODE}_{UNIQUE_ID}"
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{MODEL_VERSION}_{USE_CASE}_{MODE}_embeddings.parquet"
    args.model = f'{USE_CASE}-' + MODEL_VERSION
    
    embed_with_siglip(
        dataset_path=TEST_DATASET_PATH,
        model_weights=MODEL_WEIGHTS,
        num_classes=NUM_CLASSES,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        use_case=USE_CASE,
        output=EMBED_FILE,
        device=DEVICE,
    )