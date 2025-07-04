from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from datasets.image_datasets import ParquetImageDataset
from src.vision.siglip import load_siglip_offline, load_custom_siglip_model
from src.vision.siglip_peft import load_peft_siglip_offline
import options
from utilities import get_args
from src.vision.siglip import load_custom_siglip_model


parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = get_args()

OUTPUT = "deeptune_results/test_set_pretrained_siglip_embeddings.parquet"

TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_CASE = args.use_case
MODEL_WEIGHTS = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FREEZE_BACKBONE = args.freeze_backbone

def run_embeddings():
    
    if USE_CASE == 'finetuned':
        model = load_custom_siglip_model(
            added_layers=ADDED_LAYERS,
            embedding_dim=EMBED_SIZE,
            num_classes=NUM_CLASSES,
            freeze_backbone=FREEZE_BACKBONE
        )
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

    elif USE_CASE == 'peft':
        model = load_peft_siglip_offline(
            added_layers=ADDED_LAYERS,
            embedding_layer=EMBED_SIZE,
            freeze_backbone=FREEZE_BACKBONE,
            num_classes=NUM_CLASSES
        )
        
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))


    if USE_CASE == 'pretrained':
        model,_ = load_siglip_offline()
    

    _, processor = load_siglip_offline()

    image_dataset = ParquetImageDataset(TEST_DATASET_PATH, processor=processor)
    
    loader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    device = torch.device(DEVICE)
    df = get_vision_embeddings(model, loader, device)
    
    df.to_parquet(OUTPUT)
    print(f"Saved image embeddings to {OUTPUT}.")

def get_vision_embeddings(model, loader, device):
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
            pooled_output = model.base_model.vision_model(pixel_values=pixel_values).pooler_output

            if hasattr(model, "fc_layers"): 
                intermediate = model.fc_layers[1](model.fc_layers[0](pooled_output))  # Linear → ReLU
                image_embeddings = intermediate
            else: 
                image_embeddings = pooled_output
            
            all_embeddings.append(image_embeddings.cpu())
            print(label)
            all_labels.append(label.item())
   
    embeddings = torch.cat(all_embeddings)
    
    _, p = embeddings.shape
    cols = [f"embed{i:04d}" for i in range(p)]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_embed["target"] = all_labels
    df_embed.columns = df_embed.columns.astype(str)
   
    return df_embed

if __name__ == "__main__":
    run_embeddings()