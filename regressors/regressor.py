from datasets.image_datasets import ParquetImageDataset
from helpers import transformations
from regressors.components.backbone import RegressorBackbone
from regressors.components.loss import coord_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pandas as pd
import io
from PIL import Image

train_dataset = ParquetImageDataset.from_parquet(
        parquet_file="/home/moayad/Downloads/RP/python-scripts/circles_training_sample_parquet.parquet",
        transform=transformations,
        coords_normalized=False,)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = ParquetImageDataset.from_parquet(parquet_file="/home/moayad/Downloads/RP/python-scripts/circles_validation_sample.parquet.parquet",transform=transformations,coords_normalized=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

num_epochs = 1

model = RegressorBackbone(backbone="resnet18", pretrained=True,heatmap_size=56).to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)


def train():

    for epoch in range(num_epochs):

        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        running_loss = 0.0

        for batch_idx, (images, coords) in enumerate(progress_bar):
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            coords = coords.to("cuda" if torch.cuda.is_available() else "cpu")


            pred_coords, pred_heatmap = model(images)

            loss = coord_loss(pred_coords, target_coords=coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # show current and average loss in the progress bar

            average_loss = running_loss / (batch_idx + 1)
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{average_loss:.4f}",
            )

        epoch_average_loss = running_loss / len(train_loader)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} finished | "
            f"Average Loss: {epoch_average_loss:.4f}"
        )

def validate():

    model.eval()

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
    )

    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, coords) in enumerate(progress_bar):
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            coords = coords.to("cuda" if torch.cuda.is_available() else "cpu")

            pred_coords, pred_heatmap = model(images)

            loss = coord_loss(pred_coords, target_coords=coords)

            running_loss += loss.item()

            average_loss = running_loss / (batch_idx + 1)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{average_loss:.4f}",
            )

    validation_average_loss = running_loss / len(val_loader)
    print(
        f"\nValidation finished | "
        f"Average Loss: {validation_average_loss:.4f}"
    )


if __name__ == "__main__":
    train()
    validate()
