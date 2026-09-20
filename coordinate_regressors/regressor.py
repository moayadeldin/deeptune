from datasets.image_datasets import ParquetImageDataset
from helpers import transformations
from coordinate_regressors.components.backbone import RegressorBackbone
from coordinate_regressors.components.loss import coord_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from options import DEVICE,NUM_WORKERS,PIN_MEM,UNIQUE_ID
from cli import DeepTuneVisionOptions
from utils import RunType
from handlers.split_dataset import split_dataset
from helpers import date_id
from pathlib import Path

from helpers import PerformanceLogger

def saveModel(model,performance_logger,path):

    torch.save(model.state_dict(), path)
    performance_logger.logger.info(f"Model Saved to {path}")

def main():

    args = DeepTuneVisionOptions(RunType.COORDINATE_REGRESSION)
    OUT = args.out
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    TRAIN_SIZE = args.train_size if hasattr(args, 'train_size') else 0.7
    VAL_SIZE = args.val_size if hasattr(args, 'val_size') else 0.1
    TEST_SIZE = args.test_size if hasattr(args, 'test_size') else 0.2

    FIXED_SEED = args.fixed_seed if hasattr(args, 'fixed_seed') else True

    HEATMAP_SIZE = args.heatmap_size


    model = RegressorBackbone(backbone="resnet18", pretrained=True,heatmap_size=HEATMAP_SIZE).to(DEVICE)

    parent_dir = date_id(prefix="deeptune_coordinate_regression", root_dir=OUT)
    run_output_dir = OUT / parent_dir
    COORDINATE_REGRESSION_OUTPUT_DIR = (
        run_output_dir / f"coordinate_regression_output_resnet18_{UNIQUE_ID}"
    )
    COORDINATE_REGRESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_path = args.df

    train_path, val_path, test_path = split_dataset(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        df_path=df_path,
        fixed_seed=FIXED_SEED, # fix the seed for the initial implementation, can be changed later
        out_dir=run_output_dir,
        disable_numerical_encoding=True,
        target_column="coords",
        disable_target_column_renaming=True,
        modality="image",
        grouper=None
    )

    train(train_path=train_path, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, val_path=val_path,args=args,coordinate_regression_output=COORDINATE_REGRESSION_OUTPUT_DIR,model=model)
    test_loss = test(test_path=test_path, batch_size=BATCH_SIZE,model=model)

    file_path = COORDINATE_REGRESSION_OUTPUT_DIR / "test_log.txt"

    with open(file_path, "w") as f:
        f.write(f"Test loss: {test_loss}.")



def train(num_epochs, learning_rate, batch_size, train_path, val_path,args,coordinate_regression_output,model):
    train_dataset = ParquetImageDataset.from_parquet(
        parquet_file=train_path,
        transform=transformations,
        coords_normalized=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    performance_logger = PerformanceLogger(coordinate_regression_output)


    for epoch in range(num_epochs):

        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        running_loss = 0.0

        for batch_idx, (images, coords) in enumerate(progress_bar):
            images = images.to(DEVICE)
            coords = coords.float().to(DEVICE)


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

        # ========================
        # Validation
        # ========================
        val_loss = validate(val_path=val_path, batch_size=batch_size,model=model)


        performance_logger.log_epoch(
            epoch = epoch+1,
            epoch_loss=epoch_average_loss,
            epoch_accuracy=None,
            val_loss=val_loss,
            val_accuracy=None,
        )

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} finished | "
            f"Average Training Loss: {epoch_average_loss:.4f}"
        )

        performance_logger.save_to_csv(coordinate_regression_output / "trainval_performance_log.csv")
        print("\n Saving model weights...")

        output_dir = coordinate_regression_output / f"model_weights.pth"

        saveModel(model,performance_logger,path=output_dir)
        args.save_args(coordinate_regression_output)

def validate(val_path, batch_size,model):

    val_dataset = ParquetImageDataset.from_parquet(
        parquet_file=val_path,
        transform=transformations,
        coords_normalized=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

    model.eval()

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
    )

    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, coords) in enumerate(progress_bar):
            images = images.to(DEVICE)
            coords = coords.to(DEVICE)

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
        f"Average Validation Loss: {validation_average_loss:.4f}"
    )

    return validation_average_loss


def test(test_path, batch_size,model):

    test_dataset = ParquetImageDataset.from_parquet(
        parquet_file=test_path,
        transform=transformations,
        coords_normalized=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

    model.eval()

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
    )

    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, coords) in enumerate(progress_bar):
            images = images.to(DEVICE)
            coords = coords.to(DEVICE)

            pred_coords, pred_heatmap = model(images)

            loss = coord_loss(pred_coords, target_coords=coords)

            running_loss += loss.item()

            average_loss = running_loss / (batch_idx + 1)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{average_loss:.4f}",
            )

    test_average_loss = running_loss / len(test_loader)
    print(
        f"\nTesting finished | "
        f"Average Test Loss: {test_average_loss:.4f}"
    )

    return test_average_loss


if __name__ == "__main__":
    main()