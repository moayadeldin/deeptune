import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.nn.functional as F
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from autoencoders.components.architecture import AutoEncoder
from autoencoders.components.transforms import apply_transform
from datasets.image_datasets import ParquetImageDataset


from options import DEVICE,UNIQUE_ID

def main():

    parser = make_parser()
    args = parser.parse_args()

    TRAIN_DF_PATH: Path = args.train_df
    TEST_DF_PATH: Path = args.test_df

    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    OUT = args.out
    MODEL_WEIGHTS = args.model_weights

    IF_GRAYSCALE = args.if_grayscale

    transform = apply_transform(IF_GRAYSCALE)

    model = AutoEncoder(in_ch=1 if IF_GRAYSCALE else 3).to(DEVICE)

    run_autoencoder(
        model=model,
        train_df_path=TRAIN_DF_PATH,
        transform=transform,
        test_df_path=TEST_DF_PATH,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        out=OUT,
        model_weights=MODEL_WEIGHTS,
        device=DEVICE
    )



def run_autoencoder(model, train_df_path, transform, test_df_path, num_epochs, learning_rate, out,model_weights=None, device=DEVICE):

    test_df = pd.read_parquet(test_df_path)

    test_dataset = ParquetImageDataset(test_df, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights, map_location=device))
        print(
            f"Model weights loaded from {model_weights}. "
            "Evaluating on test set without further training."
        )
        test_autoencoder(model, test_loader, device=device)
        save_results(model, out, model_weights, test_loader)
        return

    train_df = pd.read_parquet(train_df_path)

    train_dataset = ParquetImageDataset(train_df, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()

    model.train()

    for _ in range(num_epochs):
        for images, _ in tqdm(train_loader, desc="Training"):
            x_hat, x_target = model(images.to(next(model.parameters()).device))

            loss = criterion(x_hat, x_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_autoencoder(model, test_loader, device=device)
    save_results(model, out, model_weights, test_loader)

    print("Autoencoder training complete. Model and reconstructed images were saved to the specified output directory.")



def test_autoencoder(model, test_loader, device=DEVICE):

    encoded_list = []
    decoded_list = []
    original_list = []

    model.eval()
    with torch.no_grad():

        for images, _ in tqdm(test_loader, desc="Testing"):

            x_hat, x_target = model(images.to(next(model.parameters()).device))

            encoded_list.append(x_target.cpu())
            decoded_list.append(x_hat.cpu())
            original_list.append(images.cpu())

        return encoded_list, decoded_list, original_list
    

def load_autoencoder(model_path, device=DEVICE):

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))



def save_results(model, out_dir, model_weights, test_loader):
    
    model.eval()

    split_dir = out_dir / f"reconstructed_images_{UNIQUE_ID}"

    os.makedirs(split_dir, exist_ok=True)

    if model_weights is None:
        torch.save(model.state_dict(), split_dir / "autoencoder_model.pth")

    with torch.no_grad():

        counters = {}
        for images, labels in tqdm(test_loader, desc="Saving"):

            images = images.to(DEVICE)

            x_hat, _ = model(images)

            for i in range(images.shape[0]):

                label = labels[i].item()
                label_name = f'Class {label}'

                class_dir = os.path.join(split_dir, label_name)
                os.makedirs(class_dir, exist_ok=True)

                counters.setdefault(label_name,0)
                idx = counters[label_name]

                reconstructed_image = rescale_image(x_hat[i])
                save_image(reconstructed_image, os.path.join(class_dir, f"reconstructed_{idx}.png"))

                counters[label_name] += 1

def rescale_image(x,eps=1e-6):

    """
    This function rescales the input image tensor x to the range [0,1] using min-max normalization. The smallest pixel value is the brightest and the largest pixel value is the darkest.
    """

    # x: (C,H,W)

    x = x - x.amin(dim=(-2,-1), keepdim=True)
    x = x / (x.amax(dim=(-2,-1), keepdim=True) + eps)
    return x

def make_parser():
    parser = ArgumentParser(description="Split parquet file into Train, Val, and Test splits. Allows the same splits to be used for training several deep learners. The splits maintain all features.", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--train_df",
        type=Path,
        required=False,
        help="Path of the parquet file containing the training data for the autoencoder."
    )

    parser.add_argument(
        "--test_df",
        type=Path,
        required=True,
        help="Path of the parquet file containing the test data for the autoencoder."
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        help="Number of epochs to train the autoencoder for."
    ) 

    parser.add_argument(
        "--model_weights",
        type=Path,
        help="Path to the pretrained autoencoder model weights. If not specified, the autoencoder will be trained from scratch. If specified, the autoencoder will be loaded with the pretrained weights and then evaluated on the test set without further training."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="Learning rate for training the autoencoder."
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Location and name of the directory to save the trained autoencoder model."
    )

    parser.add_argument(
        '--if-grayscale',
        action='store_true',
        help='Whether the input images are grayscale. If not specified, the default is False, meaning the input images are RGB.'
    )

    return parser

if __name__ == "__main__":
    main()




