import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime

from options import UNIQUE_ID

"""
Loading the images is more complex because we need to consider the subdirectories where data splits (train/test/val) are usually separated into different directories, and each directory has other subdirectories of the respective classes. Moreover, we need to obtain the bytes representation of the image as a proper format to feed images' representation.

We assume that your dataset is organized such that the main dataset folder contains three subdirectories, train, test, and val. Each of these splits, in turn, contains subdirectories for each class, and within those class folders are the images.

For Tabular, Text, and Time-Series data, we assume the input is in CSV or Excel format, which we can directly read into a DataFrame and then convert to Parquet.

"""


def main():
   
    parser = make_parser()
    args = parser.parse_args()

    RAW_DATASET_DIR: Path = args.raw_dataset_dir
    OUT_PATH: Path = args.out
    MODALITY: str = args.modality

    out_dir = raw_to_parquet(
        dataset_dir=RAW_DATASET_DIR,
        out=OUT_PATH,
        modality=MODALITY,
    )

    print(f"Raw dataset converted and saved to Parquet at: {out_dir}")

def raw_to_parquet(dataset_dir: Path, out: Path, modality:str):
    if modality == "images":
        data = load_images_and_labels_mcc(dataset_dir)

        df = pd.DataFrame(data)

        out = out / f"images_dataset_{UNIQUE_ID}.parquet"

        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out)

        return out

    if modality == "tabular" or modality == "timeseries" or modality == "text":

        if dataset_dir.suffix == '.xlsx':
            df = pd.read_excel(dataset_dir)
            

        elif dataset_dir.suffix == '.csv':
            df = pd.read_csv(dataset_dir)

        else:
            raise ValueError(f"Unsupported file type: {dataset_dir.suffix}")
        
        out = out / f"{modality}_dataset_{UNIQUE_ID}.parquet"

        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out)
        return out


def load_images_and_labels_mcc(dataset_dir):

    """
    Handles data for Binary/Multi-Class Classification, where it expects the input to be several directories inside the spits directories, with each directory named as the class name.

    Returns:
        combined_data (dict) : Dictionary that is containing image bytes representation after being read, and their corresponding labels.

    """

    dataset_dir = Path(dataset_dir)

    combined_data = {"images": [], "labels": []}

    for split_dir in dataset_dir.iterdir(): # iterate over each split in the three splits' directories

      if split_dir.is_dir():

        for class_dir in split_dir.iterdir(): # iterate over each class
          if class_dir.is_dir():

            class_label = class_dir.name # get the name of the class

            for image_file in class_dir.iterdir(): # iterate over each image
                if image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]: # if format proper then append it to dictonary
                    combined_data["images"].append(image_file.read_bytes())
                    combined_data["labels"].append(class_label)
                else: # if not then throw a warning
                    print(f"Warning: File {image_file} not found or unsupported format.")

          else:
              raise ValueError(f"Expected another directory for class labels, but found a file: {class_dir}")

    return combined_data

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Load images from directory structure into a dictionary with image bytes and labels.", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--raw_dataset_dir",
        type=Path,
        required=True,
        help="Path to the main dataset directory containing train, val, and test subdirectories."
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to save the output Parquet file."
    )

    parser.add_argument(
        "--modality",
        type=str,
        choices=["images", 'text', 'tabular', 'timeseries'],
        required=True,
        help="Type of data modality."
    )
    return parser


if __name__ == "__main__":
    main()

