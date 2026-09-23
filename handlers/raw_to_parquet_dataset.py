import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime

from options import UNIQUE_ID

"""
Loading images (and video) accepts two directory layouts, detected automatically per top-level entry:

1. dataset_dir/class_name/*.jpg - one subdirectory per class, directly under the main dataset folder.
2. dataset_dir/split_name/class_name/*.jpg - one or more split directories (e.g. 'train', 'val', 'test';
   the names themselves don't matter and aren't preserved - DeepTune re-splits everything itself later),
   each wrapping one subdirectory per class.

A top-level entry is treated as layout 2 (a split) if it contains subdirectories, and as layout 1 (a
class) otherwise - see _load_labeled_files_mcc. The two layouts can even be mixed across different
top-level entries of the same dataset_dir.

Video datasets follow the exact same convention as images, except each class folder holds raw video
clips (.mp4, .avi, .mov, .mkv, .webm, .m4v, .wmv, .mpeg, or .mpg) instead of images.

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
    dataset_dir, out = Path(dataset_dir), Path(out)
    if modality not in ('images', 'video', 'text', 'tabular', 'timeseries'):
        raise ValueError(f'Unsupported modality: {modality}')
    if not dataset_dir.exists():
        raise ValueError(f'Dataset path does not exist: {dataset_dir}')
    if modality == "images":
        data = load_images_and_labels_mcc(dataset_dir)

        df = pd.DataFrame(data)

        out = out / f"images_dataset_{UNIQUE_ID}.parquet"

        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out)

        return out

    if modality == "video":
        if dataset_dir.is_file():
            if dataset_dir.suffix.lower() not in ('.csv', '.xlsx'):
                raise ValueError('For training, use a labeled video folder or a CSV/XLSX list with videos and labels columns. A single MP4 can be previewed in the desktop app.')
            from handlers.video_manifest import load_video_manifest
            df = load_video_manifest(dataset_dir)
        else:
            df = pd.DataFrame(load_videos_and_labels_mcc(dataset_dir))

        out = out / f"video_dataset_{UNIQUE_ID}.parquet"

        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out)

        return out

    if modality == "tabular" or modality == "timeseries" or modality == "text":

        if dataset_dir.suffix.lower() == '.xlsx':
            df = pd.read_excel(dataset_dir)
            

        elif dataset_dir.suffix.lower() == '.csv':
            df = pd.read_csv(dataset_dir)

        else:
            raise ValueError(f"Unsupported file type: {dataset_dir.suffix}")
        
        out = out / f"{modality}_dataset_{UNIQUE_ID}.parquet"

        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out)
        return out


def _load_labeled_files_mcc(dataset_dir, data_key: str, extensions: list[str]) -> dict:
    """
    Shared directory-walking logic behind load_images_and_labels_mcc and
    load_videos_and_labels_mcc.

    Every top-level entry of `dataset_dir` is inspected on its own merits
    rather than assumed to uniformly be either "a class" or "a split
    wrapping classes":
      - if it contains subdirectories, those subdirectories are the classes
        (this is the train/val/test-style layout: dataset_dir/split/class/*);
      - otherwise, the entry itself is treated as a class, and the
        recognised files directly inside it belong to that class (this is
        the flatter, and more common, layout: dataset_dir/class/*, with no
        split wrapper at all).
    This handles both layouts - and any mix of the two - with one rule,
    instead of a single dataset-wide "exactly one top-level directory"
    special case that misreads a plain two-class dataset (two top-level
    class directories, each holding files, not further subdirectories) as
    two empty "splits" and silently loads nothing.

    Returns:
        combined_data (dict): {data_key: [...bytes...], "labels": [...]}
    """

    dataset_dir = Path(dataset_dir)
    combined_data = {data_key: [], "labels": []}

    top_level_dirs = sorted(entry for entry in dataset_dir.iterdir() if entry.is_dir() and not entry.name.startswith('.'))

    if not top_level_dirs:
        raise ValueError(f"No subdirectories found under {dataset_dir}. "
              f"Expected either one subdirectory per class, or split subdirectories "
              f"(e.g. 'train'/'val'/'test') each containing one subdirectory per class.")

    for entry in top_level_dirs:
        class_dirs = sorted(child for child in entry.iterdir() if child.is_dir() and not child.name.startswith('.'))

        if class_dirs:
            # `entry` is a split wrapper (e.g. train/val/test); its own
            # subdirectories are the classes.
            for class_dir in class_dirs:
                for data_file in sorted(class_dir.iterdir()):
                    if data_file.is_file() and data_file.suffix.lower() in extensions:
                        combined_data[data_key].append(data_file.read_bytes())
                        combined_data["labels"].append(class_dir.name)
                    else:
                        print(f"Warning: File {data_file} not found or unsupported format.")
        else:
            # `entry` itself is a class - its files belong directly to it.
            for data_file in sorted(entry.iterdir()):
                if data_file.is_file() and data_file.suffix.lower() in extensions:
                    combined_data[data_key].append(data_file.read_bytes())
                    combined_data["labels"].append(entry.name)
                else:
                    print(f"Warning: File {data_file} not found or unsupported format.")

    if not combined_data[data_key]:
        raise ValueError(f'No supported {data_key} files were found under {dataset_dir}.')
    return combined_data


def load_images_and_labels_mcc(dataset_dir):

    """
    Handles data for Binary/Multi-Class Classification. Accepts either
    dataset_dir/class/*.jpg directly, or dataset_dir/split/class/*.jpg with
    one or more split directories (e.g. train/val/test) wrapping the class
    directories - see _load_labeled_files_mcc for exactly how each
    top-level entry is classified as one or the other.

    Returns:
        combined_data (dict) : Dictionary that is containing image bytes representation after being read, and their corresponding labels.

    """

    return _load_labeled_files_mcc(dataset_dir, "images", [".png", ".jpg", ".jpeg"])


def load_videos_and_labels_mcc(dataset_dir):

    """
    Handles data for Binary/Multi-Class video classification, mirroring
    load_images_and_labels_mcc: accepts either dataset_dir/class/*.mp4
    directly, or dataset_dir/split/class/*.mp4 with one or more split
    directories wrapping the class directories.

    Returns:
        combined_data (dict) : Dictionary containing video bytes representation after being read, and their corresponding labels.

    """

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".mpeg", ".mpg"]
    return _load_labeled_files_mcc(dataset_dir, "videos", video_extensions)


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
        choices=["images", 'text', 'tabular', 'timeseries', 'video'],
        required=True,
        help="Type of data modality."
    )
    return parser


if __name__ == "__main__":
    main()

