from pathlib import Path
from datasets import Dataset
from PIL import Image
# Define the label-to-index mapping
label_to_index = {
    "Benign": 1,
    "Early": 2,
    "Pre": 3,
    "Pro": 4}
if __name__ == "__main__":
    workdir = Path(r"H:\Moayad\archive\splitted")  # Replace with the root dataset directory
    splits = ["train", "test", "val"]  # List of dataset splits
    # Create a combined data dictionary
    combined_data = {"images": [], "labels": []}
    for split in splits:
        split_path = workdir / split
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():  # Ensure it's a directory
                class_label = class_dir.name  # Use folder name as label
                # Map the string label to its numeric index
                if class_label not in label_to_index:
                    print(f"Warning: Label '{class_label}' is not in the label_to_index mapping. Skipping.")
                    continue
                numeric_label = label_to_index[class_label]
                for image_file in class_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        # Append image bytes and numeric class label
                        combined_data["images"].append(image_file.read_bytes())
                        combined_data["labels"].append(numeric_label)
    # Save the combined data to a single parquet file
    out_path = workdir / "combined.parquet"
    combined_dataset = Dataset.from_dict(combined_data)
    combined_dataset.to_parquet(out_path)