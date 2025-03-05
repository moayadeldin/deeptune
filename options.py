import argparse
import torch
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET_PATH = Path(__file__).parent / "train_split.parquet"
VAL_DATASET_PATH = Path(__file__).parent / "val_split.parquet"
TEST_DATASET_PATH = Path(__file__).parent / "test_split.parquet"

TRAINVAL_OUTPUT_DIR = Path(__file__).parent / 'output_directory_trainval'
TEST_OUTPUT_DIR = Path(__file__).parent / 'output_directory_test'

parser = argparse.ArgumentParser(description='DeepTune CLI Arguments')
# basic settings

parser.add_argument('--num_classes', type=int, help='The number of classes in your dataset.')
parser.add_argument('--input_dir', type=str, help='Directory containing input data.')
parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')
parser.add_argument('--freeze-backbone', action='store_true', help='Decide whether you want to freeze backbone or not.')
parser.add_argument('--added_layers', type=int, choices=[1,2], help='Specify the number of layers you want to add.')
parser.add_argument('--embed_size', type=int, help='Specify the size of the embeddings you would obtain through embedding layer.')
parser.add_argument('--batch_size', type=int, help='Batch Size to feed your model.')
parser.add_argument('--fixed-seed', action='store_true', help='Choose whether a seed is required or not.')

# training settings
parser.add_argument('--num_epochs', type=int, help='The number of epochs you wan the model to run on.')
parser.add_argument('--learning_rate', type=float, help='Learning Rate to apply for fine-tuning.')
parser.add_argument('--train_size', type=float, help='Mention the split ratio of the Train Dataset')
parser.add_argument('--val_size', type=float, help='Mention the split ratio of the Val Dataset')
parser.add_argument('--test_size', type=float, help='Mention the split ratio of the Test Dataset')

# evaluating settings
parser.add_argument('--test_set_input_dir', type=str, help='Directory containing test data.')

# embedding settings
parser.add_argument('--model_weights', type=str, help='Directory for your tuned model.')



