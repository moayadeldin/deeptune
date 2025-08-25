import argparse
import torch

from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count


ROOT = Path(__file__).parent.parent
DEEPTUNE_RESULTS = ROOT / 'deeptune_results'
DOWNLOADED_MODELS = ROOT / 'downloaded_models'

UNIQUE_ID = datetime.now().strftime("%Y%m%d_%H%M") # unique ID based on current date and time (YYYYMMDD_HHMM)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    
    return torch.device(device_type)

DEVICE = get_device()

CPU_COUNT: int = cpu_count()
NUM_WORKERS: int = min(8, CPU_COUNT // 2) if DEVICE.type != "cpu" else 0
PERSIST_WORK: bool = NUM_WORKERS > 0
PIN_MEM: bool = DEVICE.type == "cuda"


# The paths of the Parquet splits of the dataset we are saving/loading from
TRAIN_DATASET_PATH = DEEPTUNE_RESULTS / f'train_split_{UNIQUE_ID}.parquet'
VAL_DATASET_PATH = DEEPTUNE_RESULTS / f'val_split_{UNIQUE_ID}.parquet'
TEST_DATASET_PATH = DEEPTUNE_RESULTS / f'test_split_{UNIQUE_ID}.parquet'

TRAINVAL_OUTPUT_DIR = DEEPTUNE_RESULTS / f'output_directory_trainval_{UNIQUE_ID}'
TEST_OUTPUT_DIR = DEEPTUNE_RESULTS / f'output_directory_test_{UNIQUE_ID}'


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='DeepTune CLI Arguments')

    # basic settings

    parser.add_argument('--mode', type=str,choices=['reg','cls'], help='Mode: Classification or Regression')
    parser.add_argument('--num_classes', type=int, help='The number of classes in your dataset.')
    parser.add_argument('--input_dir', type=str, help='Directory containing input data.')
    parser.add_argument('--use-peft', action='store_true', help='Include this flag to use PEFT-adapted model.')
    parser.add_argument('--freeze-backbone', action='store_true', help='Decide whether you want to freeze backbone or not.')
    parser.add_argument('--added_layers', type=int, choices=[1,2], help='Specify the number of layers you want to add.')
    parser.add_argument('--embed_size', type=int, help='Specify the size of the embeddings you would obtain through embedding layer.')
    parser.add_argument('--batch_size', type=int, help='Batch Size to feed your model.')
    parser.add_argument('--fixed-seed', action='store_true', help='Choose whether a seed is required or not.')
    parser.add_argument('--model_weights', type=str, help='Directory for your tuned model.')


    # training settings
    parser.add_argument('--num_epochs', type=int, help='The number of epochs you wan the model to run on.')
    parser.add_argument('--learning_rate', type=float, help='Learning Rate to apply for fine-tuning.')
    parser.add_argument('--train_size', type=float, help='Mention the split ratio of the Train Dataset')
    parser.add_argument('--val_size', type=float, help='Mention the split ratio of the Val Dataset')
    parser.add_argument('--test_size', type=float, help='Mention the split ratio of the Test Dataset')
    parser.add_argument('--data_dir', type=Path, help="Directory containing dataset splits.")  # my addition due to separating data splitting into separate script

    # evaluating settings
    parser.add_argument('--test_set_input_dir', type=str, help='Directory containing test data.')

    # embedding settings
    parser.add_argument('--use_case', type=str, choices=['peft', 'finetuned','pretrained'],help='The mode you want to set embeddings extractor with') 

    # bert evaluating settings
    parser.add_argument("--adjusted_bert_dir", type=Path,help='Path to the adjusted saved weights of Finetuned BERT.')
    parser.add_argument("--adjusted_gpt2_dir", type=Path,help='Path to the adjusted saved weights of Finetuned GPT-2.')

    # Models version
    parser.add_argument('--resnet_version', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='ResNet version to use.')
    parser.add_argument('--densenet_version', type=str, choices=['densenet121', 'densenet169', 'densenet201', 'densenet161'], help='DenseNet version to use.')
    parser.add_argument('--swin_version', type=str, choices=['swin_t', 'swin_s', 'swin_b'], help='Swin version to use.')
    parser.add_argument('--efficientnet_version', type=str, choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'], help='EfficientNet version to use.')
    parser.add_argument('--vgg_net_version', type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], help='VGGNet version to use.')
    parser.add_argument('--vit_version', type=str, choices=['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14'], help='ViT version to use.')
    parser.add_argument('--siglip_version', type=str, choices=['siglip'], help="Siglip version to use.")  # only the one option

    # timeseries models
    parser.add_argument('--target_column', type=str, help='Target column for time series forecasting.')
    parser.add_argument('--time_idx_column', type=str, help='Time index column for time series forecasting.')
    parser.add_argument('--max_encoder_length', type=int, help='How much history the model sees',default=60)
    parser.add_argument('--max_prediction_length', type=int, help='How many steps into the future it will predict.', default=20)

    # GANDALF model
    parser.add_argument('--tabular_target_column', nargs='+', type=str, help='Target column for GANDALF')
    parser.add_argument('--continuous_cols', nargs='+', help='List of continuous column names for GANDALF')
    parser.add_argument('--categorical_cols', nargs='+', help='List of categorical column names for GANDALF')
    parser.add_argument('--gflu_stages', type=int, default=6, help='Number of GFLU stages for GANDALF')
    parser.add_argument('--type', type=str, choices=['classification', 'regression'], help='Task type for GANDALF')

    return parser

