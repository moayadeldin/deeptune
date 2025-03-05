from src.vision.resnet18 import adjustedResNet
from src.vision.resnet18_peft import adjustedPeftResNet
from utilities import transformations
import torch
from datasets.image_datasets import ParquetImageDataset
import pandas as pd 
from evaluators.evaluator import TestTrainer
from utilities import save_cli_args
import options


parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = parser.parse_args()

TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_PEFT = args.use_peft
MODEL_WEIGHTS = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size

if USE_PEFT:
    
    MODEL = adjustedPeftResNet(NUM_CLASSES, ADDED_LAYERS, lora_attention_dimension=EMBED_SIZE)
    args.model = 'PEFT-RESNET18'
    
else:
    MODEL = adjustedResNet(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE)
    args.model = 'RESNET18'


df = pd.read_parquet(TEST_DATASET_PATH)

test_dataset = ParquetImageDataset(parquet_file=TEST_DATASET_PATH, transform=transformations)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
  
if __name__ == "__main__":
    
    test_trainer = TestTrainer(model=MODEL, batch_size=BATCH_SIZE,test_loader=test_loader)
    
    test_trainer.test(best_model_weights_path=MODEL_WEIGHTS)
    
    save_cli_args(args, TEST_OUTPUT_DIR, mode='test')
    
    print('Test results saved successfully!')
    
    