from deeptune.src.vision.vgg import adjustedVGGNet
from deeptune.src.vision.vgg_peft import adjustedPeftVGGNet
from deeptune.utilities import transformations
import torch
from deeptune.datasets.image_datasets import ParquetImageDataset
import pandas as pd 
from deeptune.evaluators.vision.evaluator import TestTrainer
from deeptune.utilities import save_cli_args
import options

# Initialize the needed variables either from the CLI user sents or from the device.

parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = parser.parse_args()
VGGNET_VERSION = args.vgg_net_version
TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_PEFT = args.use_peft
MODEL_WEIGHTS = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
MODE = args.mode

if USE_PEFT:
    
    MODEL = adjustedPeftVGGNet(NUM_CLASSES,VGGNET_VERSION, ADDED_LAYERS, lora_attention_dimension=EMBED_SIZE,task_type=MODE)
    args.model = 'PEFT-' + VGGNET_VERSION
    
else:
    MODEL = adjustedVGGNet(NUM_CLASSES,VGGNET_VERSION, ADDED_LAYERS, EMBED_SIZE,task_type=MODE)
    args.model = VGGNET_VERSION


# Load the test dataset from its path and testloader
df = pd.read_parquet(TEST_DATASET_PATH)

test_dataset = ParquetImageDataset.from_parquet(parquet_file=TEST_DATASET_PATH, transform=transformations)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
  
if __name__ == "__main__":
    
     # Call the trainer class and save arguments.
    
    test_trainer = TestTrainer(model=MODEL, batch_size=BATCH_SIZE,test_loader=test_loader)
    
    test_trainer.test(best_model_weights_path=MODEL_WEIGHTS)
    
    save_cli_args(args, TEST_OUTPUT_DIR, mode='test')
    
    print('Test results saved successfully!')