from src.vision.vgg import adjustedVGGNet
from src.vision.vgg_peft import adjustedPeftVGGNet
import importlib
from utilities import save_cli_args, fixed_seed,split_save_load_dataset
from trainers.vision.trainer import Trainer
import numpy as np
import warnings
import options

# Initialize the needed variables either from the CLI user sents or from the device.

DEVICE = options.DEVICE
parser = options.parser
args = parser.parse_args()

INPUT_DIR = args.input_dir
VGGNET_VERSION = args.vgg_net_version
BATCH_SIZE=args.batch_size
LEARNING_RATE=args.learning_rate
NUM_CLASSES = args.num_classes
NUM_EPOCHS = args.num_epochs
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
TEST_SIZE = args.test_size
CHECK_VAL_EVERY_N_EPOCH = 1
USE_PEFT = args.use_peft
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FIXED_SEED = args.fixed_seed
FREEZE_BACKBONE = args.freeze_backbone
MODE = args.mode

# If we want to apply fixed seed or randomly initialize the weights and dataset.
if FIXED_SEED:
    SEED=42
    fixed_seed(SEED)
else:
    SEED = np.random.randint(low=0, high=1000)
    fixed_seed(SEED)
    warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

# Fetch whether the transfer-learning with PEFT version or transfer-learning without
def get_model():

    """Allows the user to choose from Adjusted ResNet18 or PEFT-ResNet18 versions.
    """
    if ADDED_LAYERS == 0:
        
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    if USE_PEFT:
        
        model = importlib.import_module('src.vision.vgg_peft')
        args.model = 'PEFT-' + VGGNET_VERSION
        return model.adjustedPeftVGGNet

        pass

    else:
        model = importlib.import_module('src.vision.vgg')
        args.model = VGGNET_VERSION
        return model.adjustedVGGNet
    
# load the dataset with appropriate paths
TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
VAL_DATASET_PATH = options.VAL_DATASET_PATH
TEST_DATASET_PATH = options.TEST_DATASET_PATH
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR

train_loader, val_loader = split_save_load_dataset(
    
    mode='train',
    type='image',
    input_dir= INPUT_DIR,
    train_size = TRAIN_SIZE,
    val_size = VAL_SIZE,
    test_size = TEST_SIZE,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
    seed=SEED,
    batch_size=BATCH_SIZE,
    tokenizer=None
)    

if __name__ == "__main__":
    
    # fetch the appropriate model
    choosed_model = get_model()
    
    # pass the options from the args user are feeding as input
    model = choosed_model(NUM_CLASSES,VGGNET_VERSION, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE,task_type=MODE)
    
    # initialize trainer class
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader)
    
    print('The Trainer class is loaded successfully.')
    
    # start training & validation
    trainer.train()
    trainer.validate()
    
    print('Saving the model and arguments is under way!')
    
    # save model and arguments
    trainer.saveModel(path=f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth')
    
    save_cli_args(args, TRAINVAL_OUTPUT_DIR, mode='train')