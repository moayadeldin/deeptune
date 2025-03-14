from src.vision.densenet121 import adjustedDenseNet
from src.vision.densenet121_peft import adjustedPEFTDenseNet
import importlib
from utilities import save_cli_args, fixed_seed,split_save_load_dataset
from trainers.trainer import Trainer
import numpy as np
import warnings
import options

DEVICE = options.DEVICE
parser = options.parser
args = parser.parse_args()

INPUT_DIR = args.input_dir
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

if FIXED_SEED:
    SEED=42
    fixed_seed(SEED)
else:
    SEED = np.random.randint(low=0, high=1000)
    fixed_seed(SEED)
    warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
    warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)

def get_model():

    """Allows the user to choose from Adjusted DenseNet121 or PEFT-DenseNet121 versions.
    """
    
    if ADDED_LAYERS == 0:
        
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')
    
    else:
        
        
        if USE_PEFT:
            model = importlib.import_module('src.vision.densenet121_peft')
            args.model = 'PEFT-DenseNet121'
            return model.adjustedPEFTDenseNet
        else:
            model = importlib.import_module('src.vision.densenet121')
            args.model = 'DenseNet121'
            return model.adjustedDenseNet
        
        
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
    
    choosed_model = get_model()
    
    model = choosed_model(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE,task_type=MODE)
    
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader)
    
    print('The Trainer class is loaded successfully.')
    
    trainer.train()
    trainer.validate()
    
    print('Saving the model and arguments is under way!')
    
    trainer.saveModel(path=f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth')
    
    save_cli_args(args, TRAINVAL_OUTPUT_DIR, mode='train')