from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert import load_nlp_bert_ml_model_offline
import importlib
from utilities import save_cli_args, fixed_seed,split_save_load_dataset,save_finetunedbertmodel
import numpy as np
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
import options
import logging
from utilities import PerformanceLogger
import sys

# Initialize the needed variables either from the CLI user sents or from the device.

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

    
TRAIN_DATASET_PATH = options.TRAIN_DATASET_PATH
VAL_DATASET_PATH = options.VAL_DATASET_PATH
TEST_DATASET_PATH = options.TEST_DATASET_PATH
TRAINVAL_OUTPUT_DIR = options.TRAINVAL_OUTPUT_DIR

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

    """
    Allows the user to choose from Adjusted ResNet18 or PEFT-ResNet18 versions.
    """
    if ADDED_LAYERS == 0:
        
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    if USE_PEFT:
        
        pass
    else:
        model = importlib.import_module('src.nlp.multilingual_bert')
        args.model = 'Multilingual BERT'
        return model.CustomMultilingualBERT
    
# load the tokenizer, the dataloaders

_,tokenizer = load_nlp_bert_ml_model_offline()

train_loader, val_loader = split_save_load_dataset(
    
    mode='train',
    type='text',
    input_dir= INPUT_DIR,
    train_size = TRAIN_SIZE,
    val_size = VAL_SIZE,
    test_size = TEST_SIZE,
    train_dataset_path=TRAIN_DATASET_PATH,
    val_dataset_path=VAL_DATASET_PATH,
    test_dataset_path=TEST_DATASET_PATH,
    seed=SEED,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer
)

# Here we construct the trainer of Multlingual BERT

class BERTrainer:
    def __init__(self,model,tokenizer):
        
        """
        Performs Training & Validation on the input text dataset.
        
        Args:
        
            model (HuggingFace Model): The NLP BERT model we are loading from the src file.
            tokenizer (HuggingFace Tokenizer): The NLP BERT model we are loading from load_nlp_bert_ml_model_offline() function in utilities file.
            
        Attributes:
        
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            optimizer (torch.optim.Optimizer): Adam optimizer for updating the model weights during training.
            logger (logging.Logger): Logger instance for tracking training progress.
        """
        
        
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.tokenizer = tokenizer
        
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.performance_logger = PerformanceLogger(f'{TRAINVAL_OUTPUT_DIR}')
        
        

    def train(self):
        
        for epoch in range(NUM_EPOCHS):
            
            # set the model to training mode
            self.model.train()
            
            # initialize metrics tracking values for training
            running_loss = 0.0
            correct_predictions = 0.0
            total_predictions = 0.0
            
            # initlaize progress bar
            train_pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader)
            )
            
            # iterate over the training dataset
            for batch_idx, (encoding,labels) in train_pbar:
                
                # move the input to GPU 
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                token_type_ids = encoding.get('token_type_ids',None)
                
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                
                labels = labels.to(DEVICE)
                
                # null the gradient
                self.optimizer.zero_grad()
                #run the model
                outputs = self.model(input_ids,attention_mask,token_type_ids)
                
                # compute the loss
                loss = self.criterion(outputs,labels)
                
                # backprop and apply optimizer
                loss.backward()
                self.optimizer.step()
                
                # accumulate loss
                running_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs,1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # update tqdm progress bar
                train_pbar.set_postfix({
                        'loss': running_loss / (batch_idx + 1)
                    })
                
                epoch_loss = running_loss / len(train_loader)
                epoch_accuracy = 100. * correct_predictions / total_predictions
                        
                        
            # update the logger
            self.logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {epoch_accuracy}"
            )
            
            # apply validation after each epoch on training set
                
            val_loss, val_accuracy = self.validate()
            self.logger.info(f"Validation loss: {val_loss}, Accuracy: {val_accuracy}")
                
            # update the performance logger
            self.performance_logger.log_epoch(
                epoch = epoch+1,
                epoch_loss=epoch_loss,
                epoch_accuracy=epoch_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy
                
            )
                
                
            self.performance_logger.save_to_csv(f"{TRAINVAL_OUTPUT_DIR}/training_log.csv")
            
    def validate(self):
        
        
        # initialize the metrics for validation
        val_accuracy=0.0
        val_loss = 0.0
        total,correct=0,0
        
        
        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader)
        )
        
        with torch.no_grad():
            
            for _, (encoding,labels) in val_pbar:
                
                # set the model to evaluation mode
                self.model.eval()
                
                # move the input to GPU
                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                token_type_ids = encoding.get('token_type_ids',None)
                
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                
                labels = labels.to(DEVICE)
                
                # apply forward pass and accumulate loss
                
                outputs = self.model(input_ids,attention_mask,token_type_ids)
                
                loss = self.criterion(outputs,labels)
                
                val_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
        val_accuracy = correct / total
        val_loss = val_loss / len(val_loader)
        return val_loss, val_accuracy
                
        
            
if __name__ == '__main__':
    
    
    # fetch the appropriate model
    choosed_model = get_model()
    
    # pass the options from the args user are feeding as input
    model = choosed_model(NUM_CLASSES, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE)
    
    # initialize trainer class
    model_trainer = BERTrainer(model,tokenizer)
    
    # start training
    model_trainer.train()
    
    # saving the model after training
    
    model_config = {
        "num_classes":NUM_CLASSES,
        "added_layers":ADDED_LAYERS,
        "embedding_layer": EMBED_SIZE
    }
    
    
    # save the finetuned model
    save_finetunedbertmodel(model,tokenizer,output_dir=TRAINVAL_OUTPUT_DIR,model_config=model_config)
    
    print('Finetuned BERT Model Saved.')