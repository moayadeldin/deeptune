
from utilities import save_cli_args
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pyarrow.parquet as pq
import torch
from datasets.image_datasets import ParquetImageDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utilities import save_cli_args,get_args
import sys
from pathlib import Path
import pandas as pd 
import logging
import numpy as np
import options
from src.vision.siglip import CustomSiglipModel, load_siglip_offline
from src.vision.siglip_peft import CustomSigLIPWithPeft, load_peft_siglip_offline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).parent.parent

parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = get_args()

TEST_DATASET_PATH = args.test_set_input_dir
BATCH_SIZE= args.batch_size
NUM_CLASSES = args.num_classes
USE_PEFT = args.use_peft
MODEL_WEIGHTS = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
MODE = args.mode
FREEZE_BACKBONE = args.freeze_backbone


# with open("H:\Moayad\deeptune-scratch\deeptune_results\output_directory_trainval_20250701_1659\custom_siglip_config.json") as f:
#     cfg = json.load(f)
# model = CustomSiglipModel(
#     base_model=base_model,
#     added_layers=cfg["added_layers"],
#     embedding_dim=cfg["embedding_dim"],
#     num_classes=cfg["num_classes"]
# )
# model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

if USE_PEFT:
    
    base_model, PROCESSOR = load_peft_siglip_offline()
    MODEL = CustomSigLIPWithPeft(
         base_model=base_model,
         added_layers=ADDED_LAYERS,
         embedding_dim=EMBED_SIZE,
         num_classes=NUM_CLASSES,
         freeze_backbone=FREEZE_BACKBONE
     )
    
    MODEL.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    args.model = 'PEFT-SIGLIP'
    
else:
    base_model, PROCESSOR = load_siglip_offline()
    MODEL = CustomSiglipModel(
         base_model=base_model,
         added_layers=ADDED_LAYERS,
         embedding_dim=EMBED_SIZE,
         num_classes=NUM_CLASSES,
         freeze_backbone=FREEZE_BACKBONE
     )
    MODEL.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    args.model = 'SIGLIP'

# WE load the dataset, split it and save them in the current directory (for reproducibility) if they aren't already saved.
df = pd.read_parquet(TEST_DATASET_PATH)

test_dataset = ParquetImageDataset(parquet_file=TEST_DATASET_PATH, processor=PROCESSOR)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class TestTrainer:
    
    def __init__(self, model, batch_size):
        
        self.model = model
        self.batch_size = batch_size
        
        self.model.to(DEVICE)
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.criterion = nn.CrossEntropyLoss()
        
    def test(self):
        
        test_accuracy=0.0
        test_loss=0.0
        total, correct= 0,0
        
        test_pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        )
        
        all_labels=[]
        all_predictions=[]
        all_probs=[]
        
        with torch.no_grad():
            for _, (image_pixels, labels) in test_pbar:
                
                self.model.eval()
                
                image_pixels,labels = image_pixels.to(DEVICE), labels.to(DEVICE)

                logits = self.model({'pixel_values': image_pixels}) 
                
                probs = torch.softmax(logits, 1)
                
                _, predicted = torch.max(probs,1)

                loss = self.criterion(logits, labels)
                
                test_loss += loss.item()
                
                # Store all probabilities, predictions, and labels to the CPU memory
                all_probs.append(probs.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                                
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
                
        all_probs = np.concatenate(all_probs, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        test_accuracy = correct / total
        test_loss = test_loss / len(test_loader)
        
        
        metrics_dict = {}
        
        metrics_dict['accuracy'] = test_accuracy
        metrics_dict['loss'] = test_loss
        
        report = classification_report(
            y_true = all_labels,
            y_pred = all_predictions,
            output_dict=True
        )
        
        print('Metrics Dictionary is being computed..')
        
        metrics_dict.update(report)
        
        metrics_dict['confusion_matrix'] = confusion_matrix(all_labels, all_predictions)
        
        try:
            metrics_dict['auroc'] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            metrics_dict['auroc'] = "AUROC not applicable for this setup"
            
            

        print(test_accuracy,test_loss)
        save_cli_args(args, TEST_OUTPUT_DIR,mode='test')
        
        print(metrics_dict)
        return metrics_dict
    
if __name__ == "__main__":
    
    test_trainer = TestTrainer(MODEL,BATCH_SIZE)
    
    test_trainer.test()
        
    