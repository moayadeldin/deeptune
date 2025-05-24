from utilities import save_test_metrics,get_args
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import sys
import logging
import options


# Initialize the needed variables either from the CLI user sents or from the device.
parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = get_args()
MODE = args.mode

test_loader = args.test_set_input_dir

class TestTrainer:
    
    """
    Performs Testing on the input image dataset.
    
    Attributes:
    
            model (PyTorch Model): The model we are loading from the src file, whether it is for transfer learning with PEFT Or without.
            test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
            batch_size (int): The batch size for the test set.
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            performance_logger (PerformanceLogger): Logger instance for tracking testing.
            logger (logging.Logger): Logger instance for tracking test progress.
    """
    
    def __init__(self, model, batch_size,test_loader):
        
        self.model = model
        self.batch_size = batch_size
        self.test_loader = test_loader
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        if MODE == 'cls':    
            self.criterion = nn.CrossEntropyLoss()
        else: # then regression if not classification
            self.criterion = nn.MSELoss()
        
        
    def test(self, best_model_weights_path=None):
        
        # Load the best model weights if they are provided
        
        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')
            
        self.model.to(DEVICE)
        # Initialize the metrics for validation
        test_accuracy=0.0
        test_loss=0.0
        total,correct = 0,0
        
        test_pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        
        all_labels = []
        all_predictions=[]
        all_probs=[]

        with torch.no_grad():
            for _, (inputs, labels) in test_pbar:
                
                self.model.eval()
                if isinstance(self.criterion, nn.MSELoss): # if regression then we must rehape the target tensor    
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    labels = labels.view(-1,1).float() # [batch_size,1]
                else:
                    inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
                
                # Apply forward pass and accumulate loss
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                
                # If mode is classification then we need to calculate accuracy
                if MODE == "cls":
                    probs = torch.softmax(outputs, 1)
                    _, predicted = torch.max(probs, 1)
                    
                    total += labels.size(0)
                    correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
                    
                    # Store classification outputs
                    all_probs.append(probs.cpu().numpy())
                    all_predictions.append(predicted.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
        
        # Compute loss for both modes, if classification then we need to compute also the other metrics as accuracy, AUROC, and classification report.
        test_loss = test_loss / len(self.test_loader)
        metrics_dict = {"loss": test_loss}
        
        if MODE == "cls":
            test_accuracy = (correct / total) * 100
            metrics_dict["accuracy"] = test_accuracy
            
            all_probs = np.concatenate(all_probs, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            report = classification_report(y_true=all_labels, y_pred=all_predictions, output_dict=True)
            metrics_dict.update(report)
            
            try:
                metrics_dict["auroc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
            except ValueError:
                metrics_dict["auroc"] = "AUROC not applicable for this setup"
            
            print(test_accuracy, test_loss)
            self.logger.info(f"Test accuracy: {test_accuracy}%")
            save_test_metrics(test_accuracy=test_accuracy, output_dir=TEST_OUTPUT_DIR)
        
        print(metrics_dict)