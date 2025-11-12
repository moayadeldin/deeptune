import json
import logging
import sys
import torch
import torch.nn as nn
from utils import save_process_times
from sklearn.metrics import classification_report, roc_auc_score
from tqdm.auto import tqdm
import time

from options import DEVICE


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
    
    def __init__(
        self,
        model,
        batch_size,
        test_loader,
        mode,
        output_dir,
        device=DEVICE
    ):
        self.model = model
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.mode = mode

        self.output_dir = output_dir

        self.device = device
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        if self.mode == 'cls':    
            self.criterion = nn.CrossEntropyLoss()
        else: # then regression if not classification
            self.criterion = nn.MSELoss()
        
        
    def test(self, best_model_weights_path=None):
        
        start_time = time.time()
        
        # Load the best model weights if they are provided
        
        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')
            
        self.model.to(self.device)
        # Initialize the metrics for validation
        test_accuracy=0.0
        test_loss=0.0
        total,correct = 0,0
        
        test_pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        
        all_labels = []
        all_predictions=[]
        all_probs=[]

        with torch.no_grad():
            for i, (inputs, labels,*_) in test_pbar:
                
                self.model.eval()
                if isinstance(self.criterion, nn.MSELoss): # if regression then we must rehape the target tensor    
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = labels.view(-1,1).float() # [batch_size,1]
                else:
                    inputs, labels = inputs.to(self.device),labels.to(self.device)
                
                # Apply forward pass and accumulate loss
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                # If mode is classification then we need to calculate accuracy
                if self.mode == "cls":
                    probs = torch.softmax(outputs, 1)
                    _, predicted = torch.max(probs, 1)
                    
                    total += labels.size(0)
                    correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
                    
                    # Store classification outputs
                    all_probs.append(probs)
                    all_predictions.append(predicted)
                    all_labels.append(labels)
        
        # Compute loss for both modes, if classification then we need to compute also the other metrics as accuracy, AUROC, and classification report.
        test_loss = test_loss / len(self.test_loader)
        metrics_dict = {"loss": test_loss}
        
        if self.mode == "cls":
            test_accuracy = (correct / total) * 100
            metrics_dict["accuracy"] = test_accuracy
            
            # all_probs = np.concatenate(all_probs, axis=0)
            # all_predictions = np.concatenate(all_predictions, axis=0)
            # all_labels = np.concatenate(all_labels, axis=0)
            all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
            all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
            
            report = classification_report(y_true=all_labels, y_pred=all_predictions, output_dict=True)
            metrics_dict.update(report)
            
            try:
                metrics_dict["auroc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
            except ValueError:
                metrics_dict["auroc"] = "AUROC not applicable for this setup"
            
            # print(test_accuracy, test_loss)
            self.logger.info(f"The test accuracy is: {test_accuracy}, while the test loss is: {test_loss}")

            with open(self.output_dir / "full_metrics.json", 'w') as f:
                json.dump(metrics_dict, f, indent=4)


            return metrics_dict
                
            
        end_time = time.time()
        total_time = end_time - start_time
        save_process_times(epoch_times=1, total_duration=total_time, outdir=self.output_dir, process="evaluation")


        
        print(metrics_dict)