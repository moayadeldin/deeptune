from utilities import save_test_metrics
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import logging
import options

parser = options.parser
DEVICE = options.DEVICE
TEST_OUTPUT_DIR = options.TEST_OUTPUT_DIR
args = parser.parse_args()

test_loader = args.test_set_input_dir

class TestTrainer:
    
    def __init__(self, model, batch_size,test_loader):
        
        self.model = model
        self.batch_size = batch_size
        self.test_loader = test_loader
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.criterion = nn.CrossEntropyLoss()
        
        
    def test(self, best_model_weights_path=None):
        
        if best_model_weights_path is None:
            pass
        else:
            self.model.load_state_dict(torch.load(best_model_weights_path))
            print('Model into the path is loaded.')
            
        self.model.to(DEVICE)
        
        test_accuracy=0.0
        test_loss=0.0
        total,correct = 0,0
        
        test_pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        
        all_labels = []
        all_predictions=[]
        all_probs=[]

        with torch.no_grad():

            for _, (input, labels) in test_pbar:

                self.model.eval()
                
                inputs, labels = input.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs,labels)
                
                probs = torch.softmax(outputs,1)
                
                _, predicted = torch.max(probs,1)
                
                test_loss += loss.item()
                
                total += labels.size(0)
                correct += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
                
                # Store all probabilities, predictions, and labels to the CPU memory
                all_probs.append(probs.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                                
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                
        all_probs = np.concatenate(all_probs, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
                
                
        test_accuracy = ( correct / total ) * 100
        test_loss = test_loss / len(self.test_loader)
        
        metrics_dict = {}
        
        metrics_dict['accuracy'] = test_accuracy
        metrics_dict['loss'] = test_loss
        
        report = classification_report(
            y_true = all_labels,
            y_pred = all_predictions,
            output_dict=True
        )
        
        metrics_dict.update(report)
        
        try:
            metrics_dict['auroc'] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            metrics_dict['auroc'] = "AUROC not applicable for this setup"
            
        
        print(test_accuracy, test_loss)
        
        self.logger.info(f"Test accuracy: {(test_accuracy)}%")

        save_test_metrics(test_accuracy=test_accuracy,output_dir=TEST_OUTPUT_DIR)
    
        print(metrics_dict)