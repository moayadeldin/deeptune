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
        if best_model_weights_path is not None:
            self.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        total_loss, total = 0.0, 0
        all_labels, all_predictions, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels, *_ in tqdm(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.mode == 'reg':
                    labels = labels.view(-1, 1).float()
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                all_labels.append(labels.detach().cpu())
                if self.mode == 'cls':
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probs.append(probabilities.cpu())
                    all_predictions.append(probabilities.argmax(dim=1).cpu())
                else:
                    all_predictions.append(outputs.detach().cpu())
        if not total:
            raise ValueError('The evaluation dataset is empty.')
        metrics = {'loss': total_loss / total}
        labels = torch.cat(all_labels).numpy()
        predictions = torch.cat(all_predictions).numpy()
        if self.mode == 'cls':
            probabilities = torch.cat(all_probs).numpy()
            metrics.update(classification_report(labels, predictions, output_dict=True, zero_division=0))
            try:
                scores = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
                metrics['auroc'] = float(roc_auc_score(labels, scores, multi_class='ovr'))
            except ValueError:
                metrics['auroc'] = None
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            metrics['mae'] = float(mean_absolute_error(labels, predictions))
            metrics['rmse'] = float(metrics['loss'] ** 0.5)
            metrics['r2'] = float(r2_score(labels, predictions)) if total >= 2 else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / 'full_metrics.json', 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, indent=2)
        save_process_times(epoch_times=1, total_duration=time.time() - start_time,
                           outdir=self.output_dir, process='evaluation')
        return metrics
