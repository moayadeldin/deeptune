import json
import logging
import sys
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datasets.image_datasets import ParquetImageDataset
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from src.vision.siglip import CustomSiglipModel, CustomSigLIPWithPeft, load_siglip_processor_offline, load_siglip_variant

from utils import save_cli_args, UseCase, save_process_times


def evaluate_siglip(
    test_dataset_path: Path,
    model_weights: Path,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    use_peft: bool,
    batch_size: int,
    device: torch.device,
    outdir: Path,
) -> None:
    use_case = UseCase.PEFT if use_peft else UseCase.FINETUNED
    model = load_siglip_variant(
        use_case=use_case,
        num_classes=num_classes,
        added_layers=added_layers,
        embed_size=embed_size,
        freeze_backbone=False,  # no effect during inference
        model_weights=model_weights,
        device=device,
    )

    processor = load_siglip_processor_offline()

    test_dataset = ParquetImageDataset.from_parquet(parquet_file=test_dataset_path, processor=processor)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=PERSIST_WORK,
    )

    test_trainer = TestTrainer(
        model=model,
        data_loader=test_loader,
        output_dir=outdir,
        device=device,
        use_peft=use_peft,
    )
    
    test_trainer.test()


class TestTrainer:
    
    def __init__(
        self,
        model,
        data_loader: DataLoader,
        output_dir: Path,
        device: torch.device,
        use_peft: bool,
    ):
        
        self.model = model
        self.model.to(device)

        self.data_loader = data_loader
        
        self.output_dir = output_dir
        self.device = device
        self.use_peft = use_peft
        
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        self.criterion = nn.CrossEntropyLoss()
        
    
    
    def test(self):
        test_accuracy=0.0
        test_loss=0.0
        total, correct= 0,0
        
        test_pbar = tqdm(
            enumerate(self.data_loader),
            total=len(self.data_loader),
        )
        
        all_labels=[]
        all_predictions=[]
        all_probs=[]
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for _, (pixel_values, labels) in test_pbar:
                
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)

                with autocast(device_type=self.device.type):
                    logits = self.model({"pixel_values": pixel_values})
                    
                probs = torch.softmax(logits, 1)
                _, predicted = torch.max(probs,1)
                loss = self.criterion(logits, labels)
                
                test_loss += loss.item()
                
                all_probs.append(probs)
                all_predictions.append(predicted)
                all_labels.append(labels)
                                 
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        # Store all probabilities, predictions, and labels to the CPU memory
        all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
        all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        test_accuracy = correct / total
        test_loss = test_loss / len(self.data_loader)
        
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
        
        metrics_dict['confusion_matrix'] = confusion_matrix(all_labels, all_predictions).tolist()
        
        try:
            metrics_dict['auroc'] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            metrics_dict['auroc'] = "AUROC not applicable for this setup"
            
        print(test_accuracy, test_loss)
        self.logger.info(f"Test accuracy: {test_accuracy}%")

        with open(self.output_dir / "full_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        end_time = time.time()
        total_time = end_time - start_time
        save_process_times(epoch_times=1, total_duration=total_time, outdir=self.output_dir, process="evaluation")
        
        return metrics_dict