import json
import numpy as np
import os
import pandas as pd
import random
import torch
import torchvision
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from tabulate import tabulate
from transformers import GPT2Tokenizer, GPT2Model
from transformers import BertModel, BertTokenizer
import shutil
from src.nlp.gpt2 import AdjustedGPT2Model
from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT

# Kindly note that right now we pass the same transformations to ResNet, Swin and DenseNet, both trained on ImageNet
transformations = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(), # as I upload raw images

    torchvision.transforms.Resize(size=(224,224)), # resize images to the needed size of ResNet50

    torchvision.transforms.ToTensor(), # convert images to tensors

    torchvision.transforms.Normalize(
        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )    
])

def fixed_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
    
        seed (int): The seed value to set for all random number generators.
    """
    
    random.seed(seed)
    print(f"random.seed({seed}) set.")

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"os.environ['PYTHONHASHSEED'] set to {seed}.")

    np.random.seed(seed)
    print(f"np.random.seed({seed}) set.")

    torch.manual_seed(seed)
    print(f"torch.manual_seed({seed}) set.")

    torch.cuda.manual_seed(seed)
    print(f"torch.cuda.manual_seed({seed}) set.")

    torch.cuda.manual_seed_all(seed)
    print(f"torch.cuda.manual_seed_all({seed}) set.")

    torch.backends.cudnn.benchmark = False
    print("torch.backends.cudnn.benchmark set to False.")

    torch.backends.cudnn.deterministic = True
    print("torch.backends.cudnn.deterministic set to True.")


def load_finetunedbert_model(model_dir, use_peft=None, use_case=None):
    """
    Load the model for inference. Here we unpack what we packed in the previous function.
    
    Args:
        model_dir (str): The path to the saved model.
    """

    cfg_path = os.path.join(model_dir, "model_config.json")
    with open(cfg_path, "r") as f:
        model_config = json.load(f)

    if use_peft is None:
        use_peft = bool(model_config.get("use_peft", False))
    if use_case is None:
        use_case = model_config.get("use_case", "finetuned")

    bert_model = BertModel.from_pretrained(os.path.join(model_dir, "bert_model"))
    print(f"Loaded BERT model from {os.path.join(model_dir, 'bert_model')}")

    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    print(f"Loaded tokenizer from {os.path.join(model_dir, 'tokenizer')}")

    if use_peft or use_case == "peft":
        model = CustomMultilingualPeftBERT(
            num_classes=model_config["num_classes"],
            added_layers=model_config["added_layers"],
            embedding_layer=model_config["embedding_layer"]
        )
    else:
        model = CustomMultilingualBERT(
            num_classes=model_config["num_classes"],
            added_layers=model_config["added_layers"],
            embedding_layer=model_config["embedding_layer"]
        )

    weights_path = os.path.join(model_dir, "model_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Loaded model weights from {weights_path}")

    return model, tokenizer


def load_finetuned_gpt2(model_dir):
    """
    Load the fine-tuned Adjusted GPT-2 model with Conv1D head.

    Args:
        model_dir (str): Path to the directory containing the model files.

    Returns:
        model (AdjustedGPT2Model): The loaded model.
        tokenizer (GPT2Tokenizer): The tokenizer used during training.
    """
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)

    output_dim = model_config.get("output_dim", 1000)

    # Load GPT-2 backbone and tokenizer
    gpt_model = GPT2Model.from_pretrained(os.path.join(model_dir, "gpt2_model"))
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded GPT-2 model from {os.path.join(model_dir, 'gpt2_model')}")
    print(f"Loaded tokenizer from {os.path.join(model_dir, 'tokenizer')}")

    # Build full model
    model = AdjustedGPT2Model(gpt_model=gpt_model, output_dim=output_dim)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pth")))
    print(f"Loaded model weights from {os.path.join(model_dir, 'model_weights.pth')}")

    return model, tokenizer


""" The following class code is adopted mainly from: https://github.com/johnkxl/peft4vision"""

class PerformanceLogger:
    
     """
    A class to log the performance of the model during training and validation.
    
    Attributes:
        log_data (dict): The dictionary to store the logged data.
        output_dir (str): The path to save the logged data
     """

     def _to_scalar(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return x
     
     def __init__(self,output_dir):
          
          self.log_data = {
               "epoch":[],
                "epoch_loss": [],
                "epoch_accuracy": [],
                "val_loss": [],
                "val_accuracy": [],
                "test_loss":[],
                "test_accuracy":[]
            }
          
          self.output_dir = output_dir
     
     def log_performance(self, epoch, epoch_loss, epoch_accuracy, val_loss, val_accuracy, test_loss, test_accuracy):
        """
        Add the performance metrics to the log data dictionary that will be converted to CSV at the end.
        """

        self.log_data["epoch"].append(self._to_scalar(epoch))
        self.log_data["epoch_loss"].append(self._to_scalar(epoch_loss))
        self.log_data["epoch_accuracy"].append(self._to_scalar(epoch_accuracy))
        self.log_data["val_loss"].append(self._to_scalar(val_loss))
        self.log_data["val_accuracy"].append(self._to_scalar(val_accuracy))
        self.log_data["test_loss"].append(self._to_scalar(test_loss))
        self.log_data["test_accuracy"].append(self._to_scalar(test_accuracy))

            
     def log_epoch(self, epoch, epoch_loss, epoch_accuracy, val_loss, val_accuracy):
        """Log end-of-epoch metrics."""
        self.log_performance(
            epoch=epoch,
            epoch_loss=epoch_loss,
            epoch_accuracy=epoch_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            test_loss=None,
            test_accuracy=None
        )
        
     
     def save_to_csv(self, file_path):
        """Save the metrics data to a CSV file."""
        os.makedirs(self.output_dir,exist_ok=True)
        df = pd.DataFrame(self.log_data)
        df.to_csv(file_path, index=False)
        print(f"Saved performance log to {file_path}")
                
     def save_to_parquet(self, file_path):
        os.makedirs(self.output_dir,exist_ok=True)
        """Save the logged data to a Parquet file."""
        df = pd.DataFrame(self.log_data)
        df.to_parquet(file_path, index=False)
        print(f"Saved performance log to {file_path}")
class PerformanceLoggerCallback(Callback):

    def __init__(self,performance_logger):

        super().__init__()
        self.performance_logger = performance_logger

    def on_validation_end(self,trainer,pl_module):

        epoch = trainer.current_epoch + 1

        metrics = trainer.callback_metrics

        # get losses and accuracy from metrics

        print(metrics)

        train_loss = metrics.get("train_loss",None) 
        train_accuracy = metrics.get("train_accuracy",None)
        val_loss = metrics.get("valid_loss",None)
        val_accuracy = metrics.get("valid_accuracy",None)

        print(f"[Epoch {epoch}] Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

        self.performance_logger.log_epoch(
            epoch=epoch,
            epoch_loss=train_loss,
            epoch_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

# def add_datetime_column_to_predictions(
#     pred_df,
#     train_df_path,
#     time_idx_col,
#     datetime_col
# ):
#     """
#     Add a datetime column to the prediction DataFrame by mapping from the original input CSV.

#     Returns:
#         pred_df (DataFrame): DataFrame with a new datetime column added.
#     """
#     original_df = pd.read_parquet(train_df_path)

#     original_df[time_idx_col] = pd.to_datetime(original_df[time_idx_col])

#     original_df = original_df.sort_values(time_idx_col).reset_index(drop=True)
#     original_df["int_time_idx"] = original_df.index.astype(int)

#     time_idx_to_date = dict(zip(original_df["int_time_idx"], original_df[time_idx_col]))

#     pred_df[datetime_col] = pred_df[time_idx_col].map(time_idx_to_date)

#     return pred_df

def tensor_to_list(obj):
    """Convert Tensors (or nested tensors in dicts/lists) to Python lists."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def save_timeseries_prediction_to_json(prediction, outdir):
    
    """
    Save PI Prediction Object to a JSON file.
    """
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    output_data =  {
        "model_prediction":tensor_to_list(prediction.output),
        "input_dictionary": tensor_to_list(prediction.x),
        "predicted_timesteps": tensor_to_list(prediction.decoder_lengths),
        "ground_truth_target": tensor_to_list(
            prediction.y if prediction.y is not None else prediction.x.get("decoder_target")
        ),
    }
    
    filename= f'{outdir}/prediction_output.json'
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Full Prediction output data information is saved to {filename}")

def date_id(prefix="deeptune", root_dir="."):
    """
    Generate a date-based ID for parent folder. Increments if a folder already exists.
    Example: deeptune-20251110-exp1, deeptune-20251110-exp2, ...
    """
    date_str = datetime.now().strftime("%Y%m%d")
    base_prefix = f"{prefix}-{date_str}-exp"
    n = 1

    # Increment until unused folder name found
    while (Path(root_dir) / f"{base_prefix}{n}").exists():
        n += 1

    final_name = f"{base_prefix}{n}"
    return final_name

def print_metrics_table(metrics_dict, embed_shape, embed_path):
    table = [
        ["Loss", "-", "-", "-", "-", f"{metrics_dict.get('loss', 0):.4f}"],
        ["Accuracy", "-", "-", "-", "-", f"{metrics_dict.get('accuracy', 0):.4f}"],
    ]

    for key in metrics_dict:
        if key.isdigit():  # class keys like '1', '3'
            class_metrics = metrics_dict[key]
            table.append([
                f"Class {key}",
                class_metrics.get('precision', 0.0),
                class_metrics.get('recall', 0.0),
                class_metrics.get('f1-score', 0.0),
                class_metrics.get('support', 0.0),
                "-"
            ])

    table.append(["AUROC", "-", "-", "-", "-", metrics_dict.get('auroc', '-')])

    table.append(["Embedding Matrix Dimension", "-", "-", "-", "-", str(embed_shape)])
    table.append(["Output Location", "-", "-", "-", "-", str(embed_path)])


    print(tabulate(
        table,
        headers=["Metric", "Precision", "Recall", "F1-Score", "Support", "Value"],
        tablefmt="github"
    ))

