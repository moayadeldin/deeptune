import torchvision
import os
import json
import pandas as pd
from torch import Tensor


# Kindly note that right now we pass the same transformations to ResNet and DenseNet, both trained on ImageNet
transformations = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(), # as I upload raw images

    torchvision.transforms.Resize(size=(224,224)), # resize images to the needed size of ResNet50

    torchvision.transforms.ToTensor(), # convert images to tensors

    torchvision.transforms.Normalize(
        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )    

])


def save_training_metrics(test_accuracy,output_dir):

    os.makedirs(output_dir,exist_ok=True)
    
    output_path = os.path.join(output_dir,'test_accuracy.txt')

    with open(output_path, 'w') as f:
        f.write("\nTest Accuracy:\n")
        f.write(f"{test_accuracy:.4f}\n")

def save_cli_args(args,output_dir,mode):
    
        if mode == 'train':
            
            args_dict = {
            'model': args.model,
            'num_classes': args.num_classes,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'added_layers': args.added_layers,
            'embed_size': args.embed_size,
            'input_dir': args.input_dir,
            'train_size': args.train_size,
            'val_size': args.val_size,
            'test_size':args.test_size,
            'use-peft': args.use_peft,
            'fixed-seed': args.fixed_seed
            }
        
        elif mode == 'test':
            
            args_dict = {
                'num_classes': args.num_classes,
                'use_peft': args.use_peft,
                'added_layers': args.added_layers,
                'embed_size': args.embed_size,
                'batch_size': args.batch_size,
                'input_dir': args.input_dir,
                'model_weights': args.model_weights
            }
            
        else:
            raise ValueError('Please choose one of the following as mode ["train", "test"]')
            
        
        # elif mode == 'embed':
            
        #     args_dict = {
        #         'num_classes':args.num_classes,
        #         'use_case':args.use_case,
        #         'added_layers': args.added_layers,
        #         'embed_size': args.embed_size,
        #         'batch_size': args.batch_size,
        #         'dataset_dir': args.dataset_dir,
        #         'finetuned_model_pth': args.finetuned_model_pth
                
        #     }            


        output_path = os.path.join(output_dir, 'cli_arguments.txt')
        
        with open(output_path, 'w') as f:
            f.write("Command Line Arguments:\n")
            f.write(json.dumps(args_dict, indent=4))
            
def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
        
def avg_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

""" The following class code is adopted from John's implementation of Siglip"""

class PerformanceLogger:
     
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
         
        self.log_data["epoch"].append(epoch)
        self.log_data["epoch_loss"].append(epoch_loss)
        self.log_data["epoch_accuracy"].append(epoch_accuracy)
        self.log_data["val_loss"].append(val_loss)
        self.log_data["val_accuracy"].append(val_accuracy)
        self.log_data["test_loss"].append(test_loss)
        self.log_data["test_accuracy"].append(test_accuracy)
            
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
    
     

