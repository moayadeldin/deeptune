import torchvision
from torchvision.models import Swin_T_Weights,Swin_S_Weights,Swin_B_Weights
import torch.nn as nn
import torch.nn.functional as F

class adjustedSwin(nn.Module):
    
    def __init__(self,num_classes,swin_version, added_layers=2, embedding_layer_size=1000, freeze_backbone=False,task_type="cls",output_dim=1):
        
        """
        
        Customised Swin class as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            swin_version (str): Version of Swin you want to use.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            embedding_layer_size (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            in_features (int): The size of the input of the last layer before we chop it.
            output_dim (int): The dimension of the output of regression model, default = 1.
            
        
        """
        
        super(adjustedSwin, self).__init__()
        
        # Task must be regression or classification nothing else
        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

        self.swin_version = swin_version
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone

        if swin_version == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_t(weights=weights)
            self.model = pretrained_swin
        elif swin_version == "swin_s":
            weights = Swin_S_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_s(weights=weights)
            self.model = pretrained_swin
        elif swin_version == "swin_b":
            weights = Swin_B_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_b(weights=weights)
            self.model = pretrained_swin
        elif swin_version == "dummy": # This is for testing purposes only.
            print("#########################################")
            print("Initializing dummy Swin model WITHOUT pretrained weights.")
            pretrained_swin = torchvision.models.swin_b(weights=None)
            self.model = pretrained_swin
        else:
            raise ValueError("Invalid swin_version. Choose from 'swin_t', 'swin_s', or 'swin_b'.")
        # Get the input size of the last layer before we chop it
        self.in_features = self.model.head.in_features
        
        
        # remove the final connected layer by putting a placeholder
        self.model.head = nn.Identity()
        self.flatten = nn.Flatten()
        
        # additional parameters for regression
        self.task_type = task_type
        self.output_dim = output_dim
        
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        if self.freeze_backbone:
            print('Backbone Parameters are frozen!')
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Add the additional layers according to prompt.
        
        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.in_features,self.embedding_layer_size)
            
            if self.task_type == 'cls':
                self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.embedding_layer_size, self.output_dim)
                
                
        elif self.added_layers == 1:
            
            if self.task_type == 'cls':
                self.fc1 = nn.Linear(self.in_features, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.in_features,self.output_dim)
        else:
            self.fc1 = None
            
    def forward(self, x, extract_embed=False):
        
        """
        Now what we want is:
        
        - If added_layers = 0 or added_layers = 1, we want to return the raw features, which is basically before fc1 if self.added_layers = 1
        
        - If added_layers = 2, we want to return the embeddings after the first fully connected layer (before the second one for classes)
        
        The implementation below ensures the following paradigm.
        """
        
        
        x = self.model(x)
        x = self.flatten(x)
        
        if self.added_layers == 1 and extract_embed:
            
            # return raw features before fc1
            return x
        
        elif self.added_layers == 2 and extract_embed:
            
            # return the embeddings after fc1 but before fc2
            x = self.fc1(x)
            return x
        

        if self.added_layers == 2:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            
        elif self.added_layers == 1:
            x = self.fc1(x)

        return x
    
    
    