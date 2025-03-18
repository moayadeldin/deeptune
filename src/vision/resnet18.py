import torchvision
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F


class adjustedResNet(nn.Module):

    def __init__(self,num_classes, added_layers=2, embedding_layer_size=1000, freeze_backbone=False, weights=ResNet18_Weights.IMAGENET1K_V1,
                 pretrained_resnet=torchvision.models.resnet18,fc1_input=512, task_type="cls",output_dim=1):
        """
        
        Customised ResNet class as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            embedding_layer_size (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            fc1_input (int): The size of the input of the last layer before we chop it.
            pretrained_resnet (torchvision.models): Determine which ResNet model you want to use.
            weights (ResNet_Weights.IMAGENET1K_V1): Determine which ResNet weights you want to use.
            output_dim (int): The dimension of the output of regression model, default = 1.
            
            
        
        """

        super(adjustedResNet, self).__init__()

        # Task must be regression or classification nothing else
        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"


        self.model = pretrained_resnet(weights)
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone
        
        # Additional parameters for regression
        self.task_type = task_type
        self.output_dim = output_dim

        # Remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        
        if self.freeze_backbone:
            print('Backbone Parameters are frozen!')
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Add the additional layers according to prompt.
        
        if self.added_layers == 2:
            self.fc1 = nn.Linear(fc1_input,self.embedding_layer_size)
            
            if self.task_type == 'cls':
                self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.embedding_layer_size, self.output_dim)
                
                
        elif self.added_layers == 1:
            
            if self.task_type == 'cls':
                self.fc1 = nn.Linear(fc1_input, self.num_classes)
            else:
                self.fc1 = nn.Linear(fc1_input,self.output_dim)
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