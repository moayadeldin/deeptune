import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class adjustedConvNext(nn.Module):

    def __init__(self, num_classes, convnext_version, added_layers=2, embedding_layer_size=1000, freeze_backbone=False, task_type="cls", output_dim=1):
        """
        Customised ConvNeXt class as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            convnext_version (str): Version of ConvNeXt you want to use.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            embedding_layer_size (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            output_dim (int): The dimension of the output of regression model, default = 1.
        """

        super(adjustedConvNext, self).__init__()

        # Task must be regression or classification nothing else
        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone
        self.convnext_version = convnext_version
        
        # Load the base model
        if convnext_version == "convnext_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        elif convnext_version == "convnext_small":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        elif convnext_version == "convnext_base":
            self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        elif convnext_version == "convnext_large":
            self.model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        elif convnext_version == "dummy": # This is for testing purposes only.
            self.model = models.convnext_tiny(weights=None)
        else:
            raise ValueError("Invalid convnext_version. Choose from 'convnext_tiny', 'convnext_small', 'convnext_base', or 'convnext_large'.")
        
        # Get the input size of the last layer before we chop it
        # ConvNeXt classifier is: (0) LayerNorm, (1) Flatten, (2) Linear
        self.fc1_input = self.model.classifier[2].in_features
        
        # Additional parameters for regression
        self.task_type = task_type
        self.output_dim = output_dim

        # Remove the final connected layer by putting a placeholder
        # This keeps the LayerNorm and Flatten intact!
        self.model.classifier[2] = nn.Identity()
        self.flatten = nn.Flatten()
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        if self.freeze_backbone:
            print('Backbone Parameters are frozen!')
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Add the additional layers according to prompt.
        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.fc1_input, self.embedding_layer_size)
            
            if self.task_type == 'cls':
                self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.embedding_layer_size, self.output_dim)
                
        elif self.added_layers == 1:
            if self.task_type == 'cls':
                self.fc1 = nn.Linear(self.fc1_input, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.fc1_input, self.output_dim)
        else:
            self.fc1 = None

    def forward(self, x, extract_embed=False):
        """
        Now what we want is:
        - If added_layers = 0 or added_layers = 1, we want to return the raw features, which is basically before fc1 if self.added_layers = 1
        - If added_layers = 2, we want to return the embeddings after the first fully connected layer (before the second one for classes)
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