import torch
import torch.nn as nn
import torch.nn.functional as F

class adjustedDenseNet(nn.Module):
    

    def __init__(self,num_classes,densenet_version, added_layers, embedding_layer_size, freeze_backbone=False,
                 task_type="cls",output_dim=1):
        
        """
        
        Customised DenseNet class applying Parameter Efficient Fine Tuning with LoRA as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            densenet_version (str): Version of DenseNet you want to use.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            lora_attention_dimension (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            output_dim (int): The dimension of the output of regression model, default = 1.
            
            
        
        """

        super(adjustedDenseNet,self).__init__()
        
        # Task must be regression or classification nothing else        
        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone
        self.densenet_version = densenet_version
        if densenet_version == "densenet121":
            self.model_to_load = "densenet121"
        elif densenet_version == "densenet161":
            self.model_to_load = "densenet161"
        elif densenet_version == "densenet169":
            self.model_to_load = "densenet169"
        elif densenet_version == "densenet201":
            self.model_to_load = "densenet201"
        else:
            raise ValueError("Invalid densenet_version. Choose from 'densenet121', 'densenet161', 'densenet169', or 'densenet201'.")
        
        # remove the final connected layer by putting a placeholder
        self.model = torch.hub.load('pytorch/vision:v0.10.0', self.model_to_load, pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
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
            self.fc1 = nn.Linear(in_features,self.embedding_layer_size)
            
            if self.task_type == 'cls':
                self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.embedding_layer_size, self.output_dim)
                
                
        elif self.added_layers == 1:
            
            if self.task_type == 'cls':
                self.fc1 = nn.Linear(in_features, self.num_classes)
            else:
                self.fc1 = nn.Linear(in_features,self.output_dim)
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
        
        if self.added_layers == 2 and extract_embed:
            
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

