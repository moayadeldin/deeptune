import torchvision
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F


class adjustedResNet(nn.Module):

    def __init__(self,num_classes, added_layers, embedding_layer_size, freeze_backbone=False, weights=ResNet18_Weights.IMAGENET1K_V1,
                 pretrained_resnet=torchvision.models.resnet18,fc1_input=512):

        super(adjustedResNet, self).__init__()

        self.model = pretrained_resnet(weights)
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone

        # remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        
        if self.freeze_backbone:
            print('Backbone Parameters are frozen!')
            for param in self.model.parameters():
                param.requires_grad = False
        
        if self.added_layers == 2:
            self.fc1 = nn.Linear(fc1_input,self.embedding_layer_size)
            self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
        elif self.added_layers == 1:
            self.fc1 = nn.Linear(fc1_input, self.num_classes)
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
            
        # x = F.softmax(x, dim=1)

        return x