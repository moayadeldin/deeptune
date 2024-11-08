import torchvision
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F


class adjustedResNet(nn.Module):

    def __init__(self,num_classes, weights=ResNet50_Weights.IMAGENET1K_V1, 
                 pretrained_resnet=torchvision.models.resnet50,fc1_input=2048):

        super(adjustedResNet, self).__init__()

        self.model = pretrained_resnet(weights)
        self.num_classes = num_classes

        # remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc1_input,1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):

        x = self.model(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x