import torchvision
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class adjustedPeftResNet(nn.Module):

    def __init__(self,num_classes,pretrained_resnet=torchvision.models.resnet18,weights=ResNet18_Weights.IMAGENET1K_V1,fc1_input=512):

        super(adjustedPeftResNet, self).__init__()

        self.model = pretrained_resnet(weights=weights)
        self.num_classes = num_classes

        # this receives the output directly from the last conv layer.

        self.model.fc = nn.Linear(fc1_input, 8)

        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self,model):

        target_modules = []
        available_types = [nn.modules.conv.Conv2d, nn.modules.linear.Linear]

        for n, m in model.named_modules():
            if type(m) in available_types:
                target_modules.append(n)
        # print('Target Modules', target_modules)
        
        self.lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none",target_modules=target_modules)
        print(self.lora_config)

        peft_model = get_peft_model(model, self.lora_config)
        return peft_model

    def forward(self, x):

        x = self.peftmodel(x)


        return x
