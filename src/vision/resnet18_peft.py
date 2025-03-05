import torchvision
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class adjustedPeftResNet(nn.Module):

    def __init__(self,num_classes, added_layers, lora_attention_dimension, freeze_backbone=False, weights=ResNet18_Weights.IMAGENET1K_V1,
                 pretrained_resnet=torchvision.models.resnet18,fc1_input=512):

        super(adjustedPeftResNet, self).__init__()

        self.model = pretrained_resnet(weights)
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
        self.freeze_backbone = freeze_backbone

        # remove the final connected layer by putting a placeholder
        self.model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        
        if self.freeze_backbone:
            
            for param in self.model.parameters():
                print('Backbone Parameters are frozen!')
                param.requires_grad = False
            
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        
        if self.added_layers == 2:
            self.fc1 = nn.Linear(fc1_input,self.lora_attention_dimension)
            self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
        elif self.added_layers == 1:
            self.fc1 = nn.Linear(fc1_input, self.num_classes)
        else:
            self.fc1 = None
            
        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self,model):

        target_modules = []
        available_types = [nn.modules.conv.Conv2d, nn.modules.linear.Linear, nn.modules.Flatten]

        for n, m in model.named_modules():
            if type(m) in available_types:
                target_modules.append(n)
        print('Target Modules', target_modules)

        """
        To get more insights on how the LoRA weights could affect the performance, please refer to the following documentation:
        https://huggingface.co/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraConfig
        """
        
        self.lora_config = LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1, bias="none",target_modules=target_modules)
        print(self.lora_config)

        peft_model = get_peft_model(model, self.lora_config)
        return peft_model

    def forward(self, x,extract_embed=False):
        x = self.peftmodel(x)
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

