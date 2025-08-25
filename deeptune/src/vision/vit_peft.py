import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

class adjustedViTPeft(nn.Module):
        def __init__(self,num_classes,vit_version, added_layers=2, lora_attention_dimension=1000, freeze_backbone=False, task_type="cls",output_dim=1):
            """
            Customised ViT class as part of DeepTune proposed Adjustments.
            
            Args:
                num_classes (int) : Number of classes in your dataset.
                vit_version (str): Version of ViT you want to use.
                added_layers (int) : Number of additional layers you want to add while finetuning your model
                lora_attention_dimension (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
                freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
                task_type (str): Determine whether you want to classification or regression.
                output_dim (int): The dimension of the output of regression model, default = 1.
                
            """

            super(adjustedViTPeft, self).__init__()

            # Task must be regression or classification nothing else
            assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

            self.num_classes = num_classes
            self.added_layers = added_layers
            self.lora_attention_dimension = lora_attention_dimension
            self.freeze_backbone = freeze_backbone
            self.vit_version = vit_version

            if vit_version == "vit_b_16":
                self.model = torchvision.models.vit_b_16(weights="DEFAULT")
            elif vit_version == "vit_b_32":
                self.model = torchvision.models.vit_b_32(weights="DEFAULT")
            elif vit_version == "vit_l_16":
                self.model = torchvision.models.vit_l_16(weights="DEFAULT")
            elif vit_version == "vit_l_32":
                self.model = torchvision.models.vit_l_32(weights="DEFAULT")
            elif vit_version == "vit_h_14":
                self.model = torchvision.models.vit_h_14(weights="DEFAULT")
            elif vit_version == "dummy": # This is for testing purposes only.
                self.model = torchvision.models.vit_h_14(weights=None)
            else:
                raise ValueError("Invalid vit_version. Choose from 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32' or 'vit_h_14'.")
            
            # Get the input size of the last layer before we chop it
            self.fc1_input = self.model.heads.head.in_features

            # remove the final connected layer by putting a placeholder
            self.model.heads.head = nn.Identity()
            self.flatten = nn.Flatten()

            # Additional parameters for regression
            self.task_type = task_type
            self.output_dim = output_dim

            # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
            
            if self.freeze_backbone:
                print('Backbone Parameters are frozen!')
                for param in self.model.parameters():
                    param.requires_grad = False
                    
            # Add the additional layers according to prompt.
            
            if self.added_layers == 2:
                self.fc1 = nn.Linear(self.fc1_input,self.lora_attention_dimension)
                
                if self.task_type == 'cls':
                    self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
                else:
                    self.fc2 = nn.Linear(self.lora_attention_dimension, self.output_dim)
                    
                    
            elif self.added_layers == 1:
                
                if self.task_type == 'cls':
                    self.fc1 = nn.Linear(self.fc1_input, self.num_classes)
                else:
                    self.fc1 = nn.Linear(self.fc1_input,self.output_dim)
            else:
                self.fc1 = None

            # Apply PEFT
            self.peftmodel = self.applyPEFT(self.model)

        def applyPEFT(self,model):
        
            """
            Apply PEFT with LoRA on the customised model.
            
            Arguments:
                model (torchvision.models): The adjusted ResNet model before applying PEFT On
                
            Returns:
            
                peft_model: PEFTed adjusted ResNet
            """

            # target_modules is actually the parts of the network we have applied PEFT on
            target_modules = []
            # available_types are the networks that support PEFT optimization
            available_types = [nn.modules.conv.Conv2d, nn.modules.linear.Linear, nn.modules.Flatten]

            # loop through the model and check what layers in it we may apply PEFT on and them to target_modules
            for n, m in model.named_modules():
                if type(m) in available_types:
                    target_modules.append(n)
            print('Target Modules', target_modules)

            """
            To get more insights on how the LoRA weights could affect the performance, please refer to the following documentation:
            https://huggingface.co/docs/peft/v0.13.0/en/package_reference/lora#peft.LoraConfig
            """
            
            # Determine the LoRA Configuration
            self.lora_config = LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1, bias="none",target_modules=target_modules)
            print(self.lora_config)

            peft_model = get_peft_model(model, self.lora_config)
            return peft_model
    
        def forward(self, x,extract_embed=False):
            """
            After applying PEFT, and according to the number of added_layers we apply the forward pass.
            
            Note: Unlike the transfer learning models versions without PEFT, if added_layers = 1 this wouldn't return the same embeddings as the pre-trained version because applying PEFT with LoRA has altered networks weights also.
            """
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

            return x