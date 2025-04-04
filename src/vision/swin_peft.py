import torchvision
from torchvision.models import Swin_T_Weights
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class adjustedPeftSwin(nn.Module):

    def __init__(self,num_classes,swin_version, added_layers, lora_attention_dimension, freeze_backbone=False,task_type='cls',output_dim=1):

        """
        
        Customised Swin class applying Parameter Efficient Fine Tuning with LoRA as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            swin_version (str): Version of Swin you want to use.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            lora_attention_dimension (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            output_dim (int): The dimension of the output of regression model, default = 1.
            
            
        
        """
        
        super(adjustedPeftSwin, self).__init__()

        self.swin_version = swin_version
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
        self.freeze_backbone = freeze_backbone

        if swin_version == "swin_t":
            weights = Swin_T_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_t(weights=weights)
            self.model = pretrained_swin
        elif swin_version == "swin_s":
            weights = Swin_T_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_s(weights=weights)
            self.model = pretrained_swin
        elif swin_version == "swin_b":
            weights = Swin_T_Weights.IMAGENET1K_V1
            pretrained_swin = torchvision.models.swin_b(weights=weights)
            self.model = pretrained_swin
        else:
            raise ValueError("Invalid swin_version. Choose from 'swin_t', 'swin_s', or 'swin_b'.")
        
        # Get the input size of the last layer before we chop it
        self.in_features = self.model.head.in_features
        
        # additional parameters for regression
        self.task_type = task_type
        self.output_dim = output_dim

        # remove the final connected layer by putting a placeholder
        self.model.head = nn.Identity()
        self.flatten = nn.Flatten()
        
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        if self.freeze_backbone:
            print("Backbone Parameters are frozen!")
            for param in self.model.parameters():
                param.requires_grad = False
            
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    
        # Add the additional layers according to prompt.
        
        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.in_features,self.lora_attention_dimension)
            
            if self.task_type == "cls": 
               self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.lora_attention_dimension,self.output_dim)
        elif self.added_layers == 1:
            
            if self.task_type == "cls":
                self.fc1 = nn.Linear(self.in_features, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.in_features, self.output_dim)
        else:
            self.fc1 = None
            
        # Apply PEFT
            
        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self,model):
        
        """
        
        Apply PEFT with LoRA on the customised model.
        
        In this implementation, the developer chose mainly to apply LoRA matricies to only three types of linear layers: proj, qkj, and head.
        
        We could have chosen applying it for all linear layers. For example, those present in the MLP in Swin's architecture.
        
        We may change according to the observations to determine which is better.
        
        Arguments:
            model (torchvision.models): The adjusted Swin model before applying PEFT On
            
        Returns:
        
            peft_model: PEFTed adjusted Swin
        
        """
        # target_modules is actually the parts of the network we have applied PEFT on
        target_modules = []
        
        for name, module in model.named_modules():
            # Filter only the 'qkv', 'proj', and 'head' Linear layers
            if any(layer in name for layer in ["qkv", "proj", "head"]) and isinstance(module, nn.Linear):
                target_modules.append(name)

        print("Applying LoRA to layers:", target_modules)

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

