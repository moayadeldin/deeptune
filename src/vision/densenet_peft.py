import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

class adjustedPEFTDenseNet(nn.Module):
    
    
    def __init__(self, num_classes,densenet_version, added_layers, lora_attention_dimension, freeze_backbone=False, model_to_load='densenet121',task_type='cls',output_dim=1):
        
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
        
        super(adjustedPEFTDenseNet, self).__init__()
        
        self.model_to_load = model_to_load
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
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
            for param in self.model.parameters():
                param.requires_grad = False
            print('Backbone Parameters are frozen!')
                
        # Add the additional layers according to prompt.
        if self.added_layers == 2:
            self.fc1 = nn.Linear(in_features,self.lora_attention_dimension)
            
            if self.task_type == "cls": 
               self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.lora_attention_dimension,self.output_dim)
        elif self.added_layers == 1:
            
            if self.task_type == "cls":
                self.fc1 = nn.Linear(in_features, self.num_classes)
            else:
                self.fc1 = nn.Linear(in_features, self.output_dim)
        else:
            self.fc1 = None
        
        self.peftmodel = self.applyPEFT(self.model)
                
    
    def applyPEFT(self,model):
        
        """
        Apply PEFT with LoRA on the customised model.
        
        Arguments:
            model (torchvision.models): The adjusted DenseNet model before applying PEFT On
            
        Returns:
        
            peft_model: PEFTed adjusted DenseNet
        """
        
        # target_modules is actually the parts of the network we have applied PEFT on
        target_modules = []
        # available_types are the networks that support PEFT optimization
        available_types = [
            nn.modules.conv.Conv2d,
            nn.modules.linear.Linear]
        
        
        # loop through the model and check what layers in it we may apply PEFT on and them to target_modules
        for n,m in model.named_modules():
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
    
    def forward(self, x, extract_embed=False):
        
        """
        After applying PEFT, and according to the number of added_layers we apply the forward pass.
        
        Note: Unlike the transfer learning models versions without PEFT, if added_layers = 1 this wouldn't return the same embeddings as the pre-trained version because applying PEFT with LoRA has altered networks weights also.
        """
        
        x = self.peftmodel(x)
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

