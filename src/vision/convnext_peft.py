import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class adjustedPeftConvNext(nn.Module):

    def __init__(self, num_classes, convnext_version, added_layers, lora_attention_dimension, freeze_backbone=False, task_type="cls", output_dim=1):
        """
        Customised ConvNeXt class applying Parameter Efficient Fine Tuning with LoRA as part of DeepTune proposed Adjustments.
        
        Args:
            num_classes (int) : Number of classes in your dataset.
            convnext_version (str): Version of ConvNeXt you want to use.
            added_layers (int) : Number of additional layers you want to add while finetuning your model
            lora_attention_dimension (int): If you chose added_layers to be 2, so this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want to classification or regression.
            output_dim (int): The dimension of the output of regression model, default = 1.
        """

        super(adjustedPeftConvNext, self).__init__()

        # Task must be regression or classification nothing else
        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
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
        
        self.fc1_input = self.model.classifier[2].in_features
        
        # Additional parameters for regression
        self.task_type = task_type
        self.output_dim = output_dim

        # Remove the final connected layer by putting a placeholder
        self.model.classifier[2] = nn.Identity()
        self.flatten = nn.Flatten()
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Backbone Parameters are frozen!')
                
        # Add the additional layers according to prompt.
        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.fc1_input, self.lora_attention_dimension)
            
            if self.task_type == "cls": 
               self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.lora_attention_dimension, self.output_dim)
                
        elif self.added_layers == 1:
            if self.task_type == "cls":
                self.fc1 = nn.Linear(self.fc1_input, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.fc1_input, self.output_dim)
        else:
            self.fc1 = None
            
        # Apply PEFT
        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self, model):
        """
        Apply PEFT with LoRA on the customised model.
        
        Arguments:
            model (torchvision.models): The adjusted ConvNeXt model before applying PEFT On
            
        Returns:
            peft_model: PEFTed adjusted ConvNeXt
        """
        # For ConvNeXt, we use "all-linear" instead of appending based on available_types.
        # This safely catches the pointwise MLPs while avoiding LayerNorm incompatibility.
        target_modules = "all-linear"
        print('Target Modules:', target_modules)

        # Determine the LoRA Configuration
        self.lora_config = LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=target_modules)
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