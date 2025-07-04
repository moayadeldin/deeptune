from pathlib import Path
from typing import cast
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from peft import PeftModel
from peft import LoraConfig, get_peft_model
import torch
import os


ROOT = Path(__file__).parent.parent.parent
SIGLIP_PATH = ROOT / "downloaded_models/siglip_so400m_patch14_384/"

SIGLIP_PEFT_ADAPTER = SIGLIP_PATH / "peft_adapter"

SIGLIP_MODEL = SIGLIP_PATH / "model"

SIGLIP_MODEL_FILES = [
    SIGLIP_MODEL / "config.json",
    SIGLIP_MODEL / "model.safetensors",
]


SIGLIP_PREPROCESSOR = ROOT / SIGLIP_PATH / "preprocessor"
SIGLIP_PREPROCESSOR_FILES = [
    SIGLIP_PREPROCESSOR / "preprocessor_config.json",
    SIGLIP_PREPROCESSOR / "special_tokens_map.json",
    SIGLIP_PREPROCESSOR / "spiece.model",
    SIGLIP_PREPROCESSOR / "tokenizer_config.json"
]

class CustomSigLIPWithPeft(nn.Module):
    
    def __init__(self, base_model, num_classes, added_layers,embedding_layer, freeze_backbone):
        
        super(CustomSigLIPWithPeft, self).__init__()
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.freeze_backbone = freeze_backbone
        self.embedding_layer = embedding_layer
        self.base_model = base_model
        
        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Configure LoRA
        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "v_proj",
                "q_proj",
            ]
        )

        print(f"Using PEFT with config: {peft_config}")
        
        # Apply PEFT to base model

        self.base_model = get_peft_model(base_model, peft_config)
        # add custion layers
        
        if self.added_layers == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152,self.embedding_layer),
                nn.Linear(self.embedding_layer, self.num_classes)
            )
        
        elif self.added_layers == 1:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, self.num_classes)
            )
        else:
            self.fc_layers = None
        
    def forward(self, input_dict):
        
        outputs = self.base_model.vision_model(pixel_values=input_dict["pixel_values"])
        pooled_output = outputs.pooler_output

        if self.fc_layers:
            logits = self.fc_layers(pooled_output)
            return logits
        return pooled_output

def download_siglip_model():
    
    model = cast(SiglipModel, AutoModel.from_pretrained("google/siglip-so400m-patch14-384"))
    
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")
    
    processor = cast(
        SiglipProcessor, AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    )
    
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")
    
    
def load_peft_siglip_offline(added_layers=None,embedding_layer=None, freeze_backbone=None,num_classes=None):
    
    base_model = AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)
    
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    )
        
    if num_classes is not None:
        model = CustomSigLIPWithPeft(base_model, num_classes,added_layers,embedding_layer,freeze_backbone)
        return model, tokenizer
    
    else: # apply PEFT only to the base model
        
        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "v_proj",
                "q_proj",
            ])
        
        model = get_peft_model(base_model, peft_config)
        model = cast(SiglipModel, model)
    
    return base_model, tokenizer

def load_custom_peft_siglip(
        added_layers,
        embedding_dim,
        num_classes,
):
    
    print(f"Loading custom SigLIP model with {added_layers} added layers, embedding dim {embedding_dim}, and {num_classes} classes.")
    base_model, _ = load_peft_siglip_offline()
    model = CustomSigLIPWithPeft(
        base_model=base_model,
        added_layers=added_layers,
        embedding_layer=embedding_dim,
        num_classes=num_classes,
        freeze_backbone=False
    )

    return model


if __name__ == "__main__":
    download_siglip_model()
