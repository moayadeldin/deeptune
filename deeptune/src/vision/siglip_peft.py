# deprecated
import torch
import torch.nn as nn

from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import SiglipModel

from deeptune.options import DOWNLOADED_MODELS


SIGLIP_PATH = DOWNLOADED_MODELS / "siglip_so400m_patch14_384/"

SIGLIP_PEFT_ADAPTER = SIGLIP_PATH / "peft_adapter"

SIGLIP_MODEL = SIGLIP_PATH / "model"
SIGLIP_MODEL_FILES = [
    SIGLIP_MODEL / "config.json",
    SIGLIP_MODEL / "model.safetensors",
]

SIGLIP_PREPROCESSOR = SIGLIP_PATH / "preprocessor"
SIGLIP_PREPROCESSOR_FILES = [
    SIGLIP_PREPROCESSOR / "preprocessor_config.json",
    SIGLIP_PREPROCESSOR / "special_tokens_map.json",
    SIGLIP_PREPROCESSOR / "spiece.model",
    SIGLIP_PREPROCESSOR / "tokenizer_config.json"
]


class CustomSigLIPWithPeft(nn.Module):
    def __init__(
        self,
        base_model: SiglipModel,
        num_classes: int,
        added_layers: int,
        embedding_layer: int,
        freeze_backbone: bool,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer = embedding_layer
        self.freeze_backbone = freeze_backbone
        
        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        if added_layers == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, embedding_layer),
                nn.Linear(embedding_layer, num_classes)
            )
        elif added_layers == 1:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, num_classes)
            )
        else:
            self.fc_layers = nn.Sequential(nn.Identity())

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
        
    def _get_pooled_output(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.base_model.vision_model(pixel_values=input_dict["pixel_values"])
        return outputs.pooler_output

    def forward(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled_output = self._get_pooled_output(input_dict)
        logits = self.fc_layers(pooled_output)
        return logits
    
    def get_image_embeddings(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled_output = self._get_pooled_output(input_dict)
        image_embeddings = self.fc_layers[0](pooled_output)
        return image_embeddings

