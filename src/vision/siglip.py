import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, cast
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from peft import get_peft_model, PeftModel, PeftConfig


ROOT = Path(__file__).parent.parent.parent
SIGLIP_PATH = ROOT / "downloaded_models/siglip_so400m_patch14_384/"

SIGLIP_MODEL = SIGLIP_PATH / "model"
SIGLIP_PREPROCESSOR = ROOT / SIGLIP_PATH / "preprocessor"

SIGLIP_MODEL_FILES = [
    SIGLIP_MODEL / "config.json",
    SIGLIP_MODEL / "model.safetensors",
]

SIGLIP_PREPROCESSOR_FILES = [
    SIGLIP_PREPROCESSOR / "preprocessor_config.json",
    SIGLIP_PREPROCESSOR / "special_tokens_map.json",
    SIGLIP_PREPROCESSOR / "spiece.model",
    SIGLIP_PREPROCESSOR / "tokenizer_config.json"
]


class CustomSiglipModel(nn.Module):
    def __init__(
        self,
        base_model: SiglipModel,
        added_layers,
        embedding_dim,
        num_classes,
        freeze_backbone
    ):
        super().__init__()
        self.added_layers = added_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.base_model = base_model
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if added_layers == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.num_classes)
            )
        elif added_layers == 1:
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
    
    
def load_siglip_offline():
    
    model = AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)
    
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    )
        
    model = cast(SiglipModel, model)
    
    return model,tokenizer

def load_custom_siglip_model(
    added_layers,
    embedding_dim,
    num_classes,
    freeze_backbone
):
    
    print(f"Loading custom SigLIP model with {added_layers} added layers, embedding dim {embedding_dim}, and {num_classes} classes.")
    base_model, _ = load_siglip_offline()
    model = CustomSiglipModel(
        base_model=base_model,
        added_layers=added_layers,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )
    return model


if __name__ == "__main__":
    download_siglip_model()
