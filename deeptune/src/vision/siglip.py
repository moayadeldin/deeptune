import torch
import torch.nn as nn

from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import SiglipModel, SiglipProcessor
from typing import Type

from deeptune.utils import UseCase
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


class CustomSiglipModel(nn.Module):
    def __init__(
        self,
        base_model: SiglipModel,
        num_classes: int,
        added_layers: int,
        embedding_dim: int,
        freeze_backbone: bool,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if added_layers == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, embedding_dim),
                nn.Linear(embedding_dim, num_classes)
            )
        elif added_layers == 1:
            self.fc_layers = nn.Sequential(
                nn.Linear(1152, num_classes)
            )
        else:
            self.fc_layers = nn.Sequential(nn.Identity())
        
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


class CustomSigLIPWithPeft(CustomSiglipModel):
    def __init__(
        self,
        base_model: SiglipModel,
        num_classes: int,
        added_layers: int,
        embedding_dim: int,
        freeze_backbone: bool,
    ):
        super().__init__(base_model, num_classes, added_layers, embedding_dim, freeze_backbone)

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
        self.base_model = get_peft_model(base_model, peft_config)


def download_siglip() -> None:
    model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")
    
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")

    
def load_siglip_offline() -> tuple[SiglipModel, SiglipProcessor]:
    model = load_siglip_model_offline()
    processor = load_siglip_processor_offline()
    return model, processor


def load_siglip_model_offline() -> SiglipModel:
    model = SiglipModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)
    return model


def load_siglip_processor_offline() -> SiglipProcessor:
    processor = SiglipProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True, use_fast=True)
    return processor


def get_siglip_cls(use_case: UseCase) -> Type[CustomSiglipModel]:
    cls_map = {
        UseCase.PRETRAINED: CustomSiglipModel,
        UseCase.FINETUNED: CustomSiglipModel,
        UseCase.PEFT: CustomSigLIPWithPeft,
    }
    try:
        return cls_map[use_case]
    except KeyError:
        raise ValueError(f"Unsupported use case for custom SigLIP loading: {use_case}")


def load_siglip_variant(
    use_case: UseCase = UseCase.PRETRAINED,
    num_classes: int = None,
    added_layers: int = None,
    embed_size: int = None,
    freeze_backbone: bool = False,
    model_weights: Path = None,
    device: torch.device = torch.device("cpu"),
) -> CustomSiglipModel:

    print(
        f"Loading custom SigLIP model with {added_layers} added layers, "
        f"embedding dim {embed_size}, and {num_classes} classes "
        f"(use_case={use_case.value}, freeze_backbone={freeze_backbone})."
    )

    base_model = load_siglip_model_offline()
    model_cls = get_siglip_cls(use_case)
    
    model = model_cls(
        base_model=base_model,
        num_classes=num_classes,
        added_layers=added_layers,
        embedding_dim=embed_size,
        freeze_backbone=(False if use_case in (UseCase.PEFT, UseCase.PRETRAINED) else freeze_backbone),
    )
    
    if model_weights is not None:
        print(f"Loading model weights from {model_weights} onto device {device}.")
        model.load_state_dict(torch.load(model_weights, map_location=device))
    
    return model


if __name__ == "__main__":
    download_siglip()
