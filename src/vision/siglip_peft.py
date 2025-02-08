from pathlib import Path
from typing import cast
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from peft import PeftModel
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).parent.parent.parent
SIGLIP_PATH = ROOT / "downloaded_models/siglip_so400m_patch14_384/"

SIGLIP_PEFT_ADAPTER = SIGLIP_PATH / "peft_adapter"

SIGLIP_MODEL = SIGLIP_PATH / "model"
SIGLIP_PEFT_TRAINED = SIGLIP_PATH / "full_peft_trained"

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

def download_siglip_model():
    
    model = cast(SiglipModel, AutoModel.from_pretrained("google/siglip-so400m-patch14-384"))
    
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")
    
    processor = cast(
        SiglipProcessor, AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    )
    
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")
    
    
def load_peft_siglip_offline():
    
    base_model = AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)
    
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    )
        
    """Here I do something different to what John did in his implementation, John would actually call PeftModel.from_pretrained here in load_siglip function, then when referring to this function again in the trainer, he would use get_peft_model(). We could think of this as an overhead.
    
    When I searched about this, I found that the PeftModel.from_pretrained is calling an existing LoRA weights that was previously trained. On the other hand, get_peft_model creates and attaches new PEFT/LoRA weights to a base model, for new training.
    
    What we want to do is the second thing, a new training, so I only call get_peft_model here.
    
    """
    
    for param in base_model.parameters():
         param.requires_grad = False

    peft_config = LoraConfig(
            inference_mode=False,  # Enable training
            r=16,                  # Low-rank dimension
            lora_alpha=32,         # Scaling factor
            lora_dropout=0.1,      # Dropout
            target_modules=[
                # "k_proj",
                "v_proj",
                "q_proj",
                # "out_proj",
            ]
        )

    # Wrap the base model with the PEFT model
    peft_model = get_peft_model(base_model, peft_config)
        
    model = cast(SiglipModel, peft_model)
    
    return model,tokenizer


def load_peft_siglip_for_image_classification_offline():
    """
    Returns PEFT SigLIP model with classification head and image processor.
    """
    
    model_path = SIGLIP_PEFT_TRAINED

    model = SiglipModel.from_pretrained(
        model_path,
        local_files_only=True
    )
    
        
    processor = AutoImageProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    
    return model, processor


if __name__ == "__main__":
    download_siglip_model()
