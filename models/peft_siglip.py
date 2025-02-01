"""
This code is mostly adopted from John's repo to download siglip.
https://github.com/johnkxl/peft4vision/
"""

from pathlib import Path
from typing import cast
from transformers import AutoModel, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from peft import PeftModel


ROOT = Path(__file__).parent.parent
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
    
    
def load_siglip_offline(peft=False):
    
    model = AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True)
    
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    )
    
    if peft:
        
        model = PeftModel.from_pretrained(model, SIGLIP_PEFT_ADAPTER)
        
    model = cast(SiglipModel, model)
    
    return model,tokenizer


def load_siglip_for_image_classification_offline(peft=False):
    """
    Returns SigLIP model with classification head and image processor. 
    Specifying `peft=True` loads the model with the PEFT LoRA adapter.

    Parameters
    ----------
    label2id: dict
        Dictionary mapping target class labels to integers in dataset.
    id2label: dict
        Dictionary mapping intergers in dataset to target class lables.
    peft: bool, default=False
        Load the model saved with the PEFT adapter if `True`.
    
    Returns
    -------
    tuple[AutoModelForImageClassification | PeftModel, AutoImageProcessor]

    """
    model_path = SIGLIP_PEFT_TRAINED if peft else SIGLIP_MODEL

    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    
    processor = AutoImageProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True)
    
    return model, processor


if __name__ == "__main__":
    download_siglip_model()
