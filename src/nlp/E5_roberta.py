from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from typing import (
    cast,
)
import os
from pathlib import Path
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

# Routes of the BERT's model and tokenizer to load.
ROOT = Path(__file__).parent.parent

INTFLOAT_MULTILINGUAL_TOKENIZER = (
    ROOT / "downloaded_models/intfloat_multi_large/tokenizer"
)


INTFLOAT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/intfloat_multi_large/model"

def download_nlp_intfloat_ml_model() -> None:
    # Download the model
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained("intfloat/multilingual-e5-large"),
    )
    model.save_pretrained(INTFLOAT_MULTILINGUAL_MODEL)
    print(f"Saved model to {INTFLOAT_MULTILINGUAL_MODEL}")
    
    # Download the tokenizer
    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large"),
    )
    tokenizer.save_pretrained(INTFLOAT_MULTILINGUAL_TOKENIZER)
    print(f"Saved tokenizer to {INTFLOAT_MULTILINGUAL_MODEL}")
    
def load_nlp_intfloat_ml_model_offline() -> (
    tuple[XLMRobertaModel, XLMRobertaTokenizerFast]
):
    # If model not downloaded then download it
    if not os.path.exists(INTFLOAT_MULTILINGUAL_MODEL):
        print(f"Model folder not found at {INTFLOAT_MULTILINGUAL_MODEL}. Downloading now...")
        download_nlp_intfloat_ml_model()
    
    # Then you Load model and tokenizer
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained(INTFLOAT_MULTILINGUAL_MODEL, local_files_only=True),
    )
    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained(
            INTFLOAT_MULTILINGUAL_TOKENIZER, local_files_only=True
        ),
    )
    return model, tokenizer

