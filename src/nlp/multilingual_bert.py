from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import cast
from pathlib import Path
import os
import torch
import torch.nn as nn
from typing import Tuple
from transformers import BertModel, BertTokenizer


# Routes of the BERT's model and tokenizer to load.
ROOT = Path(__file__).parent.parent

BERT_MULTILINGUAL_TOKENIZER = ROOT / "downloaded_models/bert_multilingual_uncased/tokenizer"
BERT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/bert_multilingual_uncased/model"


class CustomMultilingualBERT(nn.Module):
    
    """
    Customised Multi lingual BERT class according to DeepTune proposed Adjustments.
    
    Attributes:
        num_classes (int): Number of Classes of your dataset.
        added_layers (int): Number of additional layers you want to add on the top of the original model.
        embedding_layer (int): Represents the size of the intermediate layer, useful only if added_layers = 2.
        freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
        
    """
    
    def __init__(self,num_classes,added_layers,embedding_layer,freeze_backbone=None):
        
        super(CustomMultilingualBERT,self).__init__()
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.freeze_backbone = freeze_backbone
        self.embedding_layer = embedding_layer
        
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        
        if freeze_backbone:
            
            for param in self.bert.parameters():
                param.requires_grad = False
                
        output_dim = self.bert.config.hidden_size
        
        # Add the additional layers according to prompt.
        
        if added_layers == 1:
            
            self.classifier = nn.Linear(output_dim,num_classes)
            
        elif added_layers == 2:
            
            self.additional = nn.Linear(output_dim,embedding_layer)
            self.classifier = nn.Linear(embedding_layer, num_classes)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        
        # Get BERT output
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        
        if self.added_layers == 1:
            # Directly feed the input to the final added layer.
            logits = self.classifier(pooled_output)
        elif self.added_layers == 2:
            # Directly feed the input to intermediate layer and get the output from intermediate to the final added layer.
            additional = self.additional(pooled_output)
            logits = self.classifier(additional)
            
                        
        return logits


def download_nlp_bert_ml_model() -> None:
    """Download and save the BERT base multilingual uncased sentiment model locally."""
    # Download the model
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model.save_pretrained(BERT_MULTILINGUAL_MODEL)
    print(f"Saved model to {BERT_MULTILINGUAL_MODEL}")

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer.save_pretrained(BERT_MULTILINGUAL_TOKENIZER)
    print(f"Saved tokenizer to {BERT_MULTILINGUAL_TOKENIZER}")

def load_nlp_bert_ml_model_offline() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the BERT multilingual uncased sentiment model from local storage."""
    # If model not downloaded then download it
    if not os.path.exists(BERT_MULTILINGUAL_MODEL):
        print(f"Model folder not found at {BERT_MULTILINGUAL_MODEL}. Downloading now...")
        download_nlp_bert_ml_model()
        
    # If model not found then raise an error
    if not os.path.exists(BERT_MULTILINGUAL_TOKENIZER):
        raise FileNotFoundError(f"Tokenizer folder not found at {BERT_MULTILINGUAL_TOKENIZER}. Please run download_nlp_bert_ml_model() first.")
    
    # If model found then load it with the tokenizer
    print(f"Loading model from: {BERT_MULTILINGUAL_MODEL}")
    print(f"Loading tokenizer from: {BERT_MULTILINGUAL_TOKENIZER}")

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MULTILINGUAL_MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MULTILINGUAL_TOKENIZER, local_files_only=True)
    return model, tokenizer
