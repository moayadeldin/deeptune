import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import cast
from pathlib import Path
import os
import torch
import torch.nn as nn
from typing import Tuple
from transformers import BertModel, BertTokenizer
from peft import LoraConfig, get_peft_model

from deeptune.options import DOWNLOADED_MODELS


BERT_MULTILINGUAL_TOKENIZER = DOWNLOADED_MODELS / "bert_multilingual_uncased/tokenizer"
BERT_MULTILINGUAL_MODEL = DOWNLOADED_MODELS / "bert_multilingual_uncased/model"


class CustomMultilingualBERT(nn.Module):
    """
    Customised Multi lingual BERT class according to DeepTune proposed Adjustments.
    
    Attributes:
        num_classes (int): Number of Classes of your dataset.
        added_layers (int): Number of additional layers you want to add on the top of the original model.
        embedding_layer (int): Represents the size of the intermediate layer, useful only if added_layers = 2.
        freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
        
    """
    
    def __init__(self, num_classes, added_layers, embedding_layer, freeze_backbone=None):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.freeze_backbone = freeze_backbone
        self.embedding_layer = embedding_layer
        
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert = load_bert_model_offline()
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('Backbone Parameters are freezed!')
                
        output_dim = self.bert.config.hidden_size
        
        if added_layers == 1:
            self.classifier = nn.Linear(output_dim,num_classes)
        elif added_layers == 2:
            self.additional = nn.Linear(output_dim, embedding_layer)
            self.classifier = nn.Linear(embedding_layer, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Applying the forward pass to the Custom MultiLingual BERT PEFT model.
        
        Args:
            input_ids (torch.Tensor): Tensor of input token IDs with shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.
            token_type_ids (torch.Tensor): Segment token IDs for distinguishing segments in input (e.g., for sentence pairs).
            extract_embed (bool): Whether we want to apply the adjustments the user do when we extract embeddings.
            
        """
        # # Get BERT output
        # outputs = self.bert(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask,
        #     token_type_ids = token_type_ids
        # )
        
        # pooled_output = outputs.pooler_output
        
        # if self.added_layers == 1:
        #     # Directly feed the input to the final added layer.
        #     logits = self.classifier(pooled_output)
        # elif self.added_layers == 2:
        #     # Directly feed the input to intermediate layer and get the output from intermediate to the final added layer.
        #     additional = self.additional(pooled_output)
        #     logits = self.classifier(additional)
        pooled_output = self.get_embeddings(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)
      
        return logits
    
    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        pooled_output = outputs.pooler_output

        if self.added_layers == 2:
            pooled_output = self.additional(pooled_output)
        
        return pooled_output


class CustomMultilingualPeftBERT(CustomMultilingualBERT):
    def __init__(self, num_classes, added_layers, embedding_layer, freeze_backbone=True):
        super().__init__(num_classes, added_layers, embedding_layer, freeze_backbone)

        peft_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "query",
                "key",
                "value",
                "dense",
            ]
        )
        print("[INFO] Applying PEFT with config:", peft_config)
        self.bert = get_peft_model(self.bert, peft_config)


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


def load_bert_model_offline() -> BertModel:
    if not BERT_MULTILINGUAL_MODEL.exists():
        download_nlp_bert_ml_model()
    return BertModel.from_pretrained(BERT_MULTILINGUAL_MODEL, local_files_only=True)


def load_bert_tokenizer_offline() -> BertTokenizer:
    if not BERT_MULTILINGUAL_TOKENIZER.exists():
        download_nlp_bert_ml_model()
    return BertTokenizer.from_pretrained(BERT_MULTILINGUAL_TOKENIZER, local_files_only=True)


def load_tuned_bert(traindir: Path) -> CustomMultilingualBERT:
    cli_path = traindir / "cli_arguments.json"
    cli_args: dict = json.load(open(cli_path, "r"))
    num_classes = cli_args["num_classes"]
    added_layers = cli_args["added_layers"]
    embed_size = cli_args["embed_size"]
    use_peft = cli_args["use_peft"]

    model_weights = traindir / "model_weights.pth"

    model_cls = CustomMultilingualPeftBERT if use_peft else CustomMultilingualBERT
    model = model_cls(num_classes, added_layers, embed_size, freeze_backbone=False)
    model.load_state_dict(torch.load(model_weights))
    return model