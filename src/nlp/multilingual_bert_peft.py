from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import cast
from pathlib import Path
import os
import torch
import torch.nn as nn
from typing import Tuple
from transformers import BertModel, BertTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

# Routes of the BERT's model and tokenizer to load.
ROOT = Path(__file__).parent.parent

BERT_MULTILINGUAL_TOKENIZER = ROOT / "downloaded_models/bert_multilingual_uncased/tokenizer"
BERT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/bert_multilingual_uncased/model"


class CustomMultilingualPeftBERT(nn.Module):
    
    """
    Customised Multi lingual BERT class according to DeepTune proposed Adjustments.
    
    Attributes:
        num_classes (int): Number of Classes of your dataset.
        added_layers (int): Number of additional layers you want to add on the top of the original model.
        embedding_layer (int): Represents the size of the intermediate layer, useful only if added_layers = 2.
        freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
        
    """
    
    def __init__(self,num_classes,added_layers,embedding_layer,freeze_backbone=None):
        
        super(CustomMultilingualPeftBERT,self).__init__()
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.freeze_backbone = freeze_backbone
        self.embedding_layer = embedding_layer
        
        base_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        # Check if freeze_backbone true freeze the original model's weights otherwise update all weights.
        
        if freeze_backbone:
            
            for param in self.bert.parameters():
                param.requires_grad = False
            print('Backbone Parameters are freezed!')
        
        
        self.bert = self.applyPEFT(base_model)
        
        
        output_dim = self.bert.config.hidden_size
        
        # Add the additional layers according to prompt.
        
        if added_layers == 1:
            
            self.classifier = nn.Linear(output_dim,num_classes)
            
        elif added_layers == 2:
            
            self.additional = nn.Linear(output_dim,embedding_layer)
            self.classifier = nn.Linear(embedding_layer, num_classes)
            
    def applyPEFT(self, model):
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
        return get_peft_model(model, peft_config)

    

    def forward(self, input_ids, attention_mask, token_type_ids=None,extract_embed=False):
        
        """
        Applying the forward pass to the Custom MultiLingual BERT PEFT model.
        
        Args:
            input_ids (torch.Tensor): Tensor of input token IDs with shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.
            token_type_ids (torch.Tensor): Segment token IDs for distinguishing segments in input (e.g., for sentence pairs).
            extract_embed (bool): Whether we want to apply the adjustments the user do when we extract embeddings.
            
        """
        
        # Get BERT output
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        
        if self.added_layers == 1:
            # Directly feed the input to the final added layer.
            x = self.classifier(pooled_output)
        
        elif self.added_layers == 1 and extract_embed:
            
            return pooled_output
        
        if self.added_layers == 2:
            # Directly feed the input to intermediate layer and get the output from intermediate to the final added layer.
            additional = self.additional(pooled_output)
            x = self.classifier(additional)
        
        elif self.added_layers == 2 and extract_embed:
            
            x = self.additional(pooled_output)
                        
        return x