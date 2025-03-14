from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import cast
from pathlib import Path
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

ROOT = Path(__file__).parent.parent

BERT_MULTILINGUAL_TOKENIZER = ROOT / "downloaded_models/bert_multilingual_uncased/tokenizer"
BERT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/bert_multilingual_uncased/model"


class CustomMultilingualBERT(nn.Module):
    
    def __init__(self,num_classes,added_layers,embedding_layer,freeze_backbone):
        
        super(CustomMultilingualBERT,self).__init__()
        
        self.num_classes = num_classes
        self.added_layers = added_layers
        self.freeze_backbone = freeze_backbone
        self.embedding_layer = embedding_layer
        
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        if freeze_backbone:
            
            for param in self.bert.parameters():
                param.requires_grad = False
                
        output_dim = self.bert.config.hidden_size
        
        if added_layers == 1:
            
            self.classifier = nn.Linear(output_dim,num_classes)
            
        elif added_layers == 2:
            
            self.additional = nn.Linear(output_dim,embedding_layer)
            self.classifier = nn.Linear(embedding_layer, num_classes)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        
        # get BERT output
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        
        if self.added_layers == 1:
            logits = self.classifier(pooled_output)
        elif self.added_layers == 2:
            additional = self.additional(pooled_output)
            logits = self.classifier(additional)
            
                        
        return logits


def download_nlp_bert_ml_model() -> None:
    """Download and save the BERT base multilingual uncased sentiment model locally."""
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model.save_pretrained(BERT_MULTILINGUAL_MODEL)
    print(f"Saved model to {BERT_MULTILINGUAL_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer.save_pretrained(BERT_MULTILINGUAL_TOKENIZER)
    print(f"Saved tokenizer to {BERT_MULTILINGUAL_TOKENIZER}")

def load_nlp_bert_ml_model_offline() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the BERT multilingual uncased sentiment model from local storage."""
    if not os.path.exists(BERT_MULTILINGUAL_MODEL):
        raise FileNotFoundError(f"Model folder not found at {BERT_MULTILINGUAL_MODEL}. Please run download_nlp_bert_ml_model() first.")

    if not os.path.exists(BERT_MULTILINGUAL_TOKENIZER):
        raise FileNotFoundError(f"Tokenizer folder not found at {BERT_MULTILINGUAL_TOKENIZER}. Please run download_nlp_bert_ml_model() first.")

    print(f"Loading model from: {BERT_MULTILINGUAL_MODEL}")
    print(f"Loading tokenizer from: {BERT_MULTILINGUAL_TOKENIZER}")

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MULTILINGUAL_MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MULTILINGUAL_TOKENIZER, local_files_only=True)
    return model, tokenizer

# # def predict_sentiment(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> str:
# #     """Predict sentiment for a given text using the loaded model."""
# #     inputs = tokenizer(text, return_tensors="pt")
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #     sentiment_score = torch.argmax(outputs.logits).item()

# #     # Mapping sentiment scores to labels (1-5 stars for this model)
# #     sentiment_map = {
# #         0: "Very Negative",
# #         1: "Negative",
# #         2: "Neutral",
# #         3: "Positive",
# #         4: "Very Positive"
# #     }
# #     return sentiment_map.get(sentiment_score, "Unknown Sentiment")
