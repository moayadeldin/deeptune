import torch
from src.nlp.multilingual_bert import CustomMultilingualBERT
from src.nlp.multilingual_bert_peft import CustomMultilingualPeftBERT

def test_custom_mbert_model_initialization():
    model = CustomMultilingualBERT(
        num_classes=10,
        added_layers=2,
        embedding_layer=512,
        freeze_backbone=False,
        pretrained=False
    )

    # Dummy tokenized batch: batch_size=4, seq_len=16
    input_ids = torch.randint(0, model.bert.config.vocab_size, (4, 16))
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids, attention_mask)

    assert out.shape == (4, 10), "Output shape should match number of classes"

def test_custom_peft_mbert_model_initialization():
    model = CustomMultilingualPeftBERT(
        num_classes=10,
        added_layers=2,
        embedding_layer=512,
        freeze_backbone=False
    )

    # Dummy tokenized batch: batch_size = 4, seq_len = 16
    input_ids = torch.randint(0, model.bert.config.vocab_size, (4, 16))
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids, attention_mask)

    assert out.shape == (4, 10), "Output shape should match number of classes"