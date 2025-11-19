from src.nlp.gpt2 import AdjustedGPT2Model
from transformers import GPT2Model, GPT2Config
import torch
def test_adjusted_gpt2_model_initialization():

    config = GPT2Config(n_layer=2, n_head=2, n_embd=768)
    gpt_model = GPT2Model(config)

    model = AdjustedGPT2Model(
        gpt_model=gpt_model,
        freeze_backbone=False,
        output_dim=1000,
        pretrained=False
    )

    # Dummy tokenized input: batch_size=4, seq_len=16
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids, attention_mask)

    assert out.shape == (4, 1000), "Output shape should match output_dim"