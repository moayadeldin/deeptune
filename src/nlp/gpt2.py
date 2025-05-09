from transformers import GPT2Tokenizer, GPT2Model
from pathlib import Path
import torch
import os
import torch.nn as nn




ROOT = Path(__file__).parent.parent

# print(ROOT)

model_name = "gpt2"
save_path = ROOT / "downloaded_models" / model_name

def download_gpt2_model():
    
    # Download the model
    model = GPT2Model.from_pretrained(model_name)
    model.save_pretrained(save_path / "model")
    print(f"Saved model to {save_path / 'model'}")
    
    # Download the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path / "tokenizer")
    print(f"Saved tokenizer to {save_path / 'tokenizer'}")
    
def load_gpt2_model_offline():
    
    if not os.path.exists(save_path / "model"):
        print(f"Model folder not found at {save_path / 'model'}. Downloading now...")
        download_gpt2_model()
    
    tokenizer = GPT2Tokenizer.from_pretrained(save_path / "tokenizer")
    model = GPT2Model.from_pretrained(save_path / "model")
    
    print(f"Loaded model from {save_path / 'model'}")
    print(f"Loaded tokenizer from {save_path / 'tokenizer'}")
    
    return model, tokenizer

class AdjustedGPT2Model(nn.Module):
    def __init__(self, gpt_model, freeze_backbone=None, output_dim=1000):
        super(AdjustedGPT2Model, self).__init__()
        self.gpt2 = gpt_model
        self.output_dim = output_dim

        if freeze_backbone:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print('Backbone Parameters are freezed!')

        self.conv_head = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),            
            nn.Linear(512, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, 768)
        x = last_hidden.transpose(1, 2)

        return self.conv_head(x)  # (B, output_dim)







