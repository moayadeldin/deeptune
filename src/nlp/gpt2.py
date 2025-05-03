from transformers import GPT2Tokenizer, GPT2Model
from pathlib import Path
import torch
import os


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



