import json
from transformers import GPT2Tokenizer, GPT2Model
from pathlib import Path
import torch
import os
import torch.nn as nn

from deeptune.options import ROOT, DOWNLOADED_MODELS


model_name = "gpt2"
save_path = ROOT / "downloaded_models" / model_name

GPT2_PATH = DOWNLOADED_MODELS / "gpt2"

GPT2_MODEL = GPT2_PATH / "model"
GPT2_MODEL_FILES = [
    GPT2_MODEL / "config.json",
    GPT2_MODEL / "model.safetensors"
]

GPT2_TOKENIZER = GPT2_PATH / "tokenizer"
GPT2_TOKENIZER_FILES = [
    GPT2_TOKENIZER / "merges.txt",
    GPT2_TOKENIZER / "special_tokens_map.json",
    GPT2_TOKENIZER / "tokenizer_config.json",
    GPT2_TOKENIZER / "vocab.json",
]


def download_gpt2_model():
    # Download the model
    model = GPT2Model.from_pretrained(model_name)
    model.save_pretrained(save_path / "model")
    print(f"Saved model to {save_path / 'model'}")
    
    # Download the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path / "tokenizer")
    print(f"Saved tokenizer to {save_path / 'tokenizer'}")


# def load_gpt2_model_offline():
#     if not os.path.exists(save_path / "model"):
#         print(f"Model folder not found at {save_path / 'model'}. Downloading now...")
#         download_gpt2_model()
    
#     tokenizer = GPT2Tokenizer.from_pretrained(save_path / "tokenizer")
#     model = GPT2Model.from_pretrained(save_path / "model")
    
#     print(f"Loaded model from {save_path / 'model'}")
#     print(f"Loaded tokenizer from {save_path / 'tokenizer'}")
    
#     return model, tokenizer


def download_gpt2():
    model = GPT2Model.from_pretrained(model_name)
    model.save_pretrained(GPT2_MODEL)
    print(f"Saved model to {GPT2_MODEL}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(GPT2_TOKENIZER)
    print(f"Saved tokenizer to {GPT2_TOKENIZER}")


def load_gpt2_model_offline() -> GPT2Model:
    if not GPT2_PATH.exists():
        download_gpt2_model()
    model = GPT2Model.from_pretrained(GPT2_MODEL)
    return model


def load_gpt2_tokenizer_offline() -> GPT2Tokenizer:
    if not GPT2_PATH.exists():
        download_gpt2_model()
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gpt2_offline() -> tuple[GPT2Model, GPT2Tokenizer]:
    model = load_gpt2_model_offline()
    tokenizer = load_gpt2_tokenizer_offline()
    print(f"Loaded model from {GPT2_MODEL}")
    print(f"Loaded tokenizer from {GPT2_TOKENIZER}")
    return model, tokenizer


class AdjustedGPT2Model(nn.Module):
    # def __init__(self, gpt_model, num_classes, freeze_backbone=None, output_dim=1000):
    def __init__(self, num_classes, freeze_backbone=None, output_dim=1000):
        super().__init__()
        # self.gpt2 = gpt_model
        self.gpt2 = load_gpt2_model_offline()
        self.num_classes = num_classes
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
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_embeddings(input_ids, attention_mask)
        return self.classifier(pooled_output)
    
    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, 768)
        x = last_hidden.transpose(1, 2)

        return self.conv_head(x)  # (B, output_dim)


def load_tuned_gpt2(traindir: Path) -> AdjustedGPT2Model:
    cli_path = traindir / "cli_arguments.json"
    cli_args: dict = json.load(open(cli_path, "r"))
    num_classes = cli_args["num_classes"]
    embed_size = cli_args["embed_size"]

    model_weights = traindir / "model_weights.pth"

    model = AdjustedGPT2Model(num_classes=num_classes, freeze_backbone=False, output_dim=embed_size)
    model.load_state_dict(torch.load(model_weights))
    return model
    



    # def get_embeddings():
    #     ...

    # def get_nlp_embeddings(self):
    #     all_embeddings = []
    #     all_labels = []

    #     self.eval()
    #     with torch.no_grad():
    #         for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    #             batch_texts = texts[i:i + BATCH_SIZE]
    #             batch_labels = labels[i:i + BATCH_SIZE]

    #             inputs = tokenizer(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    #             inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    #             # get sentence-level embeddings directly
    #             outputs = self(**inputs)
    #             mapped_embeddings = outputs

    #             all_embeddings.append(mapped_embeddings.cpu())
    #             all_labels.extend(batch_labels)

    #     all_embeddings = torch.cat(all_embeddings, dim=0)
    #     embeddings_df = pd.DataFrame(all_embeddings.numpy())
    #     embeddings_df['label'] = all_labels

    #     return embeddings_df







