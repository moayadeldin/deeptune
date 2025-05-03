import torch 
from src.nlp.gpt2 import download_gpt2_model, load_gpt2_model_offline
from argparse import ArgumentParser
import options
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

parser = ArgumentParser(description="Extract Embeddings using GPT2 model")
OUTPUT = f"deeptune_results/test_set_pretrained_GPT2_embeddings.parquet"
parser = options.parser
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = args.batch_size
INPUT_DF_PATH = args.input_dir

# load the model, the tokenizer and the dataset.
model,tokenizer = load_gpt2_model_offline()
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_parquet(INPUT_DF_PATH)

texts = df['text'].tolist()
labels = df['label'].tolist()

class Mapping3DOutputTo2D(nn.Module):
    
    def __init__(self, input_dim=256 * 768, output_dim=1000):
        super(Mapping3DOutputTo2D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (B, 256, 768) -> (B, 196608)
        return self.model(x)
    
mapping_model = Mapping3DOutputTo2D().to(DEVICE)

def run_embeddings():
    
    device = torch.device(DEVICE)
    model.to(device)
    
    df = get_nlp_embeddings(model)
    
    df.to_parquet(OUTPUT)
    print(f"Saved text embeddings to {OUTPUT}")

def get_nlp_embeddings(model):
    
    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_labels = labels[i:i+BATCH_SIZE]

            # get inputs including attention mask
            inputs = tokenizer(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            
            # print(outputs.last_hidden_state.shape)
            
            # Mask padding tokens
            
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states * expanded_mask # remove the paddings
            
            mapped_embeddings = mapping_model(hidden_states)
            # print(mapped_embeddings.shape)
            all_embeddings.append(mapped_embeddings.cpu())
            all_labels.extend(batch_labels)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    embeddings_df = pd.DataFrame(all_embeddings.numpy())
    embeddings_df['label'] = all_labels  # add label column
    
    return embeddings_df

# def getting_mean_embeds_without_padding_tokens(inputs, outputs):
    
#     """
#     This function is mainly to mask out the padding tokens in the sequence so that the embeddings don't contain the noise of padding tokens before averaging, making them more accurate
#     """
    
#     attention_mask = inputs['attention_mask']
    
#     last_hidden_state = outputs.last_hidden_state # (B,seq_len,hidden_size)
    
#     # expand attention mask for matching dimensions
#     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()) # (B, seq_len)
    
#     # zero out the embeddings where attention mask is zero.
    
#     last_hidden_state = last_hidden_state * attention_mask_expanded
    
#     # sum along the sequence dimension to get the numerator for the mean
    
#     summed = last_hidden_state.sum(dim=1)
    
#     # count how many non-pad tokens per sentence
#     counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
    
#     # get the mean of embeddings by dividing summed vectors by token counts
    
#     mean_embeddings = summed / counts
    
#     return mean_embeddings

if __name__ == "__main__":
    
    run_embeddings()




    


