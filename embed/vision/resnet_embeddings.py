from src.vision.resnet18 import adjustedResNet
from src.vision.resnet18_peft import adjustedPeftResNet
from datasets.image_datasets import ParquetImageDataset
from utilities import transformations
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
import torchvision
from torchvision.models import ResNet18_Weights

"""
Note: If you chose to have the finetuned option with added layers = 1, it will have the same embeddings output as if pretrained. This is normal behavior and works as expected. Because in the pretrained option the last layer are actually mapping 512 inputs to 1000 outputs. For the finetuned option, it maps the same 512 inputs but to 8 outputs. The difference is in the output but input to the last layer is actually the same.
"""

parser = argparse.ArgumentParser(description="Extract the Embeddings for your fine-tuned model after entering the Hyperparameters, data and model paths.")

parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--use_case', type=str, choices=['peft', 'finetuned', 'pretrained'], required=True,help='The mode you want to set embeddings extractor with') 
parser.add_argument('--added_layers', type=int, choices=[1,2], help='The number of layers you already added while adjusting the model.')
parser.add_argument('--embed_size', type=int, help='The size of embedding layer you already added while adjusting the model.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size for embeddings.')
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing your dataset.')
parser.add_argument('--finetuned_model_pth', type=str, required=False, help='Directory for your fine-tuned model.')
parser.add_argument('--freeze-backbone', action='store_true', help='Decide whether you want to freeze backbone or not.')

args = parser.parse_args()


USE_CASE = args.use_case
DATASET_DIR = args.dataset_dir
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL_PATH = args.finetuned_model_pth
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FREEZE_BACKBONE = args.freeze_backbone

if USE_CASE == 'peft':
    
    model = adjustedPeftResNet(NUM_CLASSES,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
    TEST_OUTPUT = "test_set_peft_resnet18_embeddings.parquet"
    args.use_case = 'PEFT-ResNet18'
    
elif USE_CASE == 'finetuned':
    model = adjustedResNet(NUM_CLASSES,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE)
    TEST_OUTPUT = "test_set_finetuned_resnet18_embeddings.parquet"
    args.use_case = 'finetuned-ResNet18'

elif USE_CASE == 'pretrained':
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # Remove classification layer to use as feature extractor
    TEST_OUTPUT = "test_set_pretrained_resnet18_embeddings.parquet"
    args.use_case = 'pretrained-ResNet18'
else:
    raise ValueError('There is no fourth option other than ["finetuned", "peft", "pretrained"]')


# If the use case is peft or pretrained, and the added layers is 2, this means that we want to extract the weights of the embedding layer, otherwise the weights of the original model.
if not USE_CASE == 'pretrained':
    if ADDED_LAYERS == 2:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

def adjustModel(model):

    """The model is modified to be prepared for extracting feature embeddings rather than making predictions.
    """

    if ADDED_LAYERS == 1 or ADDED_LAYERS ==2:
        
        model.eval()
        return model
    
    # take all layers except the last one (one used for classification) to make it output feature embeddings.
    modules = list(model.children())[:-1]

    # unpacking the layers in modules and now it contains the entire model minus the last one.
    model = nn.Sequential(*modules)
    
    model.eval()

    return model

adjusted_model = adjustModel(model=model)

for p in adjusted_model.parameters(): # stop gradient calculations
    p.requires_grad = False

adjusted_model.cuda() # move the model to cuda

dataset = ParquetImageDataset(parquet_file=DATASET_DIR, transform=transformations)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

def extractEmbeddings():

    extracted_labels = []
    extracted_embeddings = []

    torch.cuda.empty_cache()
    adjusted_model.eval()

    for data, labels in tqdm(data_loader):

        new_labels = labels.numpy().tolist()

        extracted_labels += new_labels

        data = data.cuda()
        
        # If the added layers is one and we want to extract the same exact embedding features as if the added layers is zero we should handle this explicitly
        
        if ADDED_LAYERS == 1 or ADDED_LAYERS == 2:
            embeddings = adjusted_model(data, extract_embed=True)
        else:
            embeddings = adjusted_model(data)

        extracted_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))

    extracted_embeddings = np.vstack(extracted_embeddings)

    return extracted_embeddings, extracted_labels

if __name__ == "__main__":

    embeddings, labels = extractEmbeddings()

    print(f"The shape of the embeddings matrix in the dataset is{embeddings.shape}")

    embeddings_df = pd.DataFrame(embeddings)
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet(TEST_OUTPUT,index=False)







    
