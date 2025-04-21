from src.vision.vit import adjustedViT
from src.vision.vit_peft import adjustedViTPeft
from datasets.image_datasets import ParquetImageDataset
from utilities import transformations,get_args
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import options
import pandas as pd
import torchvision

# Initialize the needed variables either from the CLI user sents or from the device.

DEVICE = options.DEVICE
parser = options.parser
args = get_args()
VIT_VERSION = args.vit_version
USE_CASE = args.use_case
INPUT_DIR = args.input_dir
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL_PATH = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FREEZE_BACKBONE = args.freeze_backbone
MODE = args.mode

# Check which USE_CASE is used and based on this choose the model to get loaded. For example, if finetuned was the USE_CASE then the class call would be from the transfer-learning without PEFT version.
if USE_CASE == 'peft':
    
    model = adjustedViTPeft(NUM_CLASSES,VIT_VERSION, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = f"test_set_peft_vit_embeddings_{MODE}.parquet"
    args.use_case = 'peft- ' + VIT_VERSION
    pass
elif USE_CASE == 'finetuned':
    model = adjustedViT(NUM_CLASSES, VIT_VERSION, ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = f"test_set_finetuned_vit_embeddings_{MODE}.parquet"
    args.use_case = 'finetuned- ' + VIT_VERSION

elif USE_CASE == 'pretrained':
    if VIT_VERSION == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights="DEFAULT")
    elif VIT_VERSION == 'vit_b_32':
        model = torchvision.models.vit_b_32(weights="DEFAULT")
    elif VIT_VERSION == 'vit_l_16':
        model = torchvision.models.vit_l_16(weights="DEFAULT")
    elif VIT_VERSION == 'vit_l_32':
        model = torchvision.models.vit_l_32(weights="DEFAULT")
    elif VIT_VERSION == 'vit_h_14':
        model = torchvision.models.vit_h_14(weights="DEFAULT")
    
    model.heads = nn.Identity()  # Remove classification layer to use as feature extractor
    TEST_OUTPUT = f"test_set_pretrained_vit_embeddings_{MODE}.parquet"
    args.use_case = 'pretrained- ' + VIT_VERSION

else:
    raise ValueError('There is no fourth option other than ["finetuned", "peft", "pretrained"]')

# If the use case is peft or pretrained, and the added layers is 2, this means that we want to extract the weights of the embedding layer, otherwise the weights of the original model.
if not USE_CASE == 'pretrained':
    if ADDED_LAYERS == 2:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

def adjustModel(model):

    """The model is modified to be prepared for extracting feature embeddings rather than making predictions.
    """

    if USE_CASE == 'pretrained':
        model.eval()
        return model
    
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

# load the dataloader
dataset = ParquetImageDataset(parquet_file=INPUT_DIR, transform=transformations)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

def extractEmbeddings():
        
    """
    This function takes the dataloader input, extract the embeddings with the corresponding labels and return them at the end.
    """

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

    # run the function
    embeddings, labels = extractEmbeddings()

    print(f"The shape of the embeddings matrix in the dataset is{embeddings.shape}")
    
    # convert the embeddings to pd.DataFrame and merge the column of labels then return it
    
    embeddings_df = pd.DataFrame(embeddings)
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet(TEST_OUTPUT,index=False)