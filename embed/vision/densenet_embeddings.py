from src.vision.densenet import adjustedDenseNet
from src.vision.densenet121_peft import adjustedPEFTDenseNet
from datasets.image_datasets import ParquetImageDataset
from utilities import transformations
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import options
import pandas as pd

"""
Please Note that that extracting embeddings from DenseNet is only supported through the intermediate embedding layer (ADDED_LAYERS=2).
"""

# Initialize the needed variables either from the CLI user sents or from the device.

DEVICE = options.DEVICE
parser = options.parser
args = parser.parse_args()
DENSENET_VERSION = args.densenet_version
USE_CASE = args.use_case
DATASET_DIR = args.input_dir
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL_PATH = args.model_weights
ADDED_LAYERS = args.added_layers
EMBED_SIZE = args.embed_size
FREEZE_BACKBONE = args.freeze_backbone
MODE = args.mode
    
# Check which USE_CASE is used and based on this choose the model to get loaded. For example, if finetuned was the USE_CASE then the class call would be from the transfer-learning without PEFT version.
if USE_CASE == 'finetuned':
    model = adjustedDenseNet(NUM_CLASSES,DENSENET_VERSION,ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = f"deeptune_results/test_set_finetuned_DenseNet121_embeddings_{MODE}.parquet"
    args.use_case = 'finetuned-Densenet121'

elif USE_CASE == 'peft':
    model = adjustedPEFTDenseNet(NUM_CLASSES, DENSENET_VERSION, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = f"deeptune_results/test_set_peft_DenseNet121_embeddings_{MODE}.parquet"
else:
    raise ValueError('There is no third option other than ["finetuned", "peft"]')

# This is because DenseNet embeddings extraction using DeepTune only supports it via intermediate layer.
if ADDED_LAYERS == 1 or ADDED_LAYERS ==0:
    
    raise ValueError("Kindly note that extracting embeddings from DenseNet is only supported through the intermediate embedding layer (ADDED_LAYERS=2). Extracting embeddings from pretrained model directly could lead potentially to misleading results as the last layer in the densenet architecture before the classification layer is a BatchNorm, which are normalization features that is used to stabilize training, and doesn't provide meaningful images presentation.")

# load the weights
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

def adjustModel(model):

    """The model is modified to be prepared for extracting feature embeddings rather than making predictions, according to the user specifications with added_layers number and the USE_CASE.
    """

    if ADDED_LAYERS == 1:
        
        model.eval()
        return model
    
    if USE_CASE == 'peft':
        modules = [
            model.peftmodel,  # DenseNet features
            model.flatten,  # Flatten
            model.fc1   # First Linear layer (embedding)
        ]
        model = nn.Sequential(*modules)
    else:
        # take all layers except the last one (one used for classification) to make it output feature embeddings.
        modules = list(model.children())[:-1]

        # unpacking the layers in modules and now it contains the entire model minus the last one.
        model = nn.Sequential(*modules)
        
    model.eval()
    
    return model

adjusted_model = adjustModel(model=model)

# stop gradient calculations
for p in adjusted_model.parameters(): 
    p.requires_grad = False

# move the model to GPU
adjusted_model.cuda() 

# Load the dataloader

dataset = ParquetImageDataset(parquet_file=DATASET_DIR, transform=transformations)

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
        
        if ADDED_LAYERS == 1:
            embeddings = adjusted_model(data, extract_embed=True)
        else:
            embeddings = adjusted_model(data)
            
        # append the extracted embeddings of this batch to the list of whole embeddings extractions.    

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







    
