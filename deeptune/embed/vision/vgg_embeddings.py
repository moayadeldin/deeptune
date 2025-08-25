from deeptune.src.vision.vgg_peft import adjustedPeftVGGNet
from deeptune.src.vision.vgg import adjustedVGGNet
from deeptune.datasets.image_datasets import ParquetImageDataset
from deeptune.utilities import transformations,get_args
from deeptune.options import DEEPTUNE_RESULTS
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision

# Initialize the needed variables either from the CLI user sents or from the device.
args = get_args()
VGGNET_VERSION = args.vgg_net_version
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
    
    model = adjustedPeftVGGNet(NUM_CLASSES,VGGNET_VERSION, ADDED_LAYERS, EMBED_SIZE, FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = DEEPTUNE_RESULTS / f"test_set_peft_vgg_embeddings_{MODE}.parquet"
    args.use_case = 'peft- ' + VGGNET_VERSION
elif USE_CASE == 'finetuned':
    model = adjustedVGGNet(NUM_CLASSES, VGGNET_VERSION, ADDED_LAYERS, EMBED_SIZE,FREEZE_BACKBONE,task_type=MODE)
    TEST_OUTPUT = DEEPTUNE_RESULTS / f"test_set_finetuned_vgg_embeddings_{MODE}.parquet"
    args.use_case = 'finetuned- ' + VGGNET_VERSION

elif USE_CASE == 'pretrained':
    if VGGNET_VERSION == 'vgg11':
        model = torchvision.models.vgg11(weights="DEFAULT")
    elif VGGNET_VERSION == 'vgg13':
        model = torchvision.models.vgg13(weights="DEFAULT")
    elif VGGNET_VERSION == 'vgg16':
        model = torchvision.models.vgg16(weights="DEFAULT")
    elif VGGNET_VERSION == 'vgg19':
        model = torchvision.models.vgg19(weights="DEFAULT")
    
    model.classifier[6] = nn.Identity()  # Remove classification layer to use as feature extractor
    TEST_OUTPUT = DEEPTUNE_RESULTS / f"test_set_pretrained_vgg_embeddings_{MODE}.parquet"
    args.use_case = 'pretrained- ' + VGGNET_VERSION

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

# load the dataloader
dataset = ParquetImageDataset.from_parquet(parquet_file=INPUT_DIR, transform=transformations)

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