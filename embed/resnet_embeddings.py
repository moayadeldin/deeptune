"""This extract_embeddings.py is integrated in order to use fine-tuned ResNet50 model as an embeddings extractor for the training images, to evaluate the performance of different classic ML algorithms who excel in tabular data.
"""
from models.resnet18 import adjustedResNet
from models.resnet18_peft import adjustedPeftResNet
from dataset import ParquetImageDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Extract the Embeddings for your fine-tuned model after entering the Hyperparameters, data and model paths.")

parser.add_argument('--num_classes', type=int, required=True, help='The number of classes in your dataset.')
parser.add_argument('--batch_size', type=int, required=True, help='Batch Size for embeddings.')
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing your dataset.')
parser.add_argument('--finetuned_model_pth', type=str, required=True, help='Directory for your fine-tuned model.')

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
BATCH_SIZE = args.batch_size
NUM_CLASSES = args.num_classes
MODEL_PATH = args.finetuned_model_pth

model = adjustedPeftResNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

def adjustModel(model):

    """The model is modified to be prepared for extracting feature embeddings rather than making predictions.
    """

    
    # take all layers except the last one (one used for classification) to make it output feature embeddings.
    modules = list(model.children())[:-1]

    # unpacking the layers in modules and now it contains the entire model minus the last one.
    model = nn.Sequential(*modules)

    return model

adjusted_model = adjustModel(model=model)

for p in adjusted_model.parameters(): # stop gradient calculations
    p.requires_grad = False

adjusted_model.cuda() # move the model to cuda

dataset = ParquetImageDataset(parquet_file=DATASET_DIR)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

def extractEmbeddings():

    extracted_labels = []
    extracted_embeddings = []

    torch.cuda.empty_cache()
    model.eval()

    for data, labels in tqdm(data_loader):

        new_labels = labels.numpy().tolist()

        extracted_labels += new_labels

        data = data.cuda()

        embeddings = adjusted_model(data.cuda())

        extracted_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))

    extracted_embeddings = np.vstack(extracted_embeddings)

    return extracted_embeddings, extracted_labels

if __name__ == "__main__":

    embeddings, labels = extractEmbeddings()

    print(f"The shape of the embeddings matrix in the dataset is{embeddings.shape}")
    print(f"The number of the labels in the dataset is {len(labels)} labels.")

    embeddings_df = pd.DataFrame(embeddings)
    labels_df = pd.DataFrame(labels, columns=["label"])

    combined_df = pd.concat([embeddings_df,labels_df],axis=1)

    combined_df.to_parquet("test_set_embeddings.parquet",index=False)







    
