# import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision

from pandas import DataFrame
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import Swin_T_Weights

from datasets.image_datasets import ParquetImageDataset
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from embed.vision.custom_embed_siglip_handler import embed_with_siglip
from helpers import transformations
import time
from cli import DeepTuneVisionOptions
from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP, RunType,save_process_times


def main():
    args = DeepTuneVisionOptions(RunType.EMBED)
    DF_PATH: Path = args.df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_ARCHITECTURE = args.model_architecture
    MODEL_STR = args.model

    MODEL_PATH = args.model_weights
    USE_CASE = args.use_case.value
    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    
    BATCH_SIZE = args.batch_size
    EMBED_OUTPUT = (OUT / f"embed_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/embed_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{MODEL_STR}_{MODE}_embeddings.parquet"

    if MODEL_ARCHITECTURE == "siglip" and MODEL_VERSION == "siglip":
        embed_with_siglip(
            dataset_path=DF_PATH,
            model_weights=MODEL_PATH,
            num_classes=NUM_CLASSES,
            added_layers=ADDED_LAYERS,
            embed_size=EMBED_SIZE,
            use_case=USE_CASE,
            outdir=EMBED_OUTPUT,
            output=EMBED_FILE,
            device=DEVICE,
        )
        args.save_args(EMBED_OUTPUT)
        return

    model = load_vision_model(
        model_architecture=MODEL_ARCHITECTURE,
        model_version=MODEL_VERSION,
        use_case=USE_CASE,
        num_classes=NUM_CLASSES,
        embed_size=EMBED_SIZE,
        added_layers=ADDED_LAYERS,
        freeze_backbone=False,
        mode=MODE,
        model_path=MODEL_PATH,
    )
    
    start_time = time.time()

    adjusted_model = adjust_vision_model(model, MODEL_ARCHITECTURE, USE_CASE, ADDED_LAYERS)

    embedding_model = EmbeddingModel(adjusted_model, MODEL_ARCHITECTURE, MODEL_VERSION, ADDED_LAYERS, EMBED_SIZE, MODE, DEVICE)

    df = pd.read_parquet(DF_PATH)

    combined_df = embedding_model(df, BATCH_SIZE)

    combined_df.to_parquet(EMBED_FILE, index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT,process="embedding")

    args.save_args(EMBED_OUTPUT)


def load_vision_model(
    model_architecture: str,
    model_version: str,
    use_case: str,
    num_classes: int,
    embed_size: int,
    added_layers: int = None,
    freeze_backbone: bool = True,
    mode: str = 'cls',
    model_path = None,
) -> nn.Module:
    
    if model_architecture == "densenet":

        if use_case not in ('peft', 'finetuned'):
            raise ValueError('There is no third option other than ["finetuned", "peft"] for DenseNet.')
       
        if added_layers == 1 or added_layers == 0:  # This is because DenseNet embeddings extraction using DeepTune only supports it via intermediate layer.
            raise ValueError("Kindly note that extracting embeddings from DenseNet is only supported through the intermediate embedding layer (added_layers=2). Extracting embeddings from pretrained model directly could lead potentially to misleading results as the last layer in the densenet architecture before the classification layer is a BatchNorm, which are normalization features that is used to stabilize training, and doesn't provide meaningful images presentation.")

    if use_case == 'peft':
        adjusted_model_cls = PEFT_MODEL_CLS_MAP.get(model_architecture)
        model = adjusted_model_cls(num_classes, model_version, added_layers, embed_size, freeze_backbone, task_type=mode)

    elif use_case == 'finetuned':
        adjusted_model_cls = MODEL_CLS_MAP.get(model_architecture)
        model = adjusted_model_cls(num_classes, model_version, added_layers, embed_size, freeze_backbone, task_type=mode)

    elif use_case == 'pretrained':

        if model_architecture == "efficientnet":

            if model_version == "efficientnet_b0":
                    model = torchvision.models.efficientnet_b0(weights="DEFAULT")
            elif model_version == "efficientnet_b1":
                    model = torchvision.models.efficientnet_b1(weights="DEFAULT")
            elif model_version == "efficientnet_b2":
                    model = torchvision.models.efficientnet_b2(weights="DEFAULT")
            elif model_version == "efficientnet_b3":
                    model = torchvision.models.efficientnet_b3(weights="DEFAULT")
            elif model_version == "efficientnet_b4":
                    model = torchvision.models.efficientnet_b4(weights="DEFAULT")
            elif model_version == "efficientnet_b5":
                    model = torchvision.models.efficientnet_b5(weights="DEFAULT")
            elif model_version == "efficientnet_b6":
                    model = torchvision.models.efficientnet_b6(weights="DEFAULT")
            elif model_version == "efficientnet_b7":
                    model = torchvision.models.efficientnet_b7(weights="DEFAULT")
            else:
                raise ValueError('The pretrained model should be one of the following: ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]')
            model.classifier = nn.Identity()  # Remove classification layer to use as feature extractor

        elif model_architecture == "resnet":

            if model_version == 'resnet18':    
                model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif model_version == 'resnet34':
                model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            elif model_version == 'resnet50':
                model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            elif model_version == 'resnet101':
                model = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            elif model_version == 'resnet152':
                model = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
            else:
                raise ValueError('The pretrained model should be one of the following: ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]')
           
            model.fc = nn.Identity()  # Remove classification layer to use as feature extractor

        elif model_architecture == "swin":

            if model_version == 'swin_t':
                model = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            elif model_version == 'swin_s':
                model = torchvision.models.swin_s(weights=Swin_T_Weights.IMAGENET1K_V1)
            elif model_version == 'swin_b':
                model = torchvision.models.swin_b(weights=Swin_T_Weights.IMAGENET1K_V1)
       
            model.head = nn.Identity()  # Remove classification layer to use as feature extractor
       
        elif model_architecture == "vgg":

            if model_version == 'vgg11':
                model = torchvision.models.vgg11(weights="DEFAULT")
            elif model_version == 'vgg13':
                model = torchvision.models.vgg13(weights="DEFAULT")
            elif model_version == 'vgg16':
                model = torchvision.models.vgg16(weights="DEFAULT")
            elif model_version == 'vgg19':
                model = torchvision.models.vgg19(weights="DEFAULT")
           
            model.classifier[6] = nn.Identity()  # Remove classification layer to use as feature extractor

        elif model_architecture == "vit":

            if model_version == 'vit_b_16':
                model = torchvision.models.vit_b_16(weights="DEFAULT")
            elif model_version == 'vit_b_32':
                model = torchvision.models.vit_b_32(weights="DEFAULT")
            elif model_version == 'vit_l_16':
                model = torchvision.models.vit_l_16(weights="DEFAULT")
            elif model_version == 'vit_l_32':
                model = torchvision.models.vit_l_32(weights="DEFAULT")
            elif model_version == 'vit_h_14':
                model = torchvision.models.vit_h_14(weights="DEFAULT")
           
            model.heads = nn.Identity()  # Remove classification layer to use as feature extractor

    else:
        raise ValueError('There is no fourth option other than ["finetuned", "peft", "pretrained"]')

    if (
        model_architecture == "desnsenet"
        or (use_case != 'pretrained' and added_layers == 2) # If the use case is peft or pretrained, and the added layers is 2, this means that we want to extract the weights of the embedding layer, otherwise the weights of the original model.
    ):
        model.load_state_dict(torch.load(model_path, weights_only=True))
   
    return model


def adjust_vision_model(model: nn.Module, model_architecture: str, use_case: str, added_layers: int):
    """
    The model is modified to be prepared for extracting feature embeddings rather than making predictions,
    according to the user specifications with added_layers number and the USE_CASE.
    """

    if (
         (model_architecture == "vit" and use_case == 'pretrained') or
         added_layers == 1 or
         (model_architecture in ("efficientnet", "resnet", "swin", "vgg", "vit") and added_layers == 2)
    ):
        model.eval()
        return model

    if model_architecture == "densenet" and use_case == 'peft':
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
        model = nn.Sequential(*modules, nn.Flatten(1))

        model.eval()
   
    return model


class EmbeddingModel:
    """
    Class to extract image embeddings using trained deep learner.

    **NOTE**: Only image data is currently supported in this version.
    """
    def __init__(self, model: nn.Module, model_name: str, model_version: str, added_layers: int = 2, embed_size: int = 1000, task_type: str = 'cls', device: torch.device = torch.device("cpu")):
        self.model = model
        self.model_name = model_name
        self.version = model_version
        self.added_layers = added_layers
        self.embed_size = embed_size
        self.task_type = task_type

        self.device = device

        self.data_column = "images"
        self.target_column = "labels"
    
    def __call__(self, df: DataFrame, batch_size: int = 2) -> DataFrame:
        dataset = ParquetImageDataset(df, transform=transformations)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEM,
            persistent_workers=PERSIST_WORK,
        )
        return self._extract_embeddings(data_loader)

    def _extract_embeddings(self, data_loader: DataLoader) -> DataFrame:
        
        self.model.to(self.device)

        embeddings, labels, extracted_other_cols = self._extract_embeddings_labels(data_loader)

        print(f"The shape of the embeddings matrix in the dataset is {embeddings.shape}")

        embeddings_df = pd.DataFrame(embeddings)

        if labels is None:
            return embeddings_df

        labels_df = pd.DataFrame(labels, columns=["label"])
        other_cols_df = pd.DataFrame(extracted_other_cols) if extracted_other_cols else None

        if other_cols_df is not None:
             combined_df = pd.concat([embeddings_df, labels_df, other_cols_df], axis=1)

        else:
             combined_df = pd.concat([embeddings_df, labels_df], axis=1)
             
        return combined_df

    def _extract_embeddings_labels(self, data_loader: DataLoader) -> tuple[np.ndarray, list]:
        """
        This function takes the dataloader input, extract the embeddings with the corresponding labels and return them at the end.


        Args:
            data_loader (DataLoader): ...
    
        Returns:
            extracted_embeddings (numpy NDArray): ...
            extracted_labels (numpy NDArray): ...
        """
        extracted_labels = []
        extracted_embeddings = []
        extracted_other_cols = []

        self.model.eval()
        self.model = self.model.to(DEVICE)

        with torch.no_grad():
            for batch in tqdm(data_loader):
                data, labels, *extras = batch
                data = data.to(DEVICE)
            
                # If the added layers is one and we want to extract the same exact embedding features as if the added layers is zero we should handle this explicitly
                if self.added_layers == 1 or (self.added_layers == 2 and self.model_name != "densenet"):
                    embeddings = self.model(data, extract_embed=True)
                else:
                    embeddings = self.model(data)

                extracted_embeddings.append(embeddings.detach().cpu().numpy())

                if labels is not None:
                    extracted_labels += labels.numpy().tolist()
                if extras:
                    # extras is a list of dicts, take the first dict element
                    dict_extras = extras[0]  
                    batch_size = len(data)
                    for idx in range(batch_size):
                        sample_extras = {k: v[idx].item() if hasattr(v, 'item') else v[idx]
                                    for k, v in dict_extras.items()}
                        extracted_other_cols.append(sample_extras)

        extracted_embeddings = np.vstack(extracted_embeddings)
        labels_array = extracted_labels if len(extracted_labels) > 0 else None
        return extracted_embeddings, labels_array, extracted_other_cols


if __name__ == "__main__":

    main()
