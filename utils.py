import json
import numpy as np
import re
import warnings

import torch.nn as nn

from argparse import Namespace, ArgumentError
from enum import Enum
from pathlib import Path
from typing import Type

from src.vision.densenet import adjustedDenseNet
from src.vision.densenet_peft import adjustedPeftDenseNet
from src.vision.efficientnet import adjustedEfficientNet
from src.vision.efficientnet_peft import adjustedPeftEfficientNet
from src.vision.resnet import adjustedResNet
from src.vision.resnet_peft import adjustedPeftResNet
from src.vision.swin import adjustedSwin
from src.vision.swin_peft import adjustedPeftSwin
from src.vision.vgg import adjustedVGGNet
from src.vision.vgg_peft import adjustedPeftVGGNet
from src.vision.vit import adjustedViT
from src.vision.vit_peft import adjustedViTPeft

from utilities import fixed_seed


MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    "densenet": adjustedDenseNet,
    "efficientnet": adjustedEfficientNet,
    "resnet": adjustedResNet,
    "swin": adjustedSwin,
    "vgg": adjustedVGGNet,
    "vit": adjustedViT,
}

PEFT_MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    "densenet": adjustedPeftDenseNet,
    "efficientnet": adjustedPeftEfficientNet,
    "resnet": adjustedPeftResNet,
    "swin": adjustedPeftSwin,
    "vgg": adjustedPeftVGGNet,
    "vit": adjustedViTPeft,
}


class UseCase(Enum):
    PRETRAINED = "pretrained"
    FINETUNED = "finetuned"
    PEFT = "peft"

    @classmethod
    def from_string(cls, s: str) -> "UseCase":
        return cls(s)
    
    @classmethod
    def choices(cls) -> list[str]:
        return [x.value for x in cls]
    
    @classmethod
    def parse(cls, s: str) -> str:
        try:
            return cls(s).value
        except Exception:
            return cls(s.lower()).value


class RunType(Enum):
    TRAIN = "training"
    EVAL = "evalulation"
    EMBED = "embedding"


def get_model_cls(model_architecture: str, use_peft: bool = False) -> Type[nn.Module]:
    if use_peft:
        return PEFT_MODEL_CLS_MAP.get(model_architecture)
    else:
        return MODEL_CLS_MAP.get(model_architecture)


def get_model_version(args: Namespace) -> str:
    all_model_versions = [
        args.resnet_version,
        args.densenet_version,
        args.swin_version,
        args.efficientnet_version,
        args.vgg_net_version,
        args.vit_version,
        args.siglip_version,
    ]
    selected_model_version = list(filter(None, all_model_versions))

    if num_selected:=len(selected_model_version) != 1:
        raise ArgumentError(f"Must specify exactly one model, but {num_selected} were requested.")
   
    return selected_model_version[0]


def get_model_architecture(model_version: str):
    match = re.match(r"^[a-zA-Z]+", model_version)
    if match:
        return match.group()
    raise ValueError(f"Could not identify architecture from: {model_version}")


def save_cli_args(args: Namespace, outdir: Path) -> None:
    """
    Save the CLI arguments to a JSON file.

    Parameters
    ----------
    args: Namespace
        Parsed CLI arguments.
    
    outdir: Path
        Directory to save the `cli_arguments.json` file.
    """
    cli_dict = args.__dict__.copy()
    for k, v in cli_dict.items():
        if isinstance(v, Path):
            cli_dict[k] = str(v.resolve())
    
    out_path = outdir / "cli_arguments.json"
    with open(out_path, "w") as f:
        json.dump(cli_dict, f, indent=4)


def set_seed(use_fixed_seed: bool) -> None:
    if use_fixed_seed:
        SEED = 42
        fixed_seed(SEED)
    else:
        SEED = np.random.randint(low=0, high=1_000)
        fixed_seed(SEED)
        warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits.', category=UserWarning)
        warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)