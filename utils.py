import json
import numpy as np
import re
import warnings

import torch.nn as nn

from argparse import Namespace, ArgumentError
from enum import Enum
from pathlib import Path
from typing import Type
import os
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
from src.vision.convnext import adjustedConvNext
from src.vision.convnext_peft import adjustedPeftConvNext
from src.video.video3d import adjustedVideo3D, VIDEO3D_ARCHITECTURES
from src.video.video3d_peft import adjustedPeftVideo3D
from src.video.frame_pool import VideoFramePoolClassifier

from helpers import fixed_seed


MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    "densenet": adjustedDenseNet,
    "efficientnet": adjustedEfficientNet,
    "resnet": adjustedResNet,
    "swin": adjustedSwin,
    "vgg": adjustedVGGNet,
    "vit": adjustedViT,
    "convnext": adjustedConvNext,
}

PEFT_MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    "densenet": adjustedPeftDenseNet,
    "efficientnet": adjustedPeftEfficientNet,
    "resnet": adjustedPeftResNet,
    "swin": adjustedPeftSwin,
    "vgg": adjustedPeftVGGNet,
    "vit": adjustedViTPeft,
    "convnext": adjustedPeftConvNext,
}

# Native spatio-temporal video architectures (3D-CNNs and video transformers).
# Any model_version NOT in this set is instead treated as a frame-sampling
# video model built on one of the 2D backbones above (e.g. "resnet50").
VIDEO3D_MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    version: adjustedVideo3D for version in VIDEO3D_ARCHITECTURES
}

VIDEO3D_PEFT_MODEL_CLS_MAP: dict[str, Type[nn.Module]] = {
    version: adjustedPeftVideo3D for version in VIDEO3D_ARCHITECTURES
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
    GANDALF = "gandalf"
    TIMESERIES = 'timeseries'
    ONECALL = 'onecall'
    TabPFNTRAIN = 'tabpfntrain'
    TabPFNEVAL = 'tabpfneval'
    TabPFNEMBD = 'tabpfnembed'


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
    # Native video architectures (e.g. "r3d_18") are self-describing and don't
    # follow the "<family><depth>" naming convention (e.g. "resnet50") the
    # regex below expects, so they're matched directly against the known set
    # rather than falling through to it.
    if model_version in VIDEO3D_ARCHITECTURES or model_version == "dummy_video3d":
        return model_version
    match = re.match(r"^[a-zA-Z]+", model_version)
    if match:
        return match.group()
    raise ValueError(f"Could not identify architecture from: {model_version}")


def is_native_video_architecture(model_version: str) -> bool:
    """
    True if `model_version` refers to one of DeepTune's native spatio-temporal
    video architectures (e.g. "r3d_18"), as opposed to a frame-sampling video
    model built on one of the 2D image backbones under src/vision/ (e.g.
    "resnet50").
    """
    return model_version in VIDEO3D_ARCHITECTURES or model_version == "dummy_video3d"


def get_video_model_cls(model_version: str, use_peft: bool = False) -> Type[nn.Module]:
    """
    Resolves `model_version` to the model class DeepTune should build for the
    video modality: one of the native adjustedVideo3D/adjustedPeftVideo3D
    classes if `model_version` names a native video architecture, otherwise
    the same 2D adjusted*Net class the images modality would use, so it can be
    wrapped by VideoFramePoolClassifier.
    """
    if is_native_video_architecture(model_version):
        video_map = VIDEO3D_PEFT_MODEL_CLS_MAP if use_peft else VIDEO3D_MODEL_CLS_MAP
        return video_map.get(model_version, adjustedPeftVideo3D if use_peft else adjustedVideo3D)

    return get_model_cls(get_model_architecture(model_version), use_peft=use_peft)


def build_video_model(
    model_version: str,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    freeze_backbone: bool,
    mode: str,
    use_peft: bool = False,
    pooling: str = "mean",
) -> nn.Module:
    """
    Builds DeepTune's video classifier for `model_version` - either a native
    spatio-temporal model (see VIDEO3D_ARCHITECTURES) built and returned
    directly, or a frame-sampling model wrapping one of the existing 2D vision
    backbones via VideoFramePoolClassifier. Both accept clips shaped
    [B, T, C, H, W] and share the same forward(x, extract_embed=False) API, so
    the rest of DeepTune's Trainer/TestTrainer/embedding pipeline treats them
    identically regardless of which family was chosen.
    """
    model_cls = get_video_model_cls(model_version, use_peft=use_peft)

    if is_native_video_architecture(model_version):
        return model_cls(num_classes, model_version, added_layers, embed_size, freeze_backbone, task_type=mode)

    return VideoFramePoolClassifier(
        frame_model_cls=model_cls,
        frame_model_version=model_version,
        num_classes=num_classes,
        added_layers=added_layers,
        embedding_layer_size=embed_size,
        freeze_backbone=freeze_backbone,
        task_type=mode,
        pooling=pooling,
    )

### DEPRECATED ###
# def save_cli_args(args: Namespace, outdir: Path) -> None:
#     """
#     Save the CLI arguments to a JSON file.

#     Parameters
#     ----------
#     args: Namespace
#         Parsed CLI arguments.
    
#     outdir: Path
#         Directory to save the `cli_arguments.json` file.
#     """
#     cli_dict = args.__dict__.copy()
#     for k, v in cli_dict.items():
#         if isinstance(v, Path):
#             cli_dict[k] = str(v.resolve())
    
#     out_path = os.path.join(outdir, "cli_arguments.json")
#     with open(out_path, "w") as f:
#         json.dump(cli_dict, f, indent=4)


def set_seed(use_fixed_seed: bool) -> None:
    if use_fixed_seed:
        SEED = 42
        fixed_seed(SEED)
    else:
        SEED = np.random.randint(low=0, high=1_000)
        fixed_seed(SEED)
        warnings.warn('This will set a random seed for different initialization affecting Deeptune, inclduing weights and datasets splits. You are safe to neglect this warning if you are using Deeptune for purposes other than training or generating data splits', category=UserWarning)
        warnings.warn("This is liable to increase variability across consecutive runs of DeepTune.", category=UserWarning)
        
def save_process_times(epoch_times, total_duration, outdir,process):
    """
    Save all epoch durations and total duration to a JSON file.

    Parameters:
    - epoch_times: list of dicts, each with 'epoch' and 'duration_seconds'
    - total_duration: float, total training time in seconds
    - outdir: Path object or string representing output directory
    """
    results = {
        "epochs": epoch_times,
        "total_duration_seconds": total_duration
    }
    out_path = Path(outdir) / f"{process}_details.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)