import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from torchvision.models.video import (
    R3D_18_Weights,
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    MViT_V2_S_Weights,
    Swin3D_T_Weights,
    Swin3D_S_Weights,
    Swin3D_B_Weights,
)

# Native spatio-temporal architectures DeepTune supports for video: 3D-CNNs
# (r3d_18, mc3_18, r2plus1d_18) and video transformers (mvit_v2_s, swin3d_*),
# all pretrained on Kinetics-400. Unlike the frame-pooling models in
# src/video/frame_pool.py, these learn motion directly via 3D convolutions or
# spatio-temporal attention instead of pooling independent per-frame predictions.
VIDEO3D_ARCHITECTURES = (
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
    "mvit_v2_s",
    "swin3d_t",
    "swin3d_s",
    "swin3d_b",
)

_KINETICS400_WEIGHTS = {
    "r3d_18": R3D_18_Weights.KINETICS400_V1,
    "mc3_18": MC3_18_Weights.KINETICS400_V1,
    "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1,
    "mvit_v2_s": MViT_V2_S_Weights.KINETICS400_V1,
    "swin3d_t": Swin3D_T_Weights.KINETICS400_V1,
    "swin3d_s": Swin3D_S_Weights.KINETICS400_V1,
    "swin3d_b": Swin3D_B_Weights.KINETICS400_V1,
}


def _build_backbone(video_version: str):
    """
    Builds the pretrained torchvision video backbone for `video_version` and
    strips its classification head, returning the backbone alongside the
    number of input features the head used to take.
    """

    if video_version == "dummy_video3d":  # This is for testing purposes only.
        model = video_models.r3d_18(weights=None)
        fc_in = model.fc.in_features
        model.fc = nn.Identity()
        return model, fc_in

    if video_version in ("r3d_18", "mc3_18", "r2plus1d_18"):
        builder = getattr(video_models, video_version)
        model = builder(weights=_KINETICS400_WEIGHTS[video_version])
        fc_in = model.fc.in_features
        model.fc = nn.Identity()
        return model, fc_in

    if video_version == "mvit_v2_s":
        model = video_models.mvit_v2_s(weights=_KINETICS400_WEIGHTS[video_version])
        fc_in = model.head[1].in_features
        model.head = nn.Identity()
        return model, fc_in

    if video_version in ("swin3d_t", "swin3d_s", "swin3d_b"):
        builder = getattr(video_models, video_version)
        model = builder(weights=_KINETICS400_WEIGHTS[video_version])
        fc_in = model.head.in_features
        model.head = nn.Identity()
        return model, fc_in

    raise ValueError(
        f"Invalid video_version. Choose from {VIDEO3D_ARCHITECTURES}."
    )


class adjustedVideo3D(nn.Module):

    def __init__(self, num_classes, video_version, added_layers=2, embedding_layer_size=1000, freeze_backbone=False, task_type="cls", output_dim=1):
        """
        Customised native video classifier as part of DeepTune's proposed
        adjustments, mirroring adjustedResNet's added_layers / embedding_layer_size
        / freeze_backbone conventions, but backed by torchvision's spatio-temporal
        video architectures (3D-CNNs and video transformers), which learn motion
        directly instead of pooling independent per-frame predictions.

        Args:
            num_classes (int): Number of classes in your dataset.
            video_version (str): One of VIDEO3D_ARCHITECTURES, e.g. "r3d_18".
            added_layers (int): Number of additional layers you want to add while finetuning your model.
            embedding_layer_size (int): If you chose added_layers to be 2, this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want classification or regression.
            output_dim (int): The dimension of the output of the regression model, default = 1.

        Input:
            Expects clips of shape [B, T, C, H, W] - the layout DeepTune's
            ParquetVideoDataset produces - and internally permutes to the
            [B, C, T, H, W] layout torchvision's video models expect.
        """

        super(adjustedVideo3D, self).__init__()

        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

        self.num_classes = num_classes
        self.added_layers = added_layers
        self.embedding_layer_size = embedding_layer_size
        self.freeze_backbone = freeze_backbone
        self.video_version = video_version
        self.task_type = task_type
        self.output_dim = output_dim

        self.model, self.fc_in = _build_backbone(video_version)
        self.flatten = nn.Flatten()

        if self.freeze_backbone:
            print('Backbone Parameters are frozen!')
            for param in self.model.parameters():
                param.requires_grad = False

        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.fc_in, self.embedding_layer_size)

            if self.task_type == 'cls':
                self.fc2 = nn.Linear(self.embedding_layer_size, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.embedding_layer_size, self.output_dim)

        elif self.added_layers == 1:

            if self.task_type == 'cls':
                self.fc1 = nn.Linear(self.fc_in, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.fc_in, self.output_dim)
        else:
            self.fc1 = None

    def forward(self, x, extract_embed=False):

        # x: [B, T, C, H, W] -> torchvision's video models expect [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.model(x)
        x = self.flatten(x)

        if self.added_layers == 1 and extract_embed:
            return x

        elif self.added_layers == 2 and extract_embed:
            x = self.fc1(x)
            return x

        if self.added_layers == 2:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)

        elif self.added_layers == 1:
            x = self.fc1(x)

        return x
