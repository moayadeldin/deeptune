import torch
import torch.nn as nn


class VideoFramePoolClassifier(nn.Module):
    """
    Wraps one of DeepTune's existing 2D vision backbones (ResNet, EfficientNet,
    DenseNet, Swin, ViT, ConvNeXt, VGG - PEFT or not) to classify video clips.

    Each sampled frame of the clip is passed through the *same* backbone
    (weights are shared across frames - it is applied frame-by-frame, not one
    backbone per frame), producing one set of per-frame outputs. Those are then
    pooled over time - either by a simple average, or by a small learned
    attention layer that lets the model weigh more informative frames more
    heavily - into a single clip-level prediction.

    This keeps video classification fully compatible with DeepTune's existing
    Trainer/TestTrainer pipeline (forward(x) -> [B, num_classes]) and PEFT/LoRA
    infrastructure, without introducing a whole separate spatio-temporal model
    zoo. See src/video/video3d.py for DeepTune's native 3D-CNN/video-transformer
    alternative.

    Args:
        frame_model_cls (Type[nn.Module]): One of DeepTune's adjusted*Net classes
            (e.g. adjustedResNet or adjustedPeftResNet) used to build the shared
            per-frame backbone. Every one of these classes accepts the same
            positional constructor signature: (num_classes, model_version,
            added_layers, embedding_layer_size, freeze_backbone, task_type=...).
        frame_model_version (str): Version string of the underlying 2D architecture
            (e.g. "resnet50").
        num_classes (int): Number of classes in your dataset.
        added_layers (int): Number of additional layers on the per-frame backbone.
        embedding_layer_size (int): Size of the intermediate embedding layer if
            added_layers=2 (or the LoRA attention dimension, for PEFT backbones).
        freeze_backbone (bool): Whether to freeze the per-frame backbone's weights.
        task_type (str): "cls" or "reg".
        pooling (str): "mean" or "attention" - how per-frame outputs are combined
            over time into a single clip-level prediction.
    """

    def __init__(
        self,
        frame_model_cls,
        frame_model_version: str,
        num_classes: int,
        added_layers: int = 2,
        embedding_layer_size: int = 1000,
        freeze_backbone: bool = False,
        task_type: str = "cls",
        pooling: str = "mean",
    ):
        super().__init__()

        assert pooling in ("mean", "attention"), "pooling must be 'mean' or 'attention'"
        assert task_type in ("cls", "reg"), "task_type must be 'cls' or 'reg'"

        self.pooling = pooling
        self.task_type = task_type
        self.added_layers = added_layers

        # Every adjusted*Net class (PEFT or not) shares this exact positional
        # signature - it's how DeepTune's own vision trainer instantiates them
        # generically regardless of architecture.
        self.backbone = frame_model_cls(
            num_classes, frame_model_version, added_layers, embedding_layer_size, freeze_backbone, task_type=task_type,
        )

        per_frame_out_dim = num_classes if task_type == "cls" else 1

        if pooling == "attention":
            self.attention = nn.Linear(per_frame_out_dim, 1)

    def forward(self, x, extract_embed=False):
        """
        Args:
            x (Tensor): Video clip batch of shape [B, T, C, H, W].
            extract_embed (bool): If True, returns a pooled clip-level embedding
                (mean-pooled over the backbone's own per-frame embeddings) instead
                of a prediction.

        Returns:
            Tensor: [B, num_classes] (or [B, 1] for regression), or
                [B, embedding_dim] if extract_embed=True.
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        out = self.backbone(x, extract_embed=extract_embed)
        out = out.reshape(B, T, -1)

        if extract_embed:
            # Attention pooling is trained to weigh class-logit-shaped outputs,
            # not raw embeddings, so embeddings are always mean-pooled over time.
            return out.mean(dim=1)

        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        else:
            weights = torch.softmax(self.attention(out), dim=1)  # [B, T, 1]
            pooled = (out * weights).sum(dim=1)

        return pooled
