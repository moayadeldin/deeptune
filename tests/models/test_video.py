import torch
from src.video.video3d import adjustedVideo3D
from src.video.video3d_peft import adjustedPeftVideo3D
from src.video.frame_pool import VideoFramePoolClassifier
from src.vision.resnet import adjustedResNet
from src.vision.resnet_peft import adjustedPeftResNet


# ---------------------------------------------------------------------------
# Native spatio-temporal video models (src/video/video3d.py)
# ---------------------------------------------------------------------------

def test_video3d_model_initialization():
    model = adjustedVideo3D(num_classes=10, video_version="dummy_video3d", added_layers=2, embedding_layer_size=64)
    x = torch.randn(2, 4, 3, 112, 112)  # [B, T, C, H, W]
    out = model(x)
    assert out.shape == (2, 10), "Output shape should match number of classes"


def test_video3d_peft_model_initialization():
    model = adjustedPeftVideo3D(num_classes=10, video_version="dummy_video3d", added_layers=2, lora_attention_dimension=64)
    x = torch.randn(2, 4, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 10), "Output shape should match number of classes"


def test_video3d_model_regression_mode_initialization():
    model = adjustedVideo3D(num_classes=1, video_version="dummy_video3d", added_layers=1, embedding_layer_size=64, task_type="reg", output_dim=1)
    x = torch.randn(2, 3, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 1), "Output should match regression output shape"


def test_video3d_extract_embed():
    model = adjustedVideo3D(num_classes=5, video_version="dummy_video3d", added_layers=2, embedding_layer_size=48)
    x = torch.randn(2, 4, 3, 112, 112)
    embed = model(x, extract_embed=True)
    assert embed.shape == (2, 48), "Embedding shape should match embedding_layer_size"


def test_video3d_invalid_version_raises():
    try:
        adjustedVideo3D(num_classes=2, video_version="not_a_real_architecture")
        assert False, "Expected a ValueError for an unsupported video_version"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Frame-sampling video models built on top of existing 2D vision backbones
# (src/video/frame_pool.py)
# ---------------------------------------------------------------------------

def test_frame_pool_mean_pooling_initialization():
    model = VideoFramePoolClassifier(
        frame_model_cls=adjustedResNet, frame_model_version="dummy",
        num_classes=5, added_layers=2, embedding_layer_size=64, pooling="mean",
    )
    x = torch.randn(2, 4, 3, 224, 224)  # [B, T, C, H, W]
    out = model(x)
    assert out.shape == (2, 5), "Output shape should match number of classes"


def test_frame_pool_attention_pooling_initialization():
    model = VideoFramePoolClassifier(
        frame_model_cls=adjustedResNet, frame_model_version="dummy",
        num_classes=5, added_layers=1, embedding_layer_size=64, pooling="attention",
    )
    x = torch.randn(2, 4, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5), "Output shape should match number of classes"
    assert hasattr(model, "attention"), "Attention pooling should add a learned attention layer"


def test_frame_pool_peft_backbone():
    model = VideoFramePoolClassifier(
        frame_model_cls=adjustedPeftResNet, frame_model_version="dummy",
        num_classes=5, added_layers=2, embedding_layer_size=64, pooling="mean",
    )
    x = torch.randn(2, 4, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5), "Output shape should match number of classes"


def test_frame_pool_extract_embed():
    model = VideoFramePoolClassifier(
        frame_model_cls=adjustedResNet, frame_model_version="dummy",
        num_classes=5, added_layers=2, embedding_layer_size=48, pooling="attention",
    )
    x = torch.randn(2, 4, 3, 224, 224)
    embed = model(x, extract_embed=True)
    assert embed.shape == (2, 48), "Embedding shape should match embedding_layer_size regardless of pooling mode"


def test_frame_pool_backward_pass():
    model = VideoFramePoolClassifier(
        frame_model_cls=adjustedResNet, frame_model_version="dummy",
        num_classes=3, added_layers=1, embedding_layer_size=32, pooling="mean",
    )
    x = torch.randn(2, 2, 3, 224, 224)
    out = model(x)
    loss = out.sum()
    loss.backward()  # should not raise
