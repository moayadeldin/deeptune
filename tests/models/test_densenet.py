import torch
from src.vision.densenet import adjustedDenseNet
from src.vision.densenet_peft import adjustedPeftDenseNet

def test_densenet_model_initialization():
    model = adjustedDenseNet(num_classes=10, densenet_version="dummy", added_layers=2, embedding_layer_size=512)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 10), "Output shape should match number of classes"
def test_densenet_peft_model_initialization():
    model = adjustedPeftDenseNet(num_classes=10, densenet_version="dummy", added_layers=2, lora_attention_dimension=512)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 10), "Output shape should match number of classes"
def test_densenet_model_regression_mode_initialization():
    model = adjustedDenseNet(num_classes=1, densenet_version="dummy", added_layers=1, embedding_layer_size=512,task_type='reg', output_dim=1)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1), "Output should match regression output shape"
def test_densenet_peft_model_regression_mode_initialization():
    model = adjustedPeftDenseNet(num_classes=1, densenet_version="dummy", added_layers=1,lora_attention_dimension=512, task_type='reg', output_dim=1)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1), "Output should match regression output shape"
