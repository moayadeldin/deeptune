import torch
from torch.utils.data import DataLoader, TensorDataset
from src.vision.swin import adjustedSwin
from src.vision.swin_peft import adjustedPeftSwin
from trainers.vision.trainer import Trainer

def create_dummy_loader(num_classes=3):
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, num_classes, (8,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2)

def test_trainer_step_classification():
    model = adjustedSwin(num_classes=3, swin_version="dummy", added_layers=1, task_type="cls")
    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        mode="cls",
        num_epochs=1,
        output_dir=None
    )

    trainer.train()
    val_loss, val_acc = trainer.validate()
    
    assert isinstance(val_loss, float)
    assert 0 <= val_acc <= 1

def test_trainer_peft_step_classification():
    model = adjustedPeftSwin(num_classes=3, swin_version="dummy", added_layers=1, task_type="cls")
    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        mode="cls",
        num_epochs=1,
        output_dir=None
    )

    trainer.train()
    val_loss, val_acc = trainer.validate()
    
    assert isinstance(val_loss, float)
    assert 0 <= val_acc <= 1
