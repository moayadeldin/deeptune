import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluators.vision.evaluator import TestTrainer as Evaluator
from trainers.vision.trainer import Trainer


def test_binary_auroc_and_accuracy_are_saved(tmp_path):
    features = torch.tensor([[6., 0.], [0., 6.], [4., 0.], [0., 4.]])
    loader = DataLoader(TensorDataset(features, torch.tensor([0, 1, 0, 1])), batch_size=3)
    result = Evaluator(nn.Identity(), 3, loader, 'cls', tmp_path, device='cpu').test()
    assert result['accuracy'] == 1
    assert result['auroc'] == 1
    assert json.loads((tmp_path / 'full_metrics.json').read_text()) == result


def test_regression_metrics_are_returned_and_use_sample_weighted_loss(tmp_path):
    features = torch.tensor([[1.], [2.], [3.]])
    labels = torch.tensor([1., 2., 0.])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    result = Evaluator(nn.Identity(), 2, loader, 'reg', tmp_path, device='cpu').test()
    assert result['loss'] == 3
    assert result['mae'] == 1
    assert result['rmse'] == pytest.approx(3 ** .5)
    assert (tmp_path / 'full_metrics.json').is_file()


def test_regression_validation_compares_matching_shapes(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    loader = DataLoader(TensorDataset(torch.tensor([[1.], [2.]]), torch.tensor([1., 2.])), batch_size=2)
    trainer = Trainer(model, loader, loader, .001, 'reg', 1, tmp_path)
    assert trainer.validate() == 0
    model.weight.data.zero_()
    assert trainer.validate() == pytest.approx(2.5)
