from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from embed.tabular import gandalf_embeddings
from evaluators.tabular import evaluate_tabpfn


class SavedArgs:
    def save_args(self, path):
        path.mkdir(parents=True, exist_ok=True)


def test_gandalf_export_keeps_all_batches_and_fractional_labels(tmp_path, monkeypatch):
    frame = pd.DataFrame({'value': np.arange(23), 'target': np.arange(23) + .25})
    source = tmp_path / 'source.parquet'
    frame.to_parquet(source)

    class FittedModel:
        def __init__(self):
            self.model = nn.Module()
            self.model.backbone = nn.Identity()

        def predict(self, data, progress_bar=None):
            pd.testing.assert_frame_equal(data, frame)
            for start in range(0, len(data), 5):
                # Represents fitted preprocessing owned by predict(), rather
                # than an encoder independently fitted on the holdout data.
                values = torch.tensor((data.value.iloc[start:start + 5].to_numpy() - 10) / 5).reshape(-1, 1)
                self.model.backbone(values)

    monkeypatch.setattr(gandalf_embeddings.TabularModel, 'load_model', lambda _: FittedModel())
    directory, shape = gandalf_embeddings.embed(source, tmp_path, tmp_path, ['value'], [], 5,
                                                'target', SavedArgs(), None, device='cpu')
    result = pd.read_parquet(next(directory.glob('*.parquet')))
    assert shape == (23, 2)
    np.testing.assert_array_equal(result['label'], frame.target)
    np.testing.assert_allclose(result.iloc[:, 0], (frame.value - 10) / 5)


def test_finetuned_tabpfn_regression_returns_a_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_tabpfn, 'load', lambda _: SimpleNamespace(predict=lambda frame: np.array([1., 2., 3.])))
    result = evaluate_tabpfn.evaluate_tabpfn(pd.DataFrame({'x': [0, 1, 2]}), pd.Series([1., 2., 4.]),
                                            tmp_path, SavedArgs(), tmp_path / 'model.joblib', 'reg', True)
    assert isinstance(result, dict)
    assert result['Mean Squared Error'] == 1 / 3
