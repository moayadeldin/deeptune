import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from datasets.timeseries_data import prepare_timeseries
from helpers import save_timeseries_prediction_to_json


def test_integer_time_indices_are_not_converted_to_nanoseconds():
    frame, groups = prepare_timeseries(pd.DataFrame({'time': [10, 11, 12], 'labels': [1.2, 1.3, 1.4]}), 'time', 'labels')
    assert frame.time_idx.tolist() == [10, 11, 12]
    assert groups == ['group']


def test_subhour_timestamps_remain_distinct_and_ordered():
    frame = pd.DataFrame({'time': ['2026-01-01 00:02', '2026-01-01 00:00', '2026-01-01 00:01'], 'labels': [3., 1., 2.]})
    result, _ = prepare_timeseries(frame, 'time', 'labels')
    assert result.time_idx.tolist() == [0, 1, 2]
    assert result.labels.tolist() == [1., 2., 3.]


def test_duplicate_time_indices_require_distinct_series():
    frame = pd.DataFrame({'time': [0, 1, 0, 1], 'patient': ['a', 'a', 'b', 'b'], 'labels': [1, 2, 3, 4]})
    result, groups = prepare_timeseries(frame, 'time', 'labels', ['patient'])
    assert groups == ['patient'] and len(result) == 4
    with pytest.raises(ValueError, match='exactly one'):
        prepare_timeseries(frame, 'time', 'labels')


def test_forecast_serialization_supports_target_weight_tuple(tmp_path):
    prediction = SimpleNamespace(output=torch.tensor([[1.], [2.]]), x={'decoder_target': torch.tensor([[1.2], [2.2]])},
                                 decoder_lengths=torch.tensor([1, 1]), y=(torch.tensor([[1.2], [2.2]]), None))
    save_timeseries_prediction_to_json(prediction, tmp_path)
    result = json.loads((tmp_path / 'prediction_output.json').read_text())
    assert result['model_prediction'] == [[1.], [2.]]
    assert result['ground_truth_target'][1] is None
