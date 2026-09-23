import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desktop.config import (RunConfig, build_argv, effective_mode, load_config, loss_series,
                            metric_rows, read_run_results, resolved_config, save_config, validate_config)
from desktop.dataset_preview import preview_dataset
from desktop.log_stream import LineBuffer
from handlers.split_dataset import split_dataset


def split(frame, tmp_path, **kwargs):
    path = tmp_path / 'source.parquet'
    frame.to_parquet(path, index=False)
    values = dict(train_size=.7, val_size=.1, test_size=.2, df_path=path, out_dir=tmp_path / 'out',
                  fixed_seed=True, disable_numerical_encoding=False, target_column='target', modality='tabular')
    values.update(kwargs)
    return [pd.read_parquet(p) for p in split_dataset(**values)]


def test_split_keeps_every_row_and_stratifies(tmp_path):
    frame = pd.DataFrame({'row_id': range(1500), 'target': ['cat', 'dog', 'bird'] * 500})
    parts = split(frame, tmp_path)
    assert sum(map(len, parts)) == 1500
    assert pd.concat(parts).row_id.nunique() == 1500
    assert all(part.labels.nunique() == 3 for part in parts)
    assert set(parts[0].row_id).isdisjoint(parts[1].row_id)
    assert set(parts[0].row_id).isdisjoint(parts[2].row_id)
    again = split(frame, tmp_path)
    assert [p.row_id.tolist() for p in parts] == [p.row_id.tolist() for p in again]


def test_regression_preserves_fractional_targets(tmp_path):
    frame = pd.DataFrame({'row_id': range(100), 'target': np.linspace(-10.5, 25.75, 100)})
    parts = split(frame, tmp_path, mode='reg')
    actual = pd.concat(parts).sort_values('row_id').labels.to_numpy()
    np.testing.assert_array_equal(actual, frame.target.to_numpy())
    assert not list((tmp_path / 'out').glob('*/label_mapping.json'))


@pytest.mark.parametrize('mode', ['cls', 'reg'])
def test_group_split_has_no_leakage_and_records_original_indices(tmp_path, mode):
    frame = pd.DataFrame({'row_id': range(240), 'patient': np.repeat(range(40), 6), 'target': [0, 1] * 120})
    train, val, test = split(frame, tmp_path, grouper='patient', mode=mode)
    assert set(train.patient).isdisjoint(val.patient)
    assert set(train.patient).isdisjoint(test.patient)
    assert set(val.patient).isdisjoint(test.patient)
    indices = pd.read_csv(next((tmp_path / 'out').glob('*/test_indices.csv')))
    assert indices['Test Set Indices in the Original Dataframe'].tolist() == test.row_id.tolist()


def test_forecasting_split_is_chronological_across_groups(tmp_path):
    frame = pd.DataFrame({'time': list(range(50)) * 2, 'group': ['a'] * 50 + ['b'] * 50,
                          'target': np.linspace(1.2, 15.3, 100)}).sample(frac=1, random_state=4)
    train, val, test = split(frame, tmp_path, modality='timeseries', time_idx_column='time', grouper='group')
    assert train.time.max() < val.time.min() <= val.time.max() < test.time.min()
    assert sum(map(len, (train, val, test))) == len(frame)
    assert all(set(part.group) == {'a', 'b'} for part in (train, val, test))
    assert set(pd.concat([train, val, test]).labels) == set(frame.target)


@pytest.mark.parametrize('changes, message', [
    ({'train_size': 0}, 'positive'), ({'test_size': .4}, 'sum to 1'),
    ({'target_column': 'missing'}, 'Target column'), ({'grouper': 'missing'}, 'Grouper'),
])
def test_split_errors_explain_invalid_inputs(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        split(pd.DataFrame({'target': [0, 1] * 50}), tmp_path, **changes)


@pytest.mark.parametrize('changes, message', [
    ({'batch_size': 0}, 'Batch size'), ({'num_epochs': -1}, 'Epochs'),
    ({'learning_rate': float('nan')}, 'Learning rate'), ({'learning_rate': 0}, 'Learning rate'),
    ({'model_version': 'gpt2'}, 'supported'), ({'embed_size': 0}, 'Embedding size'),
    ({'modality': 'video', 'num_frames': 0}, 'Frames'),
    ({'modality': 'video', 'model_version': 'mvit_v2_s'}, '16 frames'),
])
def test_invalid_settings_are_caught_before_launch(tmp_path, changes, message):
    cfg = replace(RunConfig(df=tmp_path, out=tmp_path / 'output'), **changes)
    assert any(message in issue for issue in validate_config(cfg))


def test_settings_roundtrip_and_relative_paths(tmp_path):
    cfg = RunConfig(df=tmp_path / 'data', out=tmp_path / 'out', modality='video', num_frames=16,
                    pooling='attention', use_peft=True)
    path = tmp_path / 'settings.json'
    save_config(cfg, path)
    assert load_config(path) == resolved_config(cfg)
    contents = json.loads(path.read_text())
    contents['config']['df'] = 'relative/data'
    path.write_text(json.dumps(contents))
    assert load_config(path).df == tmp_path / 'relative' / 'data'
    contents['config']['batch_size'] = 'invalid'
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='batch_size'):
        load_config(path)


def test_task_selection_reaches_tabular_cli():
    cfg = RunConfig(df=Path('data'), out=Path('out'), modality='tabular', model_version='gandalf', gandalf_type='regression')
    args = build_argv(cfg)
    assert args[args.index('--mode') + 1] == 'reg'
    assert effective_mode(replace(cfg, modality='timeseries')) == 'reg'


def test_partial_results_do_not_crash_or_invent_metrics(tmp_path):
    (tmp_path / 'eval_output_model').mkdir()
    (tmp_path / 'trainval_output_model').mkdir()
    (tmp_path / 'eval_output_model' / 'full_metrics.json').write_text('{unfinished')
    (tmp_path / 'trainval_output_model' / 'training_log.csv').write_text('')
    result = read_run_results(tmp_path)
    assert result.metrics is None and result.training_log is None
    assert len(result.warnings) == 2
    assert metric_rows({'mae': .125, 'auroc': None}) == [('mae', '0.125'), ('auroc', 'Unavailable')]


def test_gandalf_and_tabpfn_artifacts_are_discoverable(tmp_path):
    (tmp_path / 'test_output_GANDALF').mkdir()
    (tmp_path / 'test_output_GANDALF' / 'full_metrics.json').write_text('{"test_loss":0.1}')
    (tmp_path / 'train_output_TABPFN').mkdir()
    model = tmp_path / 'train_output_TABPFN' / 'trained_cls.tabpfn_fit'
    model.write_text('checkpoint placeholder')
    result = read_run_results(tmp_path)
    assert result.metrics == {'test_loss': .1}
    assert result.checkpoint_path == model


def test_loss_plot_ignores_partial_and_non_numeric_rows():
    frame = pd.DataFrame({'epoch': [1, 2, 3], 'val_loss': [1.2, 'pending', .2]})
    assert loss_series(frame, 'epoch_loss') == [[], []]
    assert loss_series(frame, 'val_loss') == [[1., 3.], [1.2, .2]]


def test_logs_handle_split_crlf_tqdm_ansi_and_bounded_history():
    buffer = LineBuffer(max_lines=2)
    buffer.feed('first\r')
    buffer.feed('\nsecond\n\x1b[32m10%\r20%\x1b[0m')
    assert buffer.render() == 'first\nsecond\n20%'
    buffer.feed('\nlast\n')
    assert buffer.render() == '20%\nlast\n'
    buffer.clear()
    assert buffer.render() == ''


def test_preview_is_bounded_and_hides_binary_payload(tmp_path):
    path = tmp_path / 'video.parquet'
    pd.DataFrame({'videos': [b'not-decoded-in-preview'] * 100, 'labels': [0, 1] * 50}).to_parquet(path)
    result = preview_dataset(RunConfig(modality='video', df=path, raw_data=False), limit=5)
    assert len(result.frame) == 5
    assert '100 rows' in result.note
    assert result.frame.videos.iloc[0] == '<22 bytes>'


def test_xlsx_preview_and_csv_preview_agree(tmp_path):
    frame = pd.DataFrame({'text': ['one', 'two'], 'labels': [0, 1]})
    csv, xlsx = tmp_path / 'data.csv', tmp_path / 'data.xlsx'
    frame.to_csv(csv, index=False)
    frame.to_excel(xlsx, index=False)
    first = preview_dataset(RunConfig(modality='text', df=csv))
    second = preview_dataset(RunConfig(modality='text', df=xlsx))
    pd.testing.assert_frame_equal(first.frame, second.frame)
