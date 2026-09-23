import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from desktop.config import RunConfig, validate_config
from desktop.runner import start_run
from handlers.split_dataset import split_dataset
from handlers.split_planning import allocate_split_counts


def split_frame(frame, tmp_path, mode='cls'):
    source = tmp_path / 'source.parquet'
    frame.to_parquet(source, index=False)
    return [pd.read_parquet(path) for path in split_dataset(
        .7, .1, .2, source, tmp_path / 'output', True, False, 'labels', 'video', mode=mode)]


def test_four_videos_report_exact_class_shortfall_before_creating_splits(tmp_path):
    with pytest.raises(ValueError) as error:
        split_frame(pd.DataFrame({'labels': ['cat', 'cat', 'dog', 'dog']}), tmp_path)
    message = str(error.value)
    assert 'at least 3 different clips' in message
    assert "'cat': 2 (add 1)" in message and "'dog': 2 (add 1)" in message
    assert 'n_samples' not in message
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('counts', [(3, 3), (4, 4), (3, 20), (3, 4, 5), (20, 20)])
def test_small_stratified_splits_keep_each_class_without_reusing_rows(tmp_path, counts):
    frame = pd.DataFrame({'row_id': range(sum(counts)), 'labels': np.repeat(range(len(counts)), counts)})
    parts = split_frame(frame, tmp_path)
    assert sum(map(len, parts)) == len(frame)
    assert all(set(part.labels) == set(frame.labels) for part in parts)
    assert pd.concat(parts).row_id.nunique() == len(frame)
    first_ids = [part.row_id.tolist() for part in parts]
    assert [part.row_id.tolist() for part in split_frame(frame, tmp_path)] == first_ids
    summary = json.loads(next((tmp_path / 'output').glob('*/split_summary.json')).read_text())
    assert list(summary['sample_counts'].values()) == list(map(len, parts))
    if counts == (3, 3):
        assert list(map(len, parts)) == [2, 2, 2]
    if counts == (20, 20):
        assert list(map(len, parts)) == [28, 4, 8]


@pytest.mark.parametrize('count', [3, 4, 5, 6, 7, 10, 100, 1500])
def test_regression_rounding_always_keeps_three_disjoint_nonempty_splits(tmp_path, count):
    frame = pd.DataFrame({'row_id': range(count), 'labels': np.linspace(.1, 2.5, count)})
    parts = split_frame(frame, tmp_path, mode='reg')
    assert all(len(part) for part in parts)
    assert pd.concat(parts).row_id.nunique() == count
    assert pd.concat(parts).sort_values('row_id').labels.tolist() == frame.labels.tolist()


def test_split_rounding_keeps_large_dataset_ratios():
    assert allocate_split_counts(1500) == (1050, 150, 300)
    assert allocate_split_counts(3) == (1, 1, 1)


def test_video_csv_preflight_blocks_launch_and_refreshes_after_adding_clips(tmp_path):
    source = tmp_path / 'clips.csv'
    pd.DataFrame({'videos': ['one.mp4', 'two.mp4', 'three.mp4', 'four.mp4'],
                  'labels': ['cat', 'cat', 'dog', 'dog']}).to_csv(source, index=False)
    cfg = RunConfig(modality='video', df=source, out=tmp_path / 'output', model_version='r3d_18')
    issues = validate_config(cfg)
    assert any("'cat': 2 (add 1)" in issue for issue in issues)
    with pytest.raises(ValueError, match='at least 3 different clips'):
        start_run(cfg)
    assert not cfg.out.exists()
    # The counts cache must invalidate when the manifest changes.
    pd.DataFrame({'videos': [f'{n}.mp4' for n in range(6)], 'labels': ['cat'] * 3 + ['dog'] * 3}).to_csv(source, index=False)
    assert validate_config(cfg) == []


def test_parquet_preflight_reads_labels_without_materializing_video_payloads(tmp_path):
    source = tmp_path / 'clips.parquet'
    pd.DataFrame({'videos': [b'video'] * 4, 'labels': ['cat', 'cat', 'dog', 'dog']}).to_parquet(source)
    cfg = RunConfig(modality='video', df=source, out=tmp_path / 'out', raw_data=False)
    assert any('at least 3 different clips' in issue for issue in validate_config(cfg))
    assert not any('at least 3 different clips' in issue for issue in validate_config(replace(cfg, mode='reg')))
