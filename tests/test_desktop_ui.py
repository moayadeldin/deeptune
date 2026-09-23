"""Exercise real Dear PyGui widget callbacks without opening a viewport."""
from dataclasses import replace

import pandas as pd
import pytest

from desktop import app
from desktop.config import RunConfig, RunResults


def dispatch(ui, item, data=None):
    """Use the same registered callback dispatch as the desktop event loop."""
    callback = ui.dpg.get_item_callback(item)
    assert callback is not None
    ui.dpg.run_callbacks([[callback, item, data, ui.dpg.get_item_user_data(item)]])


@pytest.fixture
def ui():
    app.dpg.create_context()
    app.build_ui()
    try:
        yield app
    finally:
        if app._preview_future is not None:
            try:
                app._preview_future.result(timeout=10)
            except Exception:
                pass
            app._preview_future = None
        app._video_clips.clear()
        app._pending_video_paths = []
        app._preview_source_config = None
        app.stop_video_playback(timeout=2)
        app.dpg.destroy_context()


def test_each_modality_has_correct_source_picker_and_valid_model(ui):
    for modality in ui.gc.MODALITIES:
        ui.dpg.set_value('modality_radio', modality)
        ui.on_modality_changed()
        assert ui.dpg.get_value('model_version_combo') in ui.gc.MODEL_VERSIONS_BY_MODALITY[modality]
        folder = modality in ('images', 'video')
        assert ui.dpg.get_item_configuration('browse_folder_button')['show'] == folder
        assert ui.dpg.get_item_configuration('browse_file_button')['show'] == (not folder or modality == 'video')
        ui.dpg.set_value('raw_data_checkbox', False)
        ui.on_raw_data_toggled()
        assert ui.dpg.get_item_configuration('browse_file_button')['show']
        ui.dpg.set_value('raw_data_checkbox', True)


def test_restored_settings_update_every_widget_and_readiness(ui, tmp_path):
    cfg = RunConfig(modality='video', df=tmp_path, out=tmp_path / 'out', model_version='resnet18',
                    batch_size=4, num_epochs=3, num_frames=5, pooling='attention', use_peft=True)
    ui.apply_config(cfg)
    assert ui._build_run_config() == cfg
    assert ui.dpg.get_item_configuration('run_button')['enabled']
    assert '--num_frames 5' in ui.dpg.get_value('command_text')
    ui.apply_config(replace(cfg, batch_size=0))
    assert not ui.dpg.get_item_configuration('run_button')['enabled']
    assert 'Batch size' in ui.dpg.get_value('validation_text')


def test_tabpfn_regression_and_gpt2_peft_controls(ui, tmp_path):
    ui.apply_config(RunConfig(modality='tabular', model_version='tabpfn', mode='reg'))
    assert ui.dpg.get_item_configuration('mode_radio')['show']
    assert ui._build_run_config().mode == 'reg'
    ui.apply_config(RunConfig(modality='text', model_version='gpt2', use_peft=True))
    assert not ui.dpg.get_value('use_peft_checkbox')
    assert not ui.dpg.get_item_configuration('use_peft_checkbox')['enabled']


def test_partial_regression_results_render_and_clear(ui, tmp_path):
    ui.render_results(RunResults(tmp_path, metrics={'mae': .12, 'auroc': None},
                                  training_log=pd.DataFrame({'epoch': [1, 2], 'val_loss': [.5, 'pending']})))
    assert ui.dpg.get_value('results_series_loss_plot') == [[], []]
    assert ui.dpg.get_value('results_series_valloss_plot') == [[1.], [.5]]
    assert ui._selected_results.run_dir == tmp_path
    ui.refresh_results(tmp_path)
    assert ui._selected_results is None
    assert ui.dpg.get_value('results_run_combo') == ''


def test_video_run_readiness_explains_shortfall_and_updates_when_clips_are_added(ui, tmp_path):
    source = tmp_path / 'clips.csv'
    def save(count):
        pd.DataFrame({'videos': [f'{i}.mp4' for i in range(count * 2)],
                      'labels': ['cat'] * count + ['dog'] * count}).to_csv(source, index=False)
    save(2)
    ui.apply_config(RunConfig(modality='video', df=source, out=tmp_path / 'out'))
    assert not ui.dpg.get_item_configuration('run_button')['enabled']
    assert "'cat': 2 (add 1)" in ui.dpg.get_value('validation_text')
    save(3)
    ui.on_config_changed()
    assert ui.dpg.get_item_configuration('run_button')['enabled']


def test_select_mp4_preview_and_build_labeled_dataset(ui, tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from handlers.video_manifest import read_video_manifest
    monkeypatch.setattr(ui, 'SESSION_PATH', tmp_path / 'app-settings' / 'session.json')
    ui.apply_config(RunConfig(modality='video', out=tmp_path / 'out'))
    for label in ('walking', 'running'):
        path = tmp_path / f'{label} video.MP4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        assert writer.isOpened()
        for value in range(4):
            writer.write(np.full((48, 64, 3), value * 50, dtype=np.uint8))
        writer.release()
        dispatch(ui, 'file_dialog_data_file', {'selections': {path.name: str(path)}})
        # Selecting a file must visibly import it before a label is entered.
        assert str(path.resolve()) in ui._video_clips
        assert ui._video_clips[str(path.resolve())] == ''
        assert 'needs a label' in ui.dpg.get_value('video_import_status')
        ui._preview_future.result(timeout=10)
        ui.poll_preview()
        assert ui.dpg.does_item_exist('video_preview_texture')
        assert '64 x 48' in ui.dpg.get_value('preview_note')
        ui.dpg.set_value('video_label_input', label)
        dispatch(ui, 'apply_video_label_button')
    assert len(ui._video_clips) == 2
    dispatch(ui, 'use_video_dataset_button')
    cfg = ui._build_run_config()
    frame = read_video_manifest(cfg.df)
    assert frame['labels'].tolist() == ['walking', 'running']
    assert cfg.num_classes == 2 and cfg.raw_data
    assert cfg.df.suffix == '.csv'
    ui._preview_future.result(timeout=10)
    ui.poll_preview()
    ui._video_clips.clear()
    ui.on_load_preview()
    ui._preview_future.result(timeout=10)
    ui.poll_preview()
    assert list(ui._video_clips.values()) == ['walking', 'running']


def test_missing_label_is_reported_next_to_import_controls(ui, tmp_path):
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'video')
    ui._pending_video_paths = [path]
    ui._video_clips[str(path)] = ''
    ui._render_video_clips()
    ui.on_add_video_clips()
    assert 'Enter a label' in ui.dpg.get_value('video_import_status')
    ui.on_use_video_clips()
    assert 'needs a label' in ui.dpg.get_value('video_import_status')
    assert not ui.dpg.get_item_configuration('use_video_dataset_button')['enabled']


def test_individual_clip_labels_and_clear_selection(ui, tmp_path):
    path = str(tmp_path / 'clip.mp4')
    ui._video_clips[path] = ''
    ui._pending_video_paths = [tmp_path / 'clip.mp4']
    ui._render_video_clips()
    ui.on_video_clip_label_changed(app_data='walking', user_data=path)
    assert ui._video_clips[path] == 'walking'
    assert '1 clips / 1 classes' in ui.dpg.get_value('video_clips_count')
    assert ui.dpg.get_item_configuration('use_video_dataset_button')['enabled']
    ui.on_clear_video_clips()
    assert not ui._pending_video_paths and not ui._video_clips
    assert 'No clips selected' in ui.dpg.get_value('video_selection_text')


def test_missing_file_error_is_shown_in_the_import_panel(ui, tmp_path):
    ui.apply_config(RunConfig(modality='video'))
    dispatch(ui, 'file_dialog_data_file', {'file_path_name': str(tmp_path / 'missing.mp4')})
    assert not ui._video_clips
    assert 'file not found' in ui.dpg.get_value('video_import_status')


def test_decode_error_is_shown_beside_the_imported_file(ui, tmp_path):
    path = tmp_path / 'corrupt.mp4'
    path.write_bytes(b'not a video')
    ui.apply_config(RunConfig(modality='video'))
    dispatch(ui, 'file_dialog_data_file', {'file_path_name': str(path)})
    assert str(path.resolve()) in ui._video_clips
    try:
        ui._preview_future.result(timeout=10)
    except ValueError:
        pass
    ui.poll_preview()
    assert 'Could not preview this video' in ui.dpg.get_value('video_decode_status')


def test_background_manifest_preview_does_not_undo_clip_label_edits(ui, tmp_path):
    from concurrent.futures import Future
    from desktop.dataset_preview import DatasetPreview
    path = str(tmp_path / 'clip.mp4')
    cfg = RunConfig(modality='video', df=tmp_path / 'list.csv')
    ui.apply_config(cfg)
    ui._video_clips[path] = 'before'
    ui._render_video_clips()
    ui._preview_config = ui.gc.resolved_config(cfg)
    ui._preview_video_clips_snapshot = ui._video_clips.copy()
    ui._preview_future = Future()
    ui._preview_future.set_result(DatasetPreview('1 labeled clip', video_entries={path: 'before'}))
    ui.on_video_clip_label_changed(app_data='after', user_data=path)
    ui.poll_preview()
    assert ui._video_clips[path] == 'after'


def test_video_preview_updates_texture_seeks_and_preserves_training_source(ui, tmp_path):
    import cv2
    import numpy as np
    import time

    path = tmp_path / 'preview.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for value in range(20):
        writer.write(np.full((48, 64, 3), value * 10, dtype=np.uint8))
    writer.release()
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame({'videos': [str(path)], 'labels': ['walk']}).to_csv(manifest, index=False)
    ui.apply_config(RunConfig(modality='video', df=manifest))
    ui.on_nav_selected(user_data='dataset')
    ui.on_preview_video_clip(user_data=str(path))
    ui._preview_future.result(timeout=10)
    ui.poll_preview()
    assert ui._build_run_config().df == manifest
    first = np.asarray(ui.dpg.get_value('video_preview_texture')).copy()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and ui._video_player.state.frame_index < 3:
        ui.poll_video_playback()
        time.sleep(.01)
    ui.poll_video_playback()
    assert not np.array_equal(first, np.asarray(ui.dpg.get_value('video_preview_texture')))
    assert 'Playing' in ui.dpg.get_value('video_playback_status')
    dispatch(ui, 'video_play_button')
    assert not ui._video_player.state.playing
    dispatch(ui, 'video_seek_slider', 1.5)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and ui._video_player.state.frame_index != 15:
        time.sleep(.01)
    ui.poll_video_playback()
    assert ui._video_player.state.frame_index == 15
    assert ui.dpg.get_value('video_seek_slider') == pytest.approx(1.5)
    dispatch(ui, 'video_restart_button')
    assert ui._video_player.state.playing
    ui.on_nav_selected(user_data='model')
    assert not ui._video_player.state.playing
    player = ui._video_player
    ui.apply_config(RunConfig(modality='images'))
    player.thread.join(timeout=2)
    assert not player.thread.is_alive()
    assert not ui.dpg.does_item_exist('video_preview_texture')
