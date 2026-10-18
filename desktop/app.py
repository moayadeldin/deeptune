"""DeepTune desktop GUI.

Run with:  .venv\\Scripts\\python.exe desktop\\app.py

A DearPyGui front end for deeptune.py's one-call pipeline: pick a modality
and dataset, choose a model and its hyperparameters, launch the run, watch
its live log, and browse the results it produced -- everything the CLI's
--modality/--model_version/... flags already do, without having to remember
their names. Loading and launching stay light: only pandas + dearpygui are
imported here, never torch/transformers -- the run itself happens in a
subprocess (see runner.py), exactly like deeptune.py itself would be
invoked from a terminal.

Modeled on the sibling df-analyze project's desktop/app.py (same DearPyGui
shell, theme, and run/log/results plumbing), scaled down to DeepTune's
actual pipeline: one model per modality with clear hyperparameters, not an
AutoML leaderboard or a free-form preprocessing DAG, so there's no
workflow-canvas or notebook page here -- those are df-analyze-specific
concepts DeepTune has no equivalent of.
"""
from __future__ import annotations

import queue
import os
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Packaged builds (see packaging/) freeze this file into a single
# executable, so there is no separate python.exe to hand deeptune.py/
# infer.py to as a script path the way runner.py does in a normal
# checkout. In a frozen build, runner.py instead re-invokes this same
# executable with one of these sentinels as argv[1] and that script's own
# argv after it; handled here, before the GUI toolkit import, so a worker
# subprocess never pays for a GUI it doesn't use. This delegates straight
# into deeptune.main()/infer.main() -- the actual training/eval/embed/
# inference pipeline and its heavy imports (torch/transformers/...) still
# only happen in this worker process, never in the long-lived GUI process
# itself. An unfrozen `python desktop/app.py` never takes this branch;
# runner.py keeps using a real python.exe + the on-disk script path there.
if len(sys.argv) > 1 and sys.argv[1] in ("--deeptune-worker", "--infer-worker"):
    # PyInstaller's windowed bootloader doesn't reliably pick up
    # PYTHONIOENCODING for this process's stdio the way a plain python.exe
    # does -- confirmed by testing: even with runner.py's env=utf-8 set,
    # stdout still came up cp1252 here, so deeptune.py's emoji prints (e.g.
    # print_metrics_table's title) raised UnicodeEncodeError, discarding
    # the run's own success in an otherwise-completed training/eval/embed
    # job. Reconfigure the streams directly instead of relying on that env
    # var; harmless (and a no-op) when unfrozen, where the env var already
    # does the job.
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    if sys.argv[1] == "--deeptune-worker":
        import deeptune
        sys.exit(deeptune.main(sys.argv[2:]))
    else:
        import infer
        sys.exit(infer.main(sys.argv[2:]))

import dearpygui.dearpygui as dpg
import pandas as pd

from desktop import config as gc
from desktop import dpg_context as ui
from desktop.dpg_context import ItemId
from desktop.log_stream import LineBuffer
from desktop.runner import RunHandle, start_run, start_infer_run
from desktop.dataset_preview import preview_dataset
from desktop.video_player import VideoPlayer
from handlers.video_manifest import VIDEO_EXTENSIONS
from handlers.split_planning import classification_split_issues
from inference.data import IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Theme (same palette as df-analyze's desktop app, for a consistent look
# across both of this project's sibling tools)
# ---------------------------------------------------------------------------
BG_MAIN = (11, 17, 30)
BG_SIDEBAR = (8, 13, 24)
BG_CARD = (20, 28, 46)
BG_CARD_ALT = (28, 39, 63)
ACCENT = (61, 145, 235)
ACCENT_HOVER = (95, 173, 246)
ACCENT_ACTIVE = (43, 112, 191)
ACCENT_LIGHT = (150, 200, 250)
TEXT_PRIMARY = (226, 232, 240)
TEXT_MUTED = (138, 150, 172)
BORDER = (42, 54, 78)
SUCCESS = (58, 199, 130)
ERROR = (235, 105, 105)
WARNING = (232, 178, 78)

NAV_ITEMS = [
    ("overview", "01   Overview"),
    ("dataset", "02   Dataset"),
    ("model", "03   Model & Training"),
    ("run", "04   Run"),
    ("results", "05   Results"),
    ("test", "06   Test"),
]

_FONTS: dict[str, ItemId] = {}
_THEMES: dict[str, ItemId] = {}

_current_run: RunHandle | None = None
_line_buffer = LineBuffer()
_run_outdir: Path | None = None
_run_started = 0.0
_run_history: dict[str, Path] = {}
_selected_results: gc.RunResults | None = None
_preview_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dataset-preview')
_preview_future = None
_preview_config = None
_preview_source_config = None
_preview_video_clips_snapshot: dict[str, str] = {}
_video_player: VideoPlayer | None = None
_video_player_source: Path | None = None
_video_texture_size: tuple[int, int] | None = None
_scroll_to_video_preview = 0
_pending_video_paths: list[Path] = []
_video_clips: dict[str, str] = {}
SESSION_PATH = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.config'))) / 'DeepTune' / 'session.json'

# Test page: running an already-completed run's checkpoint on new data via
# infer.py (see desktop/runner.py's start_infer_run and desktop/config.py's
# InferenceConfig). Kept separate from the Run/Results page's state above
# rather than reused, since testing an older model and training a new one
# are independent actions a user could reasonably want at the same time.
_current_infer_run: RunHandle | None = None
_infer_line_buffer = LineBuffer()
_infer_out_dir: Path | None = None
_infer_started = 0.0
_test_run_history: dict[str, Path] = {}
_selected_test_run: gc.RunResults | None = None
_selected_test_results: gc.InferenceResults | None = None

CONFIG_TAGS = {
    'modality': 'modality_radio', 'df': 'data_path_input', 'raw_data': 'raw_data_checkbox',
    'out': 'outdir_input', 'target': 'target_input', 'grouper': 'grouper_input',
    'model_version': 'model_version_combo', 'num_classes': 'num_classes_input', 'mode': 'mode_radio',
    'added_layers': 'added_layers_combo', 'embed_size': 'embed_size_input',
    'freeze_backbone': 'freeze_backbone_checkbox', 'use_peft': 'use_peft_checkbox',
    'fixed_seed': 'fixed_seed_checkbox', 'batch_size': 'batch_size_input', 'num_epochs': 'num_epochs_input',
    'learning_rate': 'learning_rate_input', 'num_frames': 'num_frames_input', 'pooling': 'pooling_radio',
    'gandalf_type': 'gandalf_type_radio', 'continuous_cols': 'continuous_cols_input',
    'categorical_cols': 'categorical_cols_input', 'finetuning_mode': 'finetuning_mode_checkbox',
    'time_idx_column': 'time_idx_column_input',
}


def apply_config(cfg: gc.RunConfig) -> None:
    dpg.set_value('modality_radio', cfg.modality)
    on_modality_changed()
    for key, tag in CONFIG_TAGS.items():
        value = getattr(cfg, key)
        if key in ('df', 'out', 'added_layers'):
            value = str(value) if value is not None else ''
        elif isinstance(value, list):
            value = ', '.join(value)
        dpg.set_value(tag, value)
    on_model_version_changed()
    on_raw_data_toggled()
    on_config_changed()


def _dialog_path(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    selections = data.get('selections') or {}
    return next(iter(selections.values()), None) or data.get('file_path_name')


def on_save_settings(sender: ItemId = 0, app_data: Any = None) -> None:
    path = _dialog_path(app_data)
    if path:
        try:
            destination = Path(path).with_suffix('.json')
            gc.save_config(_build_run_config(), destination)
            _set_status(f'Settings saved: {destination.name}', SUCCESS)
        except (OSError, ValueError) as exc:
            _set_status(f'Cannot save settings: {exc}', ERROR)


def on_load_settings(sender: ItemId = 0, app_data: Any = None) -> None:
    path = _dialog_path(app_data)
    if path:
        try:
            apply_config(gc.load_config(Path(path)))
            _set_status('Settings loaded', SUCCESS)
        except (OSError, ValueError, TypeError) as exc:
            _set_status(f'Cannot load settings: {exc}', ERROR)


def on_training_preset(sender: ItemId = 0, app_data: Any = None, user_data: str = 'quick') -> None:
    quick = user_data == 'quick'
    dpg.set_value('num_epochs_input', 1 if quick else 10)
    dpg.set_value('batch_size_input', 2 if quick else 16)
    dpg.set_value('freeze_backbone_checkbox', quick)
    if _text('model_version_combo') != 'mvit_v2_s':
        dpg.set_value('num_frames_input', 2 if quick else 8)
    on_config_changed()
    _set_status('Quick check preset applied' if quick else 'Standard training preset applied', SUCCESS)



# ---------------------------------------------------------------------------
# Small widget-value helpers (mirrors df-analyze app.py's _text/_flag/etc.)
# ---------------------------------------------------------------------------

def _text(tag: str) -> str:
    return str(dpg.get_value(tag) or "").strip()


def _flag(tag: str) -> bool:
    return bool(dpg.get_value(tag))


def _integer(tag: str, default: int = 0) -> int:
    try:
        return int(dpg.get_value(tag))
    except (TypeError, ValueError):
        return default


def _number(tag: str, default: float = 0.0) -> float:
    try:
        return float(dpg.get_value(tag))
    except (TypeError, ValueError):
        return default


def _lines(tag: str) -> list[str]:
    raw = dpg.get_value(tag) or ""
    parts = [p.strip() for line in raw.splitlines() for p in line.split(",")]
    return [p for p in parts if p]


def _set_status(message: str, color: tuple[int, int, int]) -> None:
    dpg.set_value("status_text", message)
    dpg.configure_item("status_text", color=color)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def on_nav_selected(sender: ItemId = 0, app_data: Any = None, user_data: str = "overview") -> None:
    if user_data != 'dataset' and _video_player is not None:
        _video_player.pause()
    for key, _ in NAV_ITEMS:
        dpg.set_value(f"nav_{key}", key == user_data)
        dpg.configure_item(f"page_{key}", show=key == user_data)
    if user_data == 'results':
        on_refresh_results()
    elif user_data == 'test':
        on_refresh_test_runs()


def on_show_dialog(sender: ItemId, app_data: Any, user_data: str) -> None:
    dpg.show_item(user_data)


# ---------------------------------------------------------------------------
# Dataset page
# ---------------------------------------------------------------------------

def on_data_folder_selected(sender: ItemId, app_data: Any) -> None:
    path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
    if path:
        dpg.set_value("data_path_input", path)
        on_config_changed()


def on_data_file_selected(sender: ItemId, app_data: Any) -> None:
    global _pending_video_paths
    path = _dialog_path(app_data)
    if path:
        dpg.set_value("data_path_input", path)
        if _text('modality_radio') == 'video' and Path(path).suffix.lower() in VIDEO_EXTENSIONS:
            selections = app_data.get('selections') or {Path(path).name: path}
            _pending_video_paths = [Path(name).expanduser().resolve() for name in selections.values()
                                    if Path(name).suffix.lower() in VIDEO_EXTENSIONS]
            missing = [name for name in _pending_video_paths if not name.is_file()]
            if missing:
                _set_video_import_status(f'Could not import: file not found: {missing[0]}', ERROR)
                _pending_video_paths = []
                _update_video_selection()
                return
            # Import is separate from labeling: a chosen file must appear in the
            # list immediately, including when the user has not entered a class.
            for name in _pending_video_paths:
                _video_clips.setdefault(str(name), '')
            dpg.set_value('video_label_input', '')
            dpg.set_value('raw_data_checkbox', True)
            on_raw_data_toggled()
            _update_video_selection()
            _render_video_clips()
            dpg.set_value('video_decode_status', 'Reading a frame from the selected clip...')
            on_load_preview()
        else:
            _pending_video_paths = []
            _update_video_selection()
        on_config_changed()


def _set_video_import_status(message: str, color: tuple[int, int, int]) -> None:
    dpg.set_value('video_import_status', message)
    dpg.configure_item('video_import_status', color=color)
    _set_status(message, color)


def _update_video_selection() -> None:
    if _pending_video_paths:
        names = ', '.join(path.name for path in _pending_video_paths[:3])
        if len(_pending_video_paths) > 3:
            names += f' and {len(_pending_video_paths) - 3} more'
        message = f'{len(_pending_video_paths)} selected: {names}'
    else:
        message = 'No clips selected. Choose MP4 files to import them.'
    dpg.set_value('video_selection_text', message)
    dpg.configure_item('apply_video_label_button', enabled=bool(_pending_video_paths))


def _update_video_clip_summary() -> None:
    missing = sum(not label.strip() for label in _video_clips.values())
    classes = len({label for label in _video_clips.values() if label.strip()})
    dpg.set_value('video_clips_count', f'{len(_video_clips)} clips / {classes} classes / {missing} unlabeled')
    dpg.configure_item('use_video_dataset_button', enabled=bool(_video_clips) and not missing)
    if missing:
        message = ('1 clip needs a label' if missing == 1 else f'{missing} clips need labels')
        _set_video_import_status(f'{message} before training. Type a label beside each file, or apply one to the selected clips.', WARNING)
    elif _video_clips:
        issues = classification_split_issues(pd.Series(list(_video_clips.values())).value_counts().to_dict(), unit='clips')
        _set_video_import_status(' '.join(issues) if issues else 'All clips labeled. Click Use clip dataset to save this list.',
                                 WARNING if issues else SUCCESS)
    else:
        _set_video_import_status('Choose video files. They will appear in the list immediately.', TEXT_MUTED)


def on_add_video_clips(sender: ItemId = 0, app_data: Any = None) -> None:
    label = _text('video_label_input')
    if not _pending_video_paths:
        _set_video_import_status('Choose video files first, then apply a label to them.', WARNING)
        return
    if not label:
        _set_video_import_status('Enter a label such as walking, then click Apply label to selected. Your clips are already imported.', WARNING)
        dpg.focus_item('video_label_input')
        return
    for path in _pending_video_paths:
        if not path.is_file():
            _set_video_import_status(f'Video not found: {path.name}', ERROR)
            return
        existing = _video_clips.get(str(path))
        if existing and existing != label:
            _set_video_import_status(f'{path.name} is already labeled {existing}. Edit its label in the list to change it.', WARNING)
            return
    _video_clips.update({str(path): label for path in _pending_video_paths})
    _render_video_clips()


def on_video_clip_label_changed(sender: ItemId = 0, app_data: Any = None, user_data: str = '') -> None:
    if user_data in _video_clips:
        _video_clips[user_data] = str(app_data or '').strip()
        # Do not rebuild text inputs during typing; doing so discards focus.
        _update_video_clip_summary()


def on_remove_video_clip(sender: ItemId = 0, app_data: Any = None, user_data: str = '') -> None:
    global _pending_video_paths
    _video_clips.pop(user_data, None)
    _pending_video_paths = [path for path in _pending_video_paths if str(path) != user_data]
    _update_video_selection()
    _render_video_clips()


def on_clear_video_clips(sender: ItemId = 0, app_data: Any = None) -> None:
    global _pending_video_paths
    _video_clips.clear()
    _pending_video_paths = []
    dpg.set_value('video_decode_status', '')
    _update_video_selection()
    _render_video_clips()


def _render_video_clips() -> None:
    dpg.delete_item('video_clips_container', children_only=True)
    _update_video_clip_summary()
    if not _video_clips:
        dpg.add_text('Select MP4 files to import them here. Add class labels when you are ready to train.', parent='video_clips_container', color=TEXT_MUTED, wrap=750)
        return
    with dpg.table(parent='video_clips_container', header_row=True, resizable=True, scrollY=True, height=150):
        for name in ('Video file', 'Class label', 'Action'):
            dpg.add_table_column(label=name)
        for name, label in _video_clips.items():
            with dpg.table_row():
                dpg.add_text(Path(name).name)
                with ui.tooltip(dpg.last_item()):
                    dpg.add_text(name, wrap=700)
                dpg.add_input_text(default_value=label, hint='Label required for training', width=-1,
                                   callback=on_video_clip_label_changed, user_data=name)
                with ui.group(horizontal=True):
                    dpg.add_button(label='Preview', callback=on_preview_video_clip, user_data=name)
                    dpg.add_button(label='Remove', callback=on_remove_video_clip, user_data=name)


def on_use_video_clips(sender: ItemId = 0, app_data: Any = None) -> None:
    if not _video_clips:
        _set_video_import_status('Choose video files first. They will appear in the list immediately.', WARNING)
        return
    if any(not label.strip() for label in _video_clips.values()):
        _update_video_clip_summary()
        return
    try:
        directory = SESSION_PATH.parent / 'datasets'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'video-clips-{time.time_ns()}.csv'
        pd.DataFrame(list(_video_clips.items()), columns=['videos', 'labels']).to_csv(destination, index=False)
        dpg.set_value('data_path_input', str(destination))
        dpg.set_value('raw_data_checkbox', True)
        dpg.set_value('mode_radio', 'cls')
        dpg.set_value('target_input', 'labels')
        dpg.set_value('num_classes_input', len(set(_video_clips.values())))
        on_raw_data_toggled()
        on_load_preview()
        issues = classification_split_issues(pd.Series(list(_video_clips.values())).value_counts().to_dict(), unit='clips')
        _set_video_import_status('Clip dataset saved. ' + (' '.join(issues) if issues else 'Choose model and output folder to train.'),
                                 WARNING if issues else SUCCESS)
    except OSError as exc:
        _set_video_import_status(f'Could not save the clip list: {exc}', ERROR)


def on_outdir_selected(sender: ItemId, app_data: Any) -> None:
    path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
    if path:
        dpg.set_value("outdir_input", path)
        on_config_changed()


def _render_dataframe_preview(df: pd.DataFrame, container: str, limit: int = 50) -> None:
    dpg.delete_item(container, children_only=True)
    columns = list(df.columns)[:40]
    with dpg.table(parent=container, header_row=True, resizable=True,
                    scrollX=True, scrollY=True, borders_innerV=True, borders_outerV=True,
                    height=280):
        for col in columns:
            dpg.add_table_column(label=str(col))
        for _, row in df.head(limit).iterrows():
            with dpg.table_row():
                for col in columns:
                    dpg.add_text(str(row[col])[:80])


def on_load_preview(sender: ItemId = 0, app_data: Any = None, user_data: str = '') -> None:
    global _preview_future, _preview_config, _preview_source_config, _preview_video_clips_snapshot
    global _scroll_to_video_preview
    source = gc.resolved_config(_build_run_config())
    cfg = replace(source, df=Path(user_data).expanduser().resolve()) if user_data else source
    stop_video_playback()
    if cfg.df is None or not cfg.df.exists():
        dpg.set_value('preview_note', 'Choose an existing dataset path first.')
        return
    if _preview_future is not None:
        _preview_future.cancel()
    _preview_config = cfg
    _scroll_to_video_preview = 8 if cfg.modality == 'video' and cfg.df.suffix.lower() in VIDEO_EXTENSIONS else 0
    _preview_source_config = source
    _preview_video_clips_snapshot = _video_clips.copy()
    _preview_future = _preview_pool.submit(preview_dataset, cfg)
    dpg.configure_item('preview_button', enabled=False)
    dpg.set_value('preview_note', 'Reading dataset preview...')


def poll_preview() -> None:
    global _preview_future
    if _preview_future is None or not _preview_future.done():
        return
    future, _preview_future = _preview_future, None
    dpg.configure_item('preview_button', enabled=True)
    try:
        result = future.result()
        current = gc.resolved_config(_build_run_config())
        source = _preview_source_config or _preview_config
        if any(getattr(current, key) != getattr(source, key) for key in ('df', 'modality', 'raw_data', 'target', 'grouper')):
            dpg.set_value('preview_note', 'Settings changed. Load a fresh preview.')
            return
        dpg.set_value('preview_note', result.note + ('\n' + '\n'.join(result.warnings) if result.warnings else ''))
        dpg.delete_item('preview_container', children_only=True)
        if result.thumbnail is not None:
            width, height, pixels = result.thumbnail
            _show_video_frame(width, height, pixels)
            start_video_playback(_preview_config.df)
            dpg.set_value('video_decode_status', f'Video loaded: {result.note}. Playback controls are in the preview below.')
            dpg.configure_item('video_decode_status', color=SUCCESS)
        elif result.frame is not None:
            _render_dataframe_preview(result.frame, 'preview_container')
        if result.class_count is not None:
            dpg.set_value('num_classes_input', result.class_count)
        if result.video_entries is not None and _video_clips == _preview_video_clips_snapshot:
            _video_clips.clear()
            _video_clips.update(result.video_entries)
            _render_video_clips()
        _set_status('Dataset preview ready', SUCCESS if not result.warnings else WARNING)
    except Exception as exc:
        dpg.set_value('preview_note', f'Could not read dataset: {exc}')
        dpg.delete_item('preview_container', children_only=True)
        if _preview_config.modality == 'video':
            dpg.set_value('video_decode_status', f'Could not preview this video: {exc}')
            dpg.configure_item('video_decode_status', color=ERROR)
        _set_status('Could not read dataset', ERROR)
    on_config_changed()


def _show_video_frame(width: int, height: int, pixels: Any) -> None:
    global _video_texture_size
    if _video_texture_size != (width, height) or not dpg.does_item_exist('video_preview_texture'):
        if dpg.does_item_exist('video_preview_image'):
            dpg.delete_item('video_preview_image')
        if dpg.does_item_exist('video_preview_texture'):
            dpg.delete_item('video_preview_texture')
        dpg.add_dynamic_texture(width, height, pixels, tag='video_preview_texture', parent='preview_textures')
        _video_texture_size = (width, height)
    else:
        dpg.set_value('video_preview_texture', pixels)
    if not dpg.does_item_exist('video_preview_image'):
        dpg.add_image('video_preview_texture', tag='video_preview_image', parent='preview_container')


def stop_video_playback(timeout: float = 0.0) -> None:
    global _video_player, _video_player_source, _video_texture_size
    global _scroll_to_video_preview
    if _video_player is not None:
        _video_player.close(timeout=timeout)
        _video_player = None
    _video_player_source = None
    if dpg.does_item_exist('video_preview_controls'):
        dpg.configure_item('video_preview_controls', show=False)
        dpg.configure_item('preview_container', height=320)
    if dpg.does_item_exist('video_preview_image'):
        dpg.delete_item('video_preview_image')
    if dpg.does_item_exist('video_preview_texture'):
        dpg.delete_item('video_preview_texture')
    _video_texture_size = None
    _scroll_to_video_preview = 0


def start_video_playback(path: Path) -> None:
    global _video_player, _video_player_source
    if _video_player is not None:
        _video_player.close(timeout=0)
    _video_player = VideoPlayer(path, autoplay=_flag('nav_dataset'), loop=_flag('video_loop_checkbox'))
    _video_player_source = gc.resolved_config(_build_run_config()).df
    dpg.configure_item('video_preview_controls', show=True)
    dpg.configure_item('preview_container', height=400)
    dpg.configure_item('video_play_button', label='Pause', enabled=False)
    dpg.set_value('video_playback_status', 'Starting video playback...')


def _video_time(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f'{minutes}:{seconds:02d}'


def poll_video_playback() -> None:
    global _scroll_to_video_preview
    if _video_player is None:
        return
    cfg = gc.resolved_config(_build_run_config())
    if cfg.modality != 'video' or cfg.df != _video_player_source:
        stop_video_playback()
        return
    state, frame = _video_player.poll()
    if frame is not None:
        _show_video_frame(frame.width, frame.height, frame.pixels)
    ready = state.ready and not state.error
    if ready and _scroll_to_video_preview and _flag('nav_dataset'):
        card = dpg.get_item_state('dataset_preview_card').get('pos')
        if card:
            # Offscreen child windows acquire their final size only after they
            # become visible. Allow a few render frames for that layout to settle.
            target = dpg.get_y_scroll('workspace_content') + card[1]
            dpg.set_y_scroll('workspace_content', min(dpg.get_y_scroll_max('workspace_content'), max(0, target)))
            _scroll_to_video_preview -= 1
    dpg.configure_item('video_play_button', label='Pause' if state.playing else 'Play', enabled=ready)
    dpg.configure_item('video_restart_button', enabled=ready)
    dpg.configure_item('video_seek_slider', enabled=ready and state.duration > 0,
                       max_value=max(state.duration, 0.01))
    if not dpg.is_item_active('video_seek_slider'):
        dpg.set_value('video_seek_slider', state.position)
    total = _video_time(state.duration) if state.duration else '--:--'
    dpg.set_value('video_time_text', f'{_video_time(state.position)} / {total}')
    status = ('Playback error: ' + state.error if state.error else
              'Finished - press Play to replay.' if state.ended else
              f'{"Playing" if state.playing else "Paused"} | frame {state.frame_index + 1}' if state.ready else
              'Loading video...')
    dpg.set_value('video_playback_status', status)
    dpg.configure_item('video_playback_status', color=ERROR if state.error else TEXT_MUTED)
    if state.error:
        dpg.set_value('video_decode_status', status)
        dpg.configure_item('video_decode_status', color=ERROR)


def on_video_play_pause(sender: ItemId = 0, app_data: Any = None) -> None:
    if _video_player is not None:
        _video_player.pause() if _video_player.state.playing else _video_player.play()


def on_video_restart(sender: ItemId = 0, app_data: Any = None) -> None:
    if _video_player is not None:
        _video_player.seek(0)
        _video_player.play()


def on_video_seek(sender: ItemId = 0, app_data: Any = None) -> None:
    if _video_player is not None:
        _video_player.seek(float(app_data))


def on_video_loop_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    if _video_player is not None:
        _video_player.set_loop(bool(app_data))


def on_preview_video_clip(sender: ItemId = 0, app_data: Any = None, user_data: str = '') -> None:
    on_load_preview(user_data=user_data)


def on_open_video_player(sender: ItemId = 0, app_data: Any = None) -> None:
    if _video_player is None:
        return
    try:
        _video_player.pause()
        path = str(_video_player.path.resolve())
        if hasattr(os, 'startfile'):
            os.startfile(path)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', path])
    except OSError as exc:
        dpg.set_value('video_playback_status', f'Could not open the video player: {exc}')
        _set_video_import_status(f'Could not open the video player: {exc}', ERROR)


def on_raw_data_toggled(sender: ItemId = 0, app_data: Any = None) -> None:
    folder = _flag("raw_data_checkbox") and _text("modality_radio") in ("images", "video")
    dpg.configure_item("browse_folder_button", show=folder)
    video = _text('modality_radio') == 'video'
    dpg.configure_item("browse_file_button", show=not folder or video)
    dpg.configure_item('video_import_group', show=video and _flag('raw_data_checkbox'))
    dpg.configure_item('data_path_input', width=-400 if video and folder else -280)
    on_config_changed()


# ---------------------------------------------------------------------------
# Model & training page
# ---------------------------------------------------------------------------

def on_modality_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    modality = _text("modality_radio")
    if modality != 'video':
        stop_video_playback()
    versions = gc.MODEL_VERSIONS_BY_MODALITY[modality]
    dpg.configure_item("model_version_combo", items=versions)
    if dpg.get_value("model_version_combo") not in versions:
        dpg.set_value("model_version_combo", versions[0])

    show_transfer = modality in gc.NEEDS_TRANSFER_LEARNING_OPTIONS
    dpg.configure_item("transfer_learning_group", show=show_transfer)
    dpg.configure_item("num_classes_group", show=modality in gc.NEEDS_NUM_CLASSES or modality == "tabular")
    dpg.configure_item("num_classes_input", show=modality in gc.NEEDS_NUM_CLASSES)
    dpg.configure_item("video_options_group", show=modality == "video")
    dpg.configure_item("timeseries_options_group", show=modality == "timeseries")

    on_model_version_changed()
    _update_data_source_hint()
    on_raw_data_toggled()
    on_config_changed()


def on_model_version_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    modality = _text("modality_radio")
    version = _text("model_version_combo")
    supports_peft = modality in gc.NEEDS_TRANSFER_LEARNING_OPTIONS and version != 'gpt2'
    dpg.configure_item('use_peft_checkbox', enabled=supports_peft)
    if not supports_peft:
        dpg.set_value('use_peft_checkbox', False)
    dpg.configure_item('mode_radio', show=modality in ('images', 'video') or version == 'tabpfn')
    if modality == 'text' or version == 'siglip':
        dpg.set_value('mode_radio', 'cls')
    dpg.configure_item('added_layers_combo', enabled=version != 'gpt2')
    dpg.configure_item('embed_size_input', enabled=version != 'gpt2')
    if modality not in gc.NEEDS_TRANSFER_LEARNING_OPTIONS:
        dpg.set_value('freeze_backbone_checkbox', False)
    if version == 'mvit_v2_s':
        dpg.set_value('num_frames_input', 16)
    dpg.configure_item('pooling_radio', enabled=version not in gc.VIDEO_NATIVE_MODEL_VERSIONS)

    dpg.configure_item("gandalf_group", show=(modality == "tabular" and version == "gandalf"))
    dpg.configure_item("tabpfn_group", show=(modality == "tabular" and version == "tabpfn"))
    on_config_changed()


def _update_data_source_hint() -> None:
    modality = _text("modality_radio")
    if modality == 'video':
        hint = 'Open MP4 files to preview and label clips below, select a labeled class folder, or load an existing video list / parquet dataset.'
    elif modality == 'images':
        hint = ("Raw folder: one subdirectory per class (optionally grouped under split "
                "subdirectories), each holding the raw files. Or point at an existing parquet.")
    else:
        hint = "Raw: a single CSV or XLSX file. Or point at an existing parquet."
    dpg.set_value("data_source_hint", hint)


def _active_feat_lists() -> tuple[list[str], list[str]]:
    return _lines("continuous_cols_input"), _lines("categorical_cols_input")


def _build_run_config() -> gc.RunConfig:
    modality = _text("modality_radio")
    path_text = _text("data_path_input")
    out_text = _text("outdir_input")
    continuous_cols, categorical_cols = _active_feat_lists()
    return gc.RunConfig(
        modality=modality,
        df=Path(path_text) if path_text else None,
        raw_data=_flag("raw_data_checkbox"),
        out=Path(out_text) if out_text else None,
        target=_text("target_input") or "labels",
        grouper=_text("grouper_input"),
        model_version=_text("model_version_combo"),
        num_classes=_integer("num_classes_input", 2),
        mode=_text("mode_radio"),
        added_layers=_integer("added_layers_combo", 2),
        embed_size=_integer("embed_size_input", 1000),
        freeze_backbone=_flag("freeze_backbone_checkbox"),
        use_peft=_flag("use_peft_checkbox"),
        fixed_seed=_flag("fixed_seed_checkbox"),
        batch_size=_integer("batch_size_input", 16),
        num_epochs=_integer("num_epochs_input", 10),
        learning_rate=_number("learning_rate_input", 1e-4),
        num_frames=_integer("num_frames_input", 8),
        pooling=_text("pooling_radio") or "mean",
        gandalf_type=_text("gandalf_type_radio") or "classification",
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        finetuning_mode=_flag("finetuning_mode_checkbox"),
        time_idx_column=_text("time_idx_column_input") or "labels",
    )


def on_config_changed(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        cfg = gc.resolved_config(_build_run_config())
        issues = gc.validate_config(cfg)
        if issues:
            dpg.set_value("validation_text", "\n".join(issues))
            dpg.configure_item("validation_text", color=WARNING)
        else:
            dpg.set_value("validation_text", "Ready to run.")
            dpg.configure_item("validation_text", color=SUCCESS)
        dpg.configure_item("run_button", enabled=not issues and _current_run is None)
        argv = gc.build_argv(gc.resolved_config(cfg)) if not issues else []
        if argv:
            dpg.set_value("command_text", subprocess.list2cmdline([sys.executable, str(ROOT / "deeptune.py"), *argv]))
        else:
            dpg.set_value("command_text", "")
    except Exception as exc:
        dpg.set_value("validation_text", str(exc))
        dpg.configure_item("validation_text", color=ERROR)
        dpg.configure_item('run_button', enabled=False)


def on_copy_command(sender: ItemId = 0, app_data: Any = None) -> None:
    dpg.set_clipboard_text(_text("command_text"))
    _set_status("Command copied", SUCCESS)


# ---------------------------------------------------------------------------
# Run page
# ---------------------------------------------------------------------------

def on_run_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    global _current_run, _run_outdir, _run_started
    if _current_run is not None:
        return
    _line_buffer.clear()
    dpg.set_value("log_box", "")
    try:
        cfg = gc.resolved_config(_build_run_config())
        issues = gc.validate_config(cfg)
        if issues:
            raise ValueError(issues[0])
        cfg.out.mkdir(parents=True, exist_ok=True)
        gc.save_config(cfg, SESSION_PATH)
        handle = start_run(cfg)
    except Exception as exc:
        _set_status(str(exc), ERROR)
        dpg.set_value("validation_text", str(exc))
        dpg.configure_item('run_button', enabled=False)
        return
    _current_run = handle
    _run_outdir = cfg.out
    _run_started = time.monotonic()
    dpg.set_value("run_stage_text", "Starting...")
    _set_status("Run in progress", ACCENT_LIGHT)
    dpg.configure_item("run_button", enabled=False)
    dpg.configure_item("cancel_button", enabled=handle.can_cancel)
    dpg.configure_item("run_loading_indicator", show=True)
    on_nav_selected(user_data="run")


def on_cancel_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    if _current_run is not None and _current_run.cancel():
        _set_status("Stopping run...", WARNING)
        dpg.configure_item("cancel_button", enabled=False)


def poll_run_state() -> None:
    global _current_run
    if _current_run is None:
        return
    seconds = int(time.monotonic() - _run_started)
    dpg.set_value("elapsed_text", f"Elapsed  {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
    changed = False
    for _ in range(300):
        try:
            chunk = _current_run.log_queue.get_nowait()
        except queue.Empty:
            break
        _line_buffer.feed(chunk)
        changed = True
    if changed:
        content = _line_buffer.render()
        dpg.set_value("log_box", content)
        if _flag("follow_log_checkbox"):
            dpg.set_y_scroll("log_window", dpg.get_y_scroll_max("log_window"))
        stage = next((line.strip()[:120] for line in reversed(content.splitlines()) if line.strip()), "Running")
        dpg.set_value("run_stage_text", stage)
    try:
        status, message = _current_run.done_queue.get_nowait()
    except queue.Empty:
        return
    while True:
        try:
            _line_buffer.feed(_current_run.log_queue.get_nowait())
        except queue.Empty:
            break
    dpg.set_value("log_box", _line_buffer.render())
    dpg.configure_item("run_button", enabled=True)
    dpg.configure_item("cancel_button", enabled=False)
    dpg.configure_item("run_loading_indicator", show=False)
    if status == "success":
        _set_status("Run complete", SUCCESS)
        dpg.set_value("run_stage_text", "Complete. Open the Results page.")
        if _run_outdir is not None:
            refresh_results(_run_outdir)
    elif status == "cancelled":
        _set_status("Run stopped", WARNING)
        dpg.set_value("run_stage_text", "Stopped. Partial output may remain in the output folder.")
    else:
        summary = message.splitlines()[0] if message else "Unknown failure"
        _set_status("Run failed - check log", ERROR)
        dpg.set_value("run_stage_text", summary)
        _line_buffer.feed(f"\n[deeptune] {message}\n")
        dpg.set_value("log_box", _line_buffer.render())
    _current_run = None
    on_config_changed()


def on_save_log(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        outdir = _run_outdir or Path(_text("outdir_input")).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"desktop-log-{time.time_ns()}.txt"
        path.write_text(_line_buffer.render(), encoding="utf-8")
        _set_status(f"Saved: {path.name}", SUCCESS)
    except OSError as exc:
        _set_status(f"Cannot save log: {exc}", ERROR)


# ---------------------------------------------------------------------------
# Results page
# ---------------------------------------------------------------------------

def on_refresh_results(sender: ItemId = 0, app_data: Any = None) -> None:
    out_text = _text("outdir_input")
    if out_text:
        refresh_results(Path(out_text).expanduser())


def refresh_results(out_dir: Path) -> None:
    global _run_history
    try:
        runs = gc.list_runs(out_dir)
    except OSError as exc:
        _clear_results(f'Cannot read output folder: {exc}')
        return
    _run_history = {f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}  |  {p.name}": p for p in runs}
    dpg.configure_item("results_run_combo", items=list(_run_history.keys()))
    dpg.set_value("results_count_text", f"{len(runs)} run(s) found under {out_dir}")
    if _run_history:
        first = next(iter(_run_history))
        dpg.set_value("results_run_combo", first)
        on_results_run_selected(user_data=first)
    else:
        dpg.set_value("results_run_combo", "")
        _clear_results("No runs found yet. Choose an output folder or start your first run.")


def on_results_run_selected(sender: ItemId = 0, app_data: Any = None, user_data: str | None = None) -> None:
    key = user_data or _text("results_run_combo")
    run_dir = _run_history.get(key)
    if run_dir is None:
        return
    try:
        render_results(gc.read_run_results(run_dir))
    except (OSError, ValueError) as exc:
        _clear_results(f'Cannot read this run: {exc}')


def _clear_results(message: str) -> None:
    global _selected_results
    _selected_results = None
    dpg.delete_item("results_metrics_container", children_only=True)
    dpg.add_text(message, parent="results_metrics_container", color=TEXT_MUTED)
    dpg.delete_item("results_paths_container", children_only=True)
    dpg.set_value("results_series_loss_plot", [[], []])
    dpg.set_value("results_series_valloss_plot", [[], []])


def render_results(results: gc.RunResults) -> None:
    global _selected_results
    _selected_results = results
    dpg.delete_item('results_metrics_container', children_only=True)
    if results.run_status:
        dpg.add_text(f"Status: {results.run_status.get('status', 'unknown')}", parent='results_metrics_container', color=ACCENT_LIGHT)
    if results.metrics:
        with dpg.table(parent='results_metrics_container', header_row=True, borders_innerV=True, resizable=True):
            dpg.add_table_column(label='Metric / Class')
            dpg.add_table_column(label='Value (as reported by the model)')
            for name, value in gc.metric_rows(results.metrics):
                with dpg.table_row():
                    dpg.add_text(name)
                    dpg.add_text(value)
    else:
        dpg.add_text('Evaluation metrics are not available for this run yet.', parent='results_metrics_container', color=TEXT_MUTED)
    for warning in results.warnings:
        dpg.add_text(warning, parent='results_metrics_container', color=WARNING, wrap=800)
    dpg.set_value('results_series_loss_plot', gc.loss_series(results.training_log, 'epoch_loss', 'train_loss', 'loss'))
    dpg.set_value('results_series_valloss_plot', gc.loss_series(results.training_log, 'val_loss', 'validation_loss'))
    if results.training_log is not None:
        dpg.fit_axis_data('results_x_axis')
        dpg.fit_axis_data('results_y_axis')
    dpg.delete_item("results_paths_container", children_only=True)
    path_rows = [
        ("Experiment folder", results.run_dir),
        ("Checkpoint", results.checkpoint_path),
        ("Embeddings file", results.embeddings_path),
    ]
    for label, value in path_rows:
        if value is not None:
            dpg.add_text(f"{label}:  {value}", parent="results_paths_container", wrap=900, color=TEXT_MUTED)
    if results.embeddings_shape is not None:
        dpg.add_text(f"Embeddings matrix shape: {results.embeddings_shape}", parent="results_paths_container", color=ACCENT_LIGHT)


def on_open_output(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        path = _selected_results.run_dir if _selected_results else Path(_text('outdir_input')).expanduser()
        if not path.is_dir():
            raise ValueError('Choose an existing output folder first.')
        if hasattr(os, 'startfile'):
            os.startfile(str(path.resolve()))
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(path.resolve())])
    except (OSError, ValueError) as exc:
        _set_status(str(exc), ERROR)


def on_export_metrics(sender: ItemId = 0, app_data: Any = None) -> None:
    if _selected_results is None or not _selected_results.metrics:
        _set_status('Select a run with evaluation metrics first.', WARNING)
        return
    try:
        path = gc.export_report(_selected_results)
        _set_status(f'Exported {path.name}', SUCCESS)
    except OSError as exc:
        _set_status(f'Cannot export metrics: {exc}', ERROR)


# ---------------------------------------------------------------------------
# Test page: run an already-trained run's checkpoint on new data (infer.py),
# mirroring the Run/Results pages' picking-a-run/launching/streaming/
# reading-results shape but for prediction instead of training.
# ---------------------------------------------------------------------------

def on_refresh_test_runs(sender: ItemId = 0, app_data: Any = None) -> None:
    global _test_run_history
    out_text = _text('outdir_input')
    if not out_text:
        dpg.set_value('test_run_info_text', 'Choose an output folder on the Run page first.')
        return
    try:
        runs = gc.list_runs(Path(out_text).expanduser())
    except OSError as exc:
        _set_status(f'Cannot read output folder: {exc}', ERROR)
        return
    _test_run_history = {f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}  |  {p.name}": p
                         for p in runs}
    dpg.configure_item('test_run_combo', items=list(_test_run_history.keys()))
    if _test_run_history:
        first = next(iter(_test_run_history))
        dpg.set_value('test_run_combo', first)
        on_test_run_selected(user_data=first)
    else:
        dpg.set_value('test_run_combo', '')
        dpg.set_value('test_run_info_text', 'No completed runs found under this output folder yet.')
        dpg.configure_item('test_run_button', enabled=False)
        _clear_test_results('No completed runs found yet.')


def on_test_run_selected(sender: ItemId = 0, app_data: Any = None, user_data: str | None = None) -> None:
    global _selected_test_run
    key = user_data or _text('test_run_combo')
    run_dir = _test_run_history.get(key)
    if run_dir is None:
        return
    try:
        results = gc.read_run_results(run_dir)
    except (OSError, ValueError) as exc:
        _set_status(f'Cannot read this run: {exc}', ERROR)
        return
    _selected_test_run = results
    # Whatever predictions/metrics are on screen belong to whichever run was
    # selected (or tested) before this one -- clear them so switching runs
    # never leaves a stale, mismatched result displayed.
    if results.checkpoint_path is None:
        dpg.set_value('test_run_info_text', 'This run has no saved checkpoint yet (it may still be training).')
        dpg.configure_item('test_run_button', enabled=False)
        _clear_test_results('This run has no saved checkpoint to test yet.')
        return
    cli_args = results.cli_arguments or {}
    modality = cli_args.get('modality', 'unknown')
    model_version = cli_args.get('model_version') or '?'
    mode = cli_args.get('mode', 'cls')
    info = f"{modality}  |  {model_version}  |  {mode}  |  checkpoint: {results.checkpoint_path.name}"
    if modality == 'timeseries':
        info += f"  |  history splits found: {len(gc.find_history_paths(run_dir))}"
    dpg.set_value('test_run_info_text', info)
    dpg.configure_item('test_run_button', enabled=True)
    _clear_test_results('Choose new data and click Test model to see predictions for this run.')


def on_test_data_folder_selected(sender: ItemId, app_data: Any) -> None:
    path = app_data.get('file_path_name') if isinstance(app_data, dict) else None
    if path:
        dpg.set_value('test_data_path_input', path)


def on_test_data_file_selected(sender: ItemId, app_data: Any) -> None:
    path = _dialog_path(app_data)
    if path:
        dpg.set_value('test_data_path_input', path)


def on_run_test_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    global _current_infer_run, _infer_out_dir, _infer_started
    if _current_infer_run is not None:
        return
    if _selected_test_run is None or _selected_test_run.checkpoint_path is None:
        _set_status('Choose a trained run with a saved checkpoint first.', WARNING)
        return
    data_text = _text('test_data_path_input')
    if not data_text:
        _set_status('Choose new data to test against.', WARNING)
        return
    _infer_line_buffer.clear()
    dpg.set_value('test_log_box', '')
    try:
        out_dir = _selected_test_run.run_dir / 'desktop-inference' / time.strftime('%Y%m%d-%H%M%S')
        cfg = gc.infer_config_from_run(_selected_test_run, data=Path(data_text).expanduser(), out=out_dir)
        issues = gc.validate_infer_config(cfg)
        if issues:
            raise ValueError(issues[0])
        handle = start_infer_run(cfg)
    except Exception as exc:
        _set_status(str(exc), ERROR)
        dpg.set_value('test_stage_text', str(exc))
        return
    _current_infer_run = handle
    _infer_out_dir = cfg.out
    _infer_started = time.monotonic()
    dpg.set_value('test_stage_text', 'Starting...')
    _set_status('Test run in progress', ACCENT_LIGHT)
    dpg.configure_item('test_run_button', enabled=False)
    dpg.configure_item('test_cancel_button', enabled=handle.can_cancel)
    dpg.configure_item('test_loading_indicator', show=True)


def on_cancel_test_clicked(sender: ItemId = 0, app_data: Any = None) -> None:
    if _current_infer_run is not None and _current_infer_run.cancel():
        _set_status('Stopping test run...', WARNING)
        dpg.configure_item('test_cancel_button', enabled=False)


def poll_infer_run_state() -> None:
    global _current_infer_run
    if _current_infer_run is None:
        return
    seconds = int(time.monotonic() - _infer_started)
    dpg.set_value('test_elapsed_text', f"Elapsed  {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
    changed = False
    for _ in range(300):
        try:
            chunk = _current_infer_run.log_queue.get_nowait()
        except queue.Empty:
            break
        _infer_line_buffer.feed(chunk)
        changed = True
    if changed:
        content = _infer_line_buffer.render()
        dpg.set_value('test_log_box', content)
        if _flag('test_follow_log_checkbox'):
            dpg.set_y_scroll('test_log_window', dpg.get_y_scroll_max('test_log_window'))
        stage = next((line.strip()[:120] for line in reversed(content.splitlines()) if line.strip()), 'Running')
        dpg.set_value('test_stage_text', stage)
    try:
        status, message = _current_infer_run.done_queue.get_nowait()
    except queue.Empty:
        return
    while True:
        try:
            _infer_line_buffer.feed(_current_infer_run.log_queue.get_nowait())
        except queue.Empty:
            break
    dpg.set_value('test_log_box', _infer_line_buffer.render())
    dpg.configure_item('test_run_button', enabled=True)
    dpg.configure_item('test_cancel_button', enabled=False)
    dpg.configure_item('test_loading_indicator', show=False)
    if status == 'success':
        _set_status('Test run complete', SUCCESS)
        dpg.set_value('test_stage_text', 'Complete. See predictions below.')
        if _infer_out_dir is not None:
            render_test_results(gc.read_inference_results(_infer_out_dir))
    elif status == 'cancelled':
        _set_status('Test run stopped', WARNING)
        dpg.set_value('test_stage_text', 'Stopped. Partial output may remain in the output folder.')
    else:
        summary = message.splitlines()[0] if message else 'Unknown failure'
        _set_status('Test run failed - check log', ERROR)
        dpg.set_value('test_stage_text', summary)
        _infer_line_buffer.feed(f"\n[infer] {message}\n")
        dpg.set_value('test_log_box', _infer_line_buffer.render())
    _current_infer_run = None


def _clear_test_results(message: str) -> None:
    global _selected_test_results
    _selected_test_results = None
    dpg.delete_item('test_metrics_container', children_only=True)
    dpg.add_text(message, parent='test_metrics_container', color=TEXT_MUTED)
    dpg.delete_item('test_predictions_container', children_only=True)


def render_test_results(results: gc.InferenceResults) -> None:
    global _selected_test_results
    _selected_test_results = results
    dpg.delete_item('test_metrics_container', children_only=True)
    if results.metrics:
        with dpg.table(parent='test_metrics_container', header_row=True, borders_innerV=True, resizable=True):
            dpg.add_table_column(label='Metric')
            dpg.add_table_column(label='Value')
            for key, value in results.metrics.items():
                with dpg.table_row():
                    dpg.add_text(str(key))
                    dpg.add_text(f'{value:.4f}' if isinstance(value, float) else str(value))
    else:
        dpg.add_text('No ground truth in the new data -- predictions only, no accuracy metrics.',
                     parent='test_metrics_container', color=TEXT_MUTED)
    for warning in results.warnings:
        dpg.add_text(warning, parent='test_metrics_container', color=WARNING, wrap=800)

    dpg.delete_item('test_predictions_container', children_only=True)
    if results.predictions is not None and not results.predictions.empty:
        preview = results.predictions.head(200)
        with dpg.table(parent='test_predictions_container', header_row=True, borders_innerV=True,
                       resizable=True, scrollY=True, height=-1):
            for col in preview.columns:
                dpg.add_table_column(label=col)
            for _, row in preview.iterrows():
                with dpg.table_row():
                    for col in preview.columns:
                        value = row[col]
                        dpg.add_text('' if pd.isna(value) else str(value))
        if len(results.predictions) > len(preview):
            dpg.add_text(f'Showing the first {len(preview)} of {len(results.predictions)} predictions. '
                        f'Open the output folder for the full predictions.csv.',
                        parent='test_predictions_container', color=TEXT_MUTED)
    else:
        dpg.add_text('Run a test to see predictions here.', parent='test_predictions_container', color=TEXT_MUTED)


def on_open_test_output(sender: ItemId = 0, app_data: Any = None) -> None:
    try:
        path = _selected_test_results.out_dir if _selected_test_results else _infer_out_dir
        if path is None or not Path(path).is_dir():
            raise ValueError('Run a test first.')
        if hasattr(os, 'startfile'):
            os.startfile(str(Path(path).resolve()))
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(Path(path).resolve())])
    except (OSError, ValueError) as exc:
        _set_status(str(exc), ERROR)


# ---------------------------------------------------------------------------
# Reusable UI building blocks (same visual language as df-analyze's cards)
# ---------------------------------------------------------------------------

import contextlib


@contextlib.contextmanager
def _card(title: str, **kwargs: Any) -> Iterator[ItemId]:
    with ui.child_window(auto_resize_y=True, **kwargs) as card_id:
        with ui.group(horizontal=True):
            with ui.drawlist(width=5, height=20):
                dpg.draw_rectangle((0, 1), (5, 19), color=ACCENT, fill=ACCENT, rounding=2)
            dpg.add_spacer(width=4)
            title_id = dpg.add_text(title, color=ACCENT)
            if _FONTS.get("subheader"):
                dpg.bind_item_font(title_id, _FONTS["subheader"])
        dpg.add_separator()
        dpg.add_spacer(height=4)
        yield card_id


def _heading(title: str, subtitle: str) -> None:
    title_id = dpg.add_text(title)
    if "title" in _FONTS:
        dpg.bind_item_font(title_id, _FONTS["title"])
    dpg.add_text(subtitle, color=TEXT_MUTED, wrap=900)
    dpg.add_spacer(height=8)


def _tip(message: str) -> None:
    with ui.tooltip(dpg.last_item()):
        dpg.add_text(message, wrap=380)


def _build_theme() -> None:
    with ui.theme() as global_theme:
        with ui.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_CARD)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (*ACCENT, 90))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (*ACCENT, 140))
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TEXT_MUTED)
            dpg.add_theme_color(dpg.mvThemeCol_Border, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Button, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (*ACCENT, 80))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (*ACCENT, 120))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, BG_CARD_ALT)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, ACCENT_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_TabSelected, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BG_MAIN)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, BG_CARD_ALT)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 10)
    dpg.bind_theme(global_theme)

    with ui.theme() as run_button_theme:
        with ui.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_ACTIVE)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 26, 12)
    _THEMES["run_button"] = run_button_theme

    with ui.theme() as sidebar_theme:
        with ui.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_SIDEBAR)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
    _THEMES["sidebar"] = sidebar_theme


def _build_fonts() -> dict[str, ItemId]:
    """Segoe UI (ships with every Windows install) at a few sizes. Falls
    back to DPG's built-in default font silently if unavailable."""
    fonts: dict[str, ItemId] = {}
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    bold = Path(r"C:\Windows\Fonts\segoeuib.ttf")
    try:
        with ui.font_registry():
            if regular.exists():
                fonts["body"] = dpg.add_font(str(regular), 17)
            if bold.exists():
                fonts["title"] = dpg.add_font(str(bold), 28)
                fonts["subheader"] = dpg.add_font(str(bold), 18)
        if "body" in fonts:
            dpg.bind_font(fonts["body"])
    except Exception:
        return {}
    return fonts


def _build_dialogs() -> None:
    with dpg.texture_registry(tag='preview_textures', show=False):
        pass
    with ui.file_dialog(directory_selector=False, show=False, callback=on_save_settings,
                        tag='file_dialog_save_settings', width=800, height=480, default_filename='deeptune-settings'):
        dpg.add_file_extension('.json')
    with ui.file_dialog(directory_selector=False, show=False, callback=on_load_settings,
                        tag='file_dialog_load_settings', width=800, height=480):
        dpg.add_file_extension('.json')
    with ui.file_dialog(directory_selector=True, show=False, callback=on_data_folder_selected,
                        tag="file_dialog_data_folder", width=800, height=480):
        pass
    with ui.file_dialog(directory_selector=False, show=False, callback=on_data_file_selected,
                        tag="file_dialog_data_file", width=800, height=480):
        for ext in ('.*', '.mp4', '.MP4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg', ".parquet", ".csv", ".xlsx"):
            dpg.add_file_extension(ext)
    with ui.file_dialog(directory_selector=True, show=False, callback=on_outdir_selected,
                        tag="file_dialog_outdir", width=800, height=480):
        pass
    with ui.file_dialog(directory_selector=True, show=False, callback=on_test_data_folder_selected,
                        tag="file_dialog_test_data_folder", width=800, height=480):
        pass
    with ui.file_dialog(directory_selector=False, show=False, callback=on_test_data_file_selected,
                        tag="file_dialog_test_data_file", width=800, height=480):
        # '.*' first, plus every extension a single-file inference target
        # can be: a CSV/XLSX/parquet table, a single image, a single video
        # clip, or a labeled video CSV/XLSX list (already covered above).
        for ext in ('.*', '.csv', '.xlsx', '.parquet', *IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS):
            dpg.add_file_extension(ext)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _build_overview() -> None:
    with ui.group(tag="page_overview"):
        _heading("Fine-tune a model without touching the command line.",
                  "DeepTune's one-call pipeline - modality, model, and hyperparameters in, "
                  "a trained checkpoint, evaluation report, and embeddings out.")
        with _card("MAKE A RUN"):
            dpg.add_text("01   Choose your data", color=ACCENT_LIGHT)
            dpg.add_text("Pick a modality, point at a raw folder or an existing parquet file, and preview it.", wrap=850)
            dpg.add_text("02   Choose your model", color=ACCENT_LIGHT)
            dpg.add_text("Pick a model version and hyperparameters - transfer learning, PEFT/LoRA, epochs, batch size.", wrap=850)
            dpg.add_text("03   Run and review", color=ACCENT_LIGHT)
            dpg.add_text("Follow the live log, then inspect the holdout metrics, training curve, and embeddings.", wrap=850)
            with ui.group(horizontal=True):
                button = dpg.add_button(label="Choose a dataset", callback=on_nav_selected, user_data="dataset", height=42)
                dpg.bind_item_theme(button, _THEMES["run_button"])
                dpg.add_button(label="Browse results", callback=on_nav_selected, user_data="results", height=42)
        dpg.add_spacer(height=10)
        with _card("SUPPORTED MODALITIES"):
            dpg.add_text("Images  -  ResNet, DenseNet, Swin, EfficientNet, VGG, ViT, ConvNeXt, SigLIP", wrap=850)
            dpg.add_text("Text  -  Multilingual BERT, GPT-2", wrap=850)
            dpg.add_text("Tabular  -  GANDALF, TabPFN", wrap=850)
            dpg.add_text("Time series  -  DeepAR", wrap=850)
            dpg.add_text("Video  -  frame-sampling on any image backbone, or native r3d_18 / mc3_18 / "
                          "r2plus1d_18 / mvit_v2_s / swin3d", wrap=850)


def _build_dataset_page() -> None:
    with ui.group(tag="page_dataset", show=False):
        _heading("Dataset", "Choose what DeepTune should learn from.")
        with _card("MODALITY & SOURCE"):
            dpg.add_radio_button(gc.MODALITIES, tag="modality_radio", default_value="images",
                                 horizontal=True, callback=on_modality_changed)
            dpg.add_checkbox(tag="raw_data_checkbox", label="This is raw data (needs conversion to parquet)",
                             default_value=True, callback=on_raw_data_toggled)
            dpg.add_text("", tag="data_source_hint", color=TEXT_MUTED, wrap=850)
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="data_path_input", hint="Paste a path or browse...", width=-280,
                                   callback=on_config_changed)
                dpg.add_button(tag="browse_folder_button", label="Browse folder", callback=on_show_dialog,
                              user_data="file_dialog_data_folder")
                dpg.add_button(tag="browse_file_button", label="Browse file", callback=on_show_dialog,
                              user_data="file_dialog_data_file", show=False)
                dpg.add_button(tag="preview_button", label="Load / preview", callback=on_load_preview)
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="target_input", label="Target column", default_value="labels", width=220,
                                   callback=on_config_changed)
                dpg.add_input_text(tag="grouper_input", label="Grouper column (optional)", width=220,
                                   callback=on_config_changed)
                _tip("Column used for StratifiedGroupKFold splitting, e.g. a patient/subject ID, "
                     "so rows from the same group never span train and test.")
        dpg.add_spacer(height=10)
        with ui.group(tag='video_import_group', show=False):
            with _card('BUILD A VIDEO DATASET'):
                dpg.add_text('1. Select files to import them.  2. Label each clip.  3. Use the dataset for training.', color=TEXT_MUTED, wrap=850)
                dpg.add_text('A class label is what the video shows, such as walking or running. You can preview without a label.', color=TEXT_MUTED, wrap=850)
                dpg.add_text('No clips selected yet.', tag='video_selection_text', color=ACCENT_LIGHT, wrap=850)
                with ui.group(horizontal=True):
                    dpg.add_button(label='Select MP4 / video files...', callback=on_show_dialog, user_data='file_dialog_data_file')
                    dpg.add_input_text(tag='video_label_input', hint='Class label, e.g. walking', width=240)
                    dpg.add_button(label='Apply label to selected', tag='apply_video_label_button', callback=on_add_video_clips, enabled=False)
                dpg.add_text('Choose video files. They will appear in the list immediately.', tag='video_import_status', color=TEXT_MUTED, wrap=850)
                dpg.add_text('', tag='video_decode_status', color=TEXT_MUTED, wrap=850)
                dpg.add_text('0 clips / 0 classes', tag='video_clips_count', color=TEXT_MUTED)
                with ui.child_window(tag='video_clips_container', height=180):
                    dpg.add_text('Your labeled clip list will appear here.', color=TEXT_MUTED)
                with ui.group(horizontal=True):
                    dpg.add_button(label='Use clip dataset', tag='use_video_dataset_button', callback=on_use_video_clips, enabled=False)
                    dpg.add_button(label='Clear list', callback=on_clear_video_clips)
            dpg.add_spacer(height=10)
        with _card("PREVIEW", tag='dataset_preview_card'):
            dpg.add_text("Load a dataset to preview it.", tag="preview_note", color=TEXT_MUTED, wrap=850)
            with ui.group(tag='video_preview_controls', show=False):
                with ui.group(horizontal=True):
                    dpg.add_button(label='Play', tag='video_play_button', callback=on_video_play_pause)
                    dpg.add_button(label='Restart', tag='video_restart_button', callback=on_video_restart)
                    dpg.add_checkbox(label='Loop', tag='video_loop_checkbox', default_value=True, callback=on_video_loop_changed)
                    dpg.add_button(label='Open in video player', callback=on_open_video_player)
                    dpg.add_text('Muted preview | Open player for audio', color=TEXT_MUTED)
                with ui.group(horizontal=True):
                    dpg.add_slider_float(tag='video_seek_slider', width=-150, min_value=0, max_value=1,
                                         format='%.1f s', callback=on_video_seek, clamped=True, no_input=True)
                    dpg.add_text('0:00 / 0:00', tag='video_time_text')
                dpg.add_text('', tag='video_playback_status', color=TEXT_MUTED)
            with ui.child_window(tag="preview_container", height=320, horizontal_scrollbar=True):
                pass


def _build_model_page() -> None:
    with ui.group(tag="page_model", show=False):
        _heading("Model & training", "Everything deeptune.py's one-call CLI accepts, as fields instead of flags.")
        with _card("MODEL"):
            dpg.add_combo(tag="model_version_combo", label="Model version",
                         items=gc.MODEL_VERSIONS_BY_MODALITY["images"], default_value="resnet18",
                         width=320, callback=on_model_version_changed)
            with ui.group(tag="num_classes_group", horizontal=True):
                dpg.add_input_int(tag="num_classes_input", label="Number of classes", default_value=2,
                                  width=160, callback=on_config_changed)
                dpg.add_radio_button(["cls", "reg"], tag="mode_radio", default_value="cls",
                                     horizontal=True, callback=on_config_changed)
                _tip("cls = classification, reg = regression. Also used by TabPFN; GANDALF uses its own "
                     "classification/regression choice below instead.")
        dpg.add_spacer(height=10)
        with ui.group(tag="transfer_learning_group"):
            with _card("TRANSFER LEARNING"):
                with ui.group(horizontal=True):
                    dpg.add_combo(tag="added_layers_combo", label="Added layers", items=["1", "2"],
                                 default_value="2", width=100, callback=on_config_changed)
                    dpg.add_input_int(tag="embed_size_input", label="Embedding size", default_value=1000,
                                      width=140, callback=on_config_changed)
                with ui.group(horizontal=True):
                    dpg.add_checkbox(tag="freeze_backbone_checkbox", label="Freeze backbone",
                                     callback=on_config_changed)
                    dpg.add_checkbox(tag="use_peft_checkbox", label="Use PEFT / LoRA",
                                     callback=on_config_changed)
        dpg.add_spacer(height=10)
        with _card("TRAINING"):
            with ui.group(horizontal=True):
                dpg.add_button(label='Quick check: 1 epoch', callback=on_training_preset, user_data='quick')
                dpg.add_button(label='Standard: 10 epochs', callback=on_training_preset, user_data='standard')
            dpg.add_checkbox(tag='fixed_seed_checkbox', label='Use a reproducible seed (42)',
                             default_value=True, callback=on_config_changed)
            with ui.group(horizontal=True):
                dpg.add_input_int(tag="batch_size_input", label="Batch size", default_value=16, width=140,
                                  callback=on_config_changed)
                dpg.add_input_int(tag="num_epochs_input", label="Epochs", default_value=10, width=120,
                                  callback=on_config_changed)
                dpg.add_input_double(tag="learning_rate_input", label="Learning rate", default_value=1e-4,
                                     format="%.6f", width=160, callback=on_config_changed)
        dpg.add_spacer(height=10)
        with ui.group(tag="video_options_group", show=False):
            with _card("VIDEO"):
                with ui.group(horizontal=True):
                    dpg.add_input_int(tag="num_frames_input", label="Frames sampled per clip", default_value=8,
                                      width=160, callback=on_config_changed)
                    dpg.add_radio_button(["mean", "attention"], tag="pooling_radio", default_value="mean",
                                         horizontal=True, callback=on_config_changed)
                    _tip("Pooling only applies to frame-sampling video models (any 2D backbone, e.g. "
                         "resnet50). Native models (r3d_18, mvit_v2_s, swin3d_*) ignore it.")
        with ui.group(tag="gandalf_group", show=False):
            with _card("GANDALF"):
                dpg.add_radio_button(["classification", "regression"], tag="gandalf_type_radio",
                                     default_value="classification", horizontal=True, callback=on_config_changed)
                dpg.add_text("One column name per line (or comma-separated):", color=TEXT_MUTED)
                dpg.add_input_text(tag="continuous_cols_input", label="Continuous columns", multiline=True,
                                   height=60, width=400, callback=on_config_changed)
                dpg.add_input_text(tag="categorical_cols_input", label="Categorical columns", multiline=True,
                                   height=60, width=400, callback=on_config_changed)
        with ui.group(tag="tabpfn_group", show=False):
            with _card("TABPFN"):
                dpg.add_checkbox(tag="finetuning_mode_checkbox", label="Fine-tune model weights",
                                 callback=on_config_changed)
                dpg.add_text('When disabled, TabPFN uses its pretrained model with your dataset as context.',
                             color=TEXT_MUTED, wrap=850)
        with ui.group(tag="timeseries_options_group", show=False):
            with _card("TIME SERIES"):
                dpg.add_input_text(tag="time_idx_column_input", label="Time index column", default_value="labels",
                                   width=220, callback=on_config_changed)


def _build_run_page() -> None:
    with ui.group(tag="page_run", show=False):
        _heading("Run", "Launch deeptune.py with the settings from the Dataset and Model pages.")
        with _card("OUTPUT & READINESS"):
            with ui.group(horizontal=True):
                dpg.add_button(label='Save settings...', callback=on_show_dialog, user_data='file_dialog_save_settings')
                dpg.add_button(label='Load settings...', callback=on_show_dialog, user_data='file_dialog_load_settings')
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="outdir_input", hint="Where should results be saved?", width=-190,
                                   callback=on_config_changed)
                dpg.add_button(label="Browse", callback=on_show_dialog, user_data="file_dialog_outdir")
            dpg.add_text("Choose a dataset first.", tag="validation_text", color=WARNING, wrap=850)
            with ui.collapsing_header(label="Reproducible command"):
                dpg.add_input_text(tag="command_text", readonly=True, multiline=True, width=-1, height=80)
                dpg.add_button(label="Copy command", callback=on_copy_command)
        dpg.add_spacer(height=10)
        with _card("LAUNCH"):
            with ui.group(horizontal=True):
                button = dpg.add_button(tag="run_button", label="Run", callback=on_run_clicked, height=44)
                dpg.bind_item_theme(button, _THEMES["run_button"])
                dpg.add_button(tag="cancel_button", label="Stop", callback=on_cancel_clicked, enabled=False, height=44)
                dpg.add_text("Elapsed  00:00:00", tag="elapsed_text", color=TEXT_MUTED)
                dpg.add_loading_indicator(tag="run_loading_indicator", show=False, radius=2.0)
            dpg.add_text("Ready when you are.", tag="run_stage_text", color=ACCENT_LIGHT, wrap=850)
        dpg.add_spacer(height=10)
        with _card("LIVE OUTPUT"):
            with ui.group(horizontal=True):
                dpg.add_checkbox(tag="follow_log_checkbox", label="Follow output", default_value=True)
                dpg.add_button(label="Save log", callback=on_save_log)
            with ui.child_window(tag="log_window", height=320, horizontal_scrollbar=True):
                dpg.add_text("", tag="log_box", wrap=0)


def _build_results_page() -> None:
    with ui.group(tag="page_results", show=False):
        _heading("Results", "Browse completed and partial experiments, inspect metrics, and export results.")
        with _card("RUN HISTORY"):
            with ui.group(horizontal=True):
                dpg.add_button(label="Refresh", callback=on_refresh_results)
                dpg.add_button(label="Open selected folder", callback=on_open_output)
                dpg.add_button(label="Export metrics CSV", callback=on_export_metrics)
                dpg.add_text("0 runs found", tag="results_count_text", color=TEXT_MUTED)
            dpg.add_combo(tag="results_run_combo", items=[], width=-1, callback=on_results_run_selected)
        dpg.add_spacer(height=10)
        with _card("HOLDOUT METRICS"):
            with ui.child_window(tag="results_metrics_container", height=220, horizontal_scrollbar=True):
                dpg.add_text("Run something first.", color=TEXT_MUTED)
        dpg.add_spacer(height=10)
        with _card("TRAINING CURVE"):
            with ui.plot(label="Loss per epoch", height=260, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="results_x_axis", label="Epoch")
                with dpg.plot_axis(dpg.mvYAxis, tag="results_y_axis", label="Loss"):
                    dpg.add_line_series([], [], label="Training loss", tag="results_series_loss_plot",
                                       parent="results_y_axis")
                    dpg.add_line_series([], [], label="Validation loss", tag="results_series_valloss_plot",
                                       parent="results_y_axis")
        dpg.add_spacer(height=10)
        with _card("PATHS"):
            with ui.child_window(tag="results_paths_container", height=130, horizontal_scrollbar=True):
                dpg.add_text("Run something first.", color=TEXT_MUTED)


def _build_test_page() -> None:
    with ui.group(tag="page_test", show=False):
        _heading("Test", "Load a trained run's checkpoint and run it on new data: predicted class or "
                          "value, confidence, and accuracy against the new data's own labels, when it has any.")
        with _card("TRAINED MODEL"):
            with ui.group(horizontal=True):
                dpg.add_button(label="Refresh", callback=on_refresh_test_runs)
                dpg.add_text("Runs are listed from the Run page's output folder.", color=TEXT_MUTED)
            dpg.add_combo(tag="test_run_combo", items=[], width=-1, callback=on_test_run_selected)
            dpg.add_text("Choose an output folder on the Run page, then Refresh.", tag="test_run_info_text",
                         color=ACCENT_LIGHT, wrap=850)
        dpg.add_spacer(height=10)
        with _card("NEW DATA"):
            dpg.add_text("A single image or video file, a folder of them (one subfolder per class to "
                         "compare against known labels, or a flat folder to just predict), or a "
                         "CSV/XLSX/parquet file for text/tabular/timeseries.", wrap=850, color=TEXT_MUTED)
            with ui.group(horizontal=True):
                dpg.add_input_text(tag="test_data_path_input", hint="Path to new data", width=-280)
                dpg.add_button(label="Browse folder", callback=on_show_dialog, user_data="file_dialog_test_data_folder")
                dpg.add_button(label="Browse file", callback=on_show_dialog, user_data="file_dialog_test_data_file")
        dpg.add_spacer(height=10)
        with _card("LAUNCH"):
            with ui.group(horizontal=True):
                button = dpg.add_button(tag="test_run_button", label="Test model", callback=on_run_test_clicked,
                                        height=44, enabled=False)
                dpg.bind_item_theme(button, _THEMES["run_button"])
                dpg.add_button(tag="test_cancel_button", label="Stop", callback=on_cancel_test_clicked,
                               enabled=False, height=44)
                dpg.add_text("Elapsed  00:00:00", tag="test_elapsed_text", color=TEXT_MUTED)
                dpg.add_loading_indicator(tag="test_loading_indicator", show=False, radius=2.0)
            dpg.add_text("Choose a trained run and new data to begin.", tag="test_stage_text",
                         color=ACCENT_LIGHT, wrap=850)
        dpg.add_spacer(height=10)
        with _card("LIVE OUTPUT"):
            with ui.group(horizontal=True):
                dpg.add_checkbox(tag="test_follow_log_checkbox", label="Follow output", default_value=True)
                dpg.add_button(label="Open output folder", callback=on_open_test_output)
            with ui.child_window(tag="test_log_window", height=180, horizontal_scrollbar=True):
                dpg.add_text("", tag="test_log_box", wrap=0)
        dpg.add_spacer(height=10)
        with _card("METRICS"):
            with ui.child_window(tag="test_metrics_container", height=120, horizontal_scrollbar=True):
                dpg.add_text("Run a test first.", color=TEXT_MUTED)
        dpg.add_spacer(height=10)
        with _card("PREDICTIONS"):
            with ui.child_window(tag="test_predictions_container", height=280, horizontal_scrollbar=True):
                dpg.add_text("Run a test first.", color=TEXT_MUTED)


def build_ui() -> None:
    global _FONTS
    _build_theme()
    _FONTS = _build_fonts()
    _build_dialogs()
    with ui.window(tag="main_window", label="DeepTune"):
        with ui.child_window(height=92, border=False, no_scrollbar=True):
            with ui.group(horizontal=True):
                with ui.drawlist(width=50, height=50):
                    dpg.draw_rectangle((3, 27), (14, 47), color=ACCENT, fill=ACCENT, rounding=3)
                    dpg.draw_rectangle((19, 17), (30, 47), color=ACCENT_HOVER, fill=ACCENT_HOVER, rounding=3)
                    dpg.draw_rectangle((35, 5), (46, 47), color=ACCENT_LIGHT, fill=ACCENT_LIGHT, rounding=3)
                dpg.add_spacer(width=6)
                with ui.group():
                    brand = dpg.add_text("DeepTune", color=TEXT_PRIMARY)
                    if "title" in _FONTS:
                        dpg.bind_item_font(brand, _FONTS["title"])
                    dpg.add_text("ONE COMMAND. FIVE MODALITIES.", color=ACCENT_LIGHT)
        dpg.add_separator()
        with ui.group(horizontal=True):
            with ui.child_window(width=225, height=-1) as sidebar:
                dpg.bind_item_theme(sidebar, _THEMES["sidebar"])
                dpg.add_text("WORKSPACE", color=TEXT_MUTED)
                dpg.add_spacer(height=12)
                for key, label in NAV_ITEMS:
                    dpg.add_selectable(label=label, tag=f"nav_{key}", default_value=key == "overview",
                                       callback=on_nav_selected, user_data=key, height=36)
                dpg.add_spacer(height=25)
                dpg.add_separator()
                dpg.add_text("SESSION", color=TEXT_MUTED)
                dpg.add_text("Ready", tag="status_text", color=SUCCESS, wrap=180)
                dpg.add_spacer(height=20)
                dpg.add_text("Runs happen in a subprocess.\nClosing this window stops any active run.",
                             color=TEXT_MUTED, wrap=190)
            with ui.child_window(tag="workspace_content", width=-1, height=-1, border=False):
                _build_overview()
                _build_dataset_page()
                _build_model_page()
                _build_run_page()
                _build_results_page()
                _build_test_page()
    on_modality_changed()
    on_config_changed()


def run_gui() -> None:
    dpg.create_context()
    try:
        dpg.configure_app(manual_callback_management=True)
        build_ui()
        if SESSION_PATH.exists():
            try:
                apply_config(gc.load_config(SESSION_PATH))
            except (OSError, ValueError, TypeError) as exc:
                _set_status(f'Previous settings could not be restored: {exc}', WARNING)
        dpg.create_viewport(title="DeepTune", width=1320, height=920, min_width=1080, min_height=720)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            dpg.run_callbacks(dpg.get_callback_queue())
            poll_run_state()
            poll_infer_run_state()
            poll_preview()
            poll_video_playback()
            dpg.render_dearpygui_frame()
    finally:
        try:
            gc.save_config(_build_run_config(), SESSION_PATH)
        except (OSError, ValueError, SystemError):
            pass
        if _current_run is not None:
            _current_run.close()
        if _current_infer_run is not None:
            _current_infer_run.close()
        stop_video_playback(timeout=2)
        dpg.destroy_context()


def run_self_test(data_dir: Path, modality: str, outdir: Path) -> int:
    """Exercise widget construction, nav switching, and a real (tiny) run
    without any mouse/window interaction -- mirrors df-analyze desktop's
    --self-test entry point."""
    dpg.create_context()
    try:
        build_ui()
        dpg.create_viewport(title="DeepTune self-test", width=1320, height=920)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        for _ in range(5):
            dpg.render_dearpygui_frame()
        for key, _ in NAV_ITEMS:
            on_nav_selected(user_data=key)
            for _ in range(3):
                dpg.render_dearpygui_frame()
        for name in gc.MODALITIES:
            dpg.set_value('modality_radio', name)
            on_modality_changed()
            dpg.render_dearpygui_frame()
        print("[self-test] UI construction and navigation OK")

        dpg.set_value("modality_radio", modality)
        on_modality_changed()
        dpg.set_value("data_path_input", str(data_dir))
        dpg.set_value("raw_data_checkbox", True)
        on_raw_data_toggled()
        dpg.set_value("outdir_input", str(outdir))
        dpg.set_value("num_epochs_input", 1)
        dpg.set_value("num_frames_input", 2)
        dpg.set_value("freeze_backbone_checkbox", True)
        on_config_changed()
    finally:
        dpg.destroy_context()

    print("[self-test] Running a real (tiny) DeepTune job...")
    cfg = _build_run_config_headless(modality, data_dir, outdir)
    handle = start_run(cfg)
    handle.thread.join(timeout=600)
    if handle.thread.is_alive():
        handle.close()
        print('[self-test] FAILED: training exceeded 10 minutes.')
        return 1
    chunks: list[str] = []
    while True:
        try:
            chunks.append(handle.log_queue.get_nowait())
        except queue.Empty:
            break
    encoding = sys.stdout.encoding or "utf-8"
    print("".join(chunks)[-4000:].encode(encoding, errors="replace").decode(encoding))
    status, message = handle.done_queue.get_nowait()
    if status != "success":
        print(f"[self-test] FAILED: {message}")
        return 1
    latest = gc.find_latest_run(outdir)
    if latest is None:
        print(f"[self-test] FAILED: no completed run found under {outdir}")
        return 1
    results = gc.read_run_results(latest)
    if results.metrics is None or results.checkpoint_path is None or results.embeddings_path is None:
        print('[self-test] FAILED: missing checkpoint, metrics, or embeddings.')
        return 1
    print(f"[self-test] PASSED: {latest}")
    return 0


def _build_run_config_headless(modality: str, data_dir: Path, outdir: Path) -> gc.RunConfig:
    cfg = gc.RunConfig(
        modality=modality, df=data_dir, raw_data=True, out=outdir, target="labels",
        model_version="resnet18" if modality in ("images", "video") else gc.MODEL_VERSIONS_BY_MODALITY[modality][0],
        num_classes=2, mode="cls", added_layers=1, embed_size=32, freeze_backbone=True,
        batch_size=2, num_epochs=1, learning_rate=1e-3, num_frames=2,
    )
    if modality in ('images', 'video'):
        summary = preview_dataset(cfg)
        cfg.num_classes = summary.class_count or 2
    return cfg


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='DeepTune desktop workspace')
    parser.add_argument('--self-test', action='store_true', help='Exercise the UI and a small training run.')
    parser.add_argument('--data', type=Path, help='Raw dataset for the self-test.')
    parser.add_argument('--modality', choices=gc.MODALITIES, default='images')
    parser.add_argument('--outdir', type=Path, default=Path('desktop-smoke-results'))
    args = parser.parse_args()
    if args.self_test:
        if args.data is None:
            parser.error('--self-test requires --data pointing to a real dataset.')
        if args.modality not in ('images', 'video'):
            parser.error('--self-test uses a raw images or video folder. Test table modalities through the normal app.')
        return run_self_test(args.data, args.modality, args.outdir)
    run_gui()
    return 0


if __name__ == '__main__':
    # No-op unless this process was itself spawned by the `multiprocessing`
    # module (e.g. a DataLoader worker with num_workers > 0) from inside a
    # frozen build's --deeptune-worker process, where sys.executable is
    # this same combined GUI+worker executable rather than a plain
    # python.exe -- required by multiprocessing's own docs for any frozen
    # Windows executable that spawns child processes, harmless otherwise.
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
