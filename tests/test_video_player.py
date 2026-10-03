import time

import cv2
import numpy as np
import pytest

from desktop.video_player import VideoPlayer


def make_clip(path, count=20, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (160, 90))
    assert writer.isOpened()
    try:
        for index in range(count):
            writer.write(np.full((90, 160, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()
    return path


def wait_for(player, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    last_frame = None
    while time.monotonic() < deadline:
        state, frame = player.poll()
        last_frame = frame if frame is not None else last_frame
        assert not state.error, state.error
        if predicate(state, last_frame):
            return state, last_frame
        time.sleep(.01)
    raise AssertionError(f'Playback timed out: {player.state}')


def test_real_mp4_advances_frames_and_pause_keeps_position(tmp_path):
    player = VideoPlayer(make_clip(tmp_path / 'motion.MP4'))
    try:
        _, first = wait_for(player, lambda state, frame: frame is not None)
        state, later = wait_for(player, lambda state, frame: frame is not None and frame.index >= first.index + 2)
        assert not np.array_equal(first.pixels, later.pixels)
        assert state.position > 0
        player.pause()
        time.sleep(.05)  # Allow an in-flight decode to finish.
        paused = player.state.position
        time.sleep(.2)
        assert player.state.position == paused
        assert not player.state.playing
        player.play()
        wait_for(player, lambda state, frame: state.position > paused)
    finally:
        player.close()
    assert not player.thread.is_alive()
    player.path.rename(tmp_path / 'released.mp4')


def test_seek_updates_the_frame_while_paused_and_clamps_to_duration(tmp_path):
    player = VideoPlayer(make_clip(tmp_path / 'seek.mp4'), autoplay=False)
    try:
        wait_for(player, lambda state, frame: state.ready)
        player.seek(1.5)
        state, frame = wait_for(player, lambda state, frame: frame is not None and frame.index == 15)
        assert state.position == pytest.approx(1.5)
        assert not state.playing
        player.seek(999)
        wait_for(player, lambda state, frame: frame is not None and frame.index == 19)
        player.seek(0)
        wait_for(player, lambda state, frame: frame is not None and frame.index == 0)
    finally:
        player.close()


def test_end_of_video_replay_loop_and_single_frame_mailbox(tmp_path):
    player = VideoPlayer(make_clip(tmp_path / 'short.mp4', count=4, fps=20), loop=False)
    try:
        deadline = time.monotonic() + 4
        while not player.state.ended and time.monotonic() < deadline:
            time.sleep(.01)
        assert player.state.ended and not player.state.playing
        _, frame = player.poll()
        assert frame.index == 3  # A slow UI receives only the latest frame.
        assert player.poll()[1] is None
        player.play()
        wait_for(player, lambda state, frame: state.playing and frame is not None and frame.index < 3)
        player.set_loop(True)
        time.sleep(.5)
        assert player.state.playing and not player.state.ended
    finally:
        player.close()


def test_corrupt_video_reports_error_and_releases_worker(tmp_path):
    path = tmp_path / 'broken.mp4'
    path.write_bytes(b'not a video')
    player = VideoPlayer(path)
    player.thread.join(timeout=4)
    assert player.state.error
    assert not player.state.playing and not player.thread.is_alive()
    player.close()
