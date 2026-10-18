"""Bounded, background video playback using OpenCV's bundled video decoder.

The worker owns the decoder. The render thread only consumes the newest RGBA
frame, so a slow interface never accumulates video frames in memory.
"""
from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time


@dataclass(frozen=True)
class PlaybackState:
    ready: bool = False
    playing: bool = False
    ended: bool = False
    position: float = 0.0
    duration: float = 0.0
    fps: float = 0.0
    frame_index: int = 0
    error: str = ''


@dataclass(frozen=True)
class VideoFrame:
    width: int
    height: int
    pixels: object
    index: int


class VideoPlayer:
    def __init__(self, path: Path, *, autoplay: bool = True, loop: bool = True):
        self.path = Path(path)
        self._condition = threading.Condition()
        self._state = PlaybackState(playing=autoplay)
        self._frame = None
        self._playing = autoplay
        self._loop = loop
        self._seek = None
        self._revision = 0
        self._closed = False
        self.thread = threading.Thread(target=self._run, daemon=True, name='deeptune-video-preview')
        self.thread.start()

    @property
    def state(self) -> PlaybackState:
        with self._condition:
            return self._state

    def poll(self) -> tuple[PlaybackState, VideoFrame | None]:
        with self._condition:
            frame, self._frame = self._frame, None
            return self._state, frame

    def play(self) -> None:
        with self._condition:
            if self._closed or self._state.error:
                return
            if self._state.ended:
                self._seek = 0.0
                self._revision += 1
            self._playing = True
            self._state = replace(self._state, playing=True, ended=False)
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            self._playing = False
            self._state = replace(self._state, playing=False)
            self._condition.notify_all()

    def seek(self, seconds: float) -> None:
        if not math.isfinite(seconds):
            return
        with self._condition:
            self._seek = max(0.0, seconds)
            self._revision += 1
            self._frame = None  # Discard any frame queued before the seek.
            self._state = replace(self._state, ended=False)
            self._condition.notify_all()

    def set_loop(self, enabled: bool) -> None:
        with self._condition:
            self._loop = bool(enabled)

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._closed = True
            self._frame = None
            self._playing = False
            self._state = replace(self._state, playing=False)
            self._condition.notify_all()
        if timeout:
            self.thread.join(timeout=timeout)

    def _run(self) -> None:
        import cv2
        import numpy as np

        capture = None
        try:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise ValueError('Could not open the video. Its codec may be unsupported or the file may be corrupt.')
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
            reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = int(reported_count) if math.isfinite(reported_count) and reported_count > 0 else 0
            duration = count / fps
            with self._condition:
                self._state = replace(self._state, fps=fps, duration=duration)
            first_frame = True
            index = -1
            next_due = time.monotonic()
            while True:
                with self._condition:
                    if self._closed:
                        break
                    seek, self._seek = self._seek, None
                    revision = self._revision
                    if seek is None and not first_frame:
                        if not self._playing:
                            self._condition.wait()
                            next_due = time.monotonic()
                            continue
                        delay = next_due - time.monotonic()
                        if delay > 0:
                            self._condition.wait(timeout=delay)
                            continue
                if seek is not None:
                    target = max(0, int(seek * fps))
                    if count:
                        target = min(target, count - 1)
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                        raise ValueError('This video does not support seeking. Open it in your video player instead.')
                    index = target - 1
                    next_due = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    if first_frame:
                        raise ValueError('No video frame could be decoded from this file.')
                    if count and index < count - 2:
                        raise ValueError(f'Could not decode video frame {index + 1}. The clip may be damaged.')
                    with self._condition:
                        if self._closed:
                            break
                        if revision != self._revision:
                            continue
                        if self._loop and self._playing:
                            self._seek = 0.0
                            self._revision += 1
                        else:
                            self._playing = False
                            self._state = replace(self._state, playing=False, ended=True)
                    continue
                index += 1
                height, width = frame.shape[:2]
                scale = min(640 / width, 360 / height, 1.0)
                preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                pixels = cv2.cvtColor(preview, cv2.COLOR_BGR2RGBA).astype(np.float32).ravel() / 255
                packet = VideoFrame(preview.shape[1], preview.shape[0], pixels, index)
                with self._condition:
                    if self._closed:
                        break
                    if revision != self._revision:
                        continue
                    self._frame = packet
                    self._state = replace(self._state, ready=True, playing=self._playing,
                                          ended=False, position=index / fps, frame_index=index)
                first_frame = False
                next_due = max(next_due + 1 / fps, time.monotonic())
        except Exception as exc:
            with self._condition:
                self._state = replace(self._state, playing=False, error=str(exc))
        finally:
            if capture is not None:
                capture.release()
