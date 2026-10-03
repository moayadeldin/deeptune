# Pure, framework-free log-capture helpers used by desktop/runner.py.
#
# No threading, no DearPyGui dependency here on purpose -- this is the part
# of the desktop app that's cheapest to unit-test in isolation.
from __future__ import annotations

import re


_SPLIT_RE = re.compile(r"(\r|\n)")


class LineBuffer:
    """Turns a stream of raw text chunks (as produced by print()/tqdm's
    carriage-return-based progress updates) into a bounded, display-ready
    string.

    A '\\n' commits the current in-progress line to history. A '\\r'
    (tqdm-style in-place redraw) discards the in-progress line instead of
    committing it, so a progress bar's dozens of redraws collapse into one
    continuously-overwritten line rather than spamming the log with
    near-duplicate committed lines -- DeepTune's trainers print a tqdm bar
    per epoch, so this matters a lot for keeping the log readable.
    """

    def __init__(self, max_lines: int = 400) -> None:
        if max_lines < 1:
            raise ValueError('max_lines must be positive.')
        self._lines: list[str] = []
        self._current: str = ""
        self._max_lines = max_lines
        self._carriage_return = False

    def clear(self) -> None:
        self._lines.clear()
        self._current = ''
        self._carriage_return = False

    def feed(self, chunk: str) -> None:
        for part in _SPLIT_RE.split(chunk):
            if not part:
                continue
            if self._carriage_return:
                if part != '\n':
                    self._current = ''
                self._carriage_return = False
            if part == "\n":
                self._lines.append(self._current)
                self._lines = self._lines[-self._max_lines :]
                self._current = ""
            elif part == "\r":
                self._carriage_return = True
            elif part:
                self._current = (self._current + part)[-16000:]

    def render(self) -> str:
        return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', "\n".join([*self._lines, self._current]))
