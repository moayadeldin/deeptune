"""Correct Dear PyGui's generated context-manager return annotations.

Dear PyGui 2.3 annotates its @contextmanager generators as ``int | str``
instead of ``Iterator[int | str]``. Runtime behavior is correct; this small
boundary preserves every argument signature while correcting the return
type. Ordinary widget calls retain the library's original type checking.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, ParamSpec, cast

import dearpygui.dearpygui as dpg

ItemId = int | str
P = ParamSpec("P")


def _context(function: Callable[P, object]) -> Callable[P, AbstractContextManager[ItemId]]:
    return cast(Callable[P, AbstractContextManager[ItemId]], function)


window = _context(dpg.window)
child_window = _context(dpg.child_window)
group = _context(dpg.group)
theme = _context(dpg.theme)
theme_component = _context(dpg.theme_component)
font_registry = _context(dpg.font_registry)
file_dialog = _context(dpg.file_dialog)
collapsing_header = _context(dpg.collapsing_header)
table = _context(dpg.table)
table_row = _context(dpg.table_row)
plot = _context(dpg.plot)
tab_bar = _context(dpg.tab_bar)
tab = _context(dpg.tab)
tooltip = _context(dpg.tooltip)
drawlist = _context(dpg.drawlist)
