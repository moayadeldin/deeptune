import json
import time

import pytest

from desktop import runner
from desktop.config import RunConfig


@pytest.mark.parametrize('code,status', [(0, 'success'), (3, 'error')])
def test_subprocess_streams_unicode_progress_and_persists_status(tmp_path, monkeypatch, code, status):
    script = tmp_path / 'worker.py'
    script.write_text(f"import sys\nsys.stdout.write('progress\\rfinished ✓\\n')\nsys.exit({code})", encoding='utf-8')
    monkeypatch.setattr(runner, 'DEEPTUNE_SCRIPT', script)
    handle = runner.start_run(RunConfig(df=tmp_path, out=tmp_path / 'out'))
    handle.thread.join(timeout=15)
    assert not handle.thread.is_alive()
    assert handle.done_queue.get_nowait()[0] == status
    assert 'progress\rfinished ✓' in (handle.session_dir / 'run.log').read_bytes().decode('utf-8')
    assert json.loads((handle.session_dir / 'run.json').read_text())['status'] == status
    assert (handle.session_dir / 'settings.json').is_file()


def test_cancel_terminates_worker_and_records_cancellation(tmp_path, monkeypatch):
    script = tmp_path / 'worker.py'
    script.write_text('import time\nprint("ready", flush=True)\ntime.sleep(120)\n')
    monkeypatch.setattr(runner, 'DEEPTUNE_SCRIPT', script)
    handle = runner.start_run(RunConfig(df=tmp_path, out=tmp_path / 'out'))
    deadline = time.monotonic() + 10
    while handle._process_state.process is None and time.monotonic() < deadline:
        time.sleep(.01)
    handle.close()
    assert not handle.thread.is_alive()
    assert handle._process_state.process.poll() is not None
    assert handle.done_queue.get_nowait()[0] == 'cancelled'
    assert json.loads((handle.session_dir / 'run.json').read_text())['status'] == 'cancelled'
