"""Run deeptune.py off the render thread, with cancellation support.

Adapted from the same pattern used by the sibling df-analyze project's
desktop app (desktop/runner.py there): a background thread owns the
subprocess, streams its stdout into a queue the GUI polls from the render
loop, and cancellation kills the whole process tree so training/eval/embed
subprocesses (and their DataLoader workers) don't linger.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import traceback
import codecs
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from . import config as gc

DoneResult = Tuple[str, str]
DEEPTUNE_SCRIPT = Path(__file__).resolve().parents[1] / "deeptune.py"
INFER_SCRIPT = Path(__file__).resolve().parents[1] / "infer.py"


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Stop the run and any worker processes (e.g. DataLoader workers) it started."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass


@dataclass
class _ProcessState:
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: Optional[subprocess.Popen[str]] = None
    stopping: bool = False

    def stop_if_requested(self) -> None:
        with self.lock:
            process = self.process
            if (
                not self.cancel_requested.is_set()
                or self.stopping
                or process is None
                or process.poll() is not None
            ):
                return
            self.stopping = True
        threading.Thread(
            target=_terminate_process,
            args=(process,),
            daemon=True,
            name="deeptune-cancel",
        ).start()


@dataclass
class RunHandle:
    thread: threading.Thread
    log_queue: "queue.Queue[str]"
    done_queue: "queue.Queue[DoneResult]"
    _process_state: Optional[_ProcessState] = field(default=None, repr=False)
    session_dir: Optional[Path] = None

    @property
    def can_cancel(self) -> bool:
        state = self._process_state
        return (
            state is not None
            and self.thread.is_alive()
            and not state.cancel_requested.is_set()
        )

    def cancel(self) -> bool:
        """Request cancellation without blocking the GUI; false if unavailable."""
        state = self._process_state
        if state is None or not self.can_cancel:
            return False
        state.cancel_requested.set()
        try:
            self.log_queue.put_nowait("\nCancelling the run and its worker processes...\n")
        except queue.Full:
            pass
        state.stop_if_requested()
        return True

    def close(self, timeout: float = 15) -> None:
        """Keep the process owner alive until cancellation has reached its child."""
        self.cancel()
        self.thread.join(timeout=timeout)
        state = self._process_state
        if self.thread.is_alive() and state is not None and state.process is not None:
            _terminate_process(state.process)


def _worker_command(script: Path, worker_sentinel: str, args: list[str]) -> Tuple[list[str], Path]:
    """Builds the subprocess command + working directory for running
    `script` (deeptune.py or infer.py) with `args`, in either a normal
    checkout (a real python.exe handed the script's on-disk path) or a
    packaged build (see packaging/): there, sys.executable is this same
    combined GUI+worker executable rather than a standalone python.exe, so
    there's no on-disk script to hand it -- `worker_sentinel`
    ("--deeptune-worker" / "--infer-worker") tells desktop/app.py's own
    entry point (see the top of that file) which one to run in-process
    instead, and the executable's own install directory is used as cwd
    rather than the (source-checkout-only) script's parent.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, worker_sentinel, *args], Path(sys.executable).resolve().parent
    return [sys.executable, "-u", str(script), *args], script.parent


def start_run(cfg: gc.RunConfig) -> RunHandle:
    """Launch deeptune.py for `cfg` and return progress queues the GUI polls."""
    cfg = gc.resolved_config(cfg)
    issues = gc.validate_config(cfg)
    if issues:
        raise ValueError('\n'.join(issues))
    session_dir = cfg.out / 'desktop-sessions' / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    gc.save_config(cfg, session_dir / 'settings.json')
    log_q: "queue.Queue[str]" = queue.Queue(maxsize=2000)
    done_q: "queue.Queue[DoneResult]" = queue.Queue()
    args = gc.build_argv(cfg)
    state = _ProcessState()
    thread = threading.Thread(
        target=_run_process,
        args=(args, log_q, done_q, state, session_dir),
        daemon=True,
        name="deeptune-run",
    )
    handle = RunHandle(thread, log_q, done_q, state, session_dir)
    thread.start()
    return handle


def _run_process(
    args: list[str],
    log_q: "queue.Queue[str]",
    done_q: "queue.Queue[DoneResult]",
    state: _ProcessState,
    session_dir: Optional[Path] = None,
) -> None:
    process: Optional[subprocess.Popen[str]] = None
    log_file = None
    out_dir = Path(args[args.index('--out') + 1])
    existing_runs = set(gc.list_runs(out_dir))
    result: DoneResult = ('error', 'Run did not finish.')
    started = datetime.now(timezone.utc).isoformat()

    def emit(chunk: str) -> None:
        if log_file is not None:
            log_file.write(chunk)
            log_file.flush()
        try:
            log_q.put_nowait(chunk)
        except queue.Full:
            try:
                log_q.get_nowait()
            except queue.Empty:
                pass
            log_q.put_nowait(chunk)

    try:
        if session_dir is not None:
            log_file = (session_dir / 'run.log').open('w', encoding='utf-8')
            gc.write_json(session_dir / 'run.json', {'status': 'running', 'started': started, 'argv': args})
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
            return
        env = os.environ.copy()
        # DeepTune's own print statements use emoji (e.g. print_training_log_table's
        # title); on a default Windows console codepage that raises a
        # UnicodeEncodeError that would otherwise look like the run itself failed,
        # even after training/eval/embed all completed successfully.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # Training has no interactive stdin; report authentication requirements
        # in the log instead of starting TabPFN's browser/login prompt.
        env.setdefault("TABPFN_NO_BROWSER", "1")
        # Avoid multiplying BLAS threads across every DataLoader worker.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        emit("Starting DeepTune. The first import (torch/transformers) can take a moment...\n")
        cmd, run_cwd = _worker_command(DEEPTUNE_SCRIPT, "--deeptune-worker", args)
        process = subprocess.Popen(
            cmd,
            cwd=run_cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=-1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        with state.lock:
            state.process = process
        state.stop_if_requested()
        if process.stdout is not None:
            with process.stdout:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                while chunk := process.stdout.read1(4096):
                    emit(decoder.decode(chunk))
                emit(decoder.decode(b'', final=True))
        code = process.wait()
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
        elif code == 0:
            result = ('success', '')
        else:
            result = ('error', f'DeepTune exited with code {code}. See the run log for details.')
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
        else:
            result = ('error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
    finally:
        if log_file is not None:
            log_file.close()
            log_file = None
        if session_dir is not None:
            try:
                manifest = {'status': result[0], 'message': result[1], 'started': started,
                            'finished': datetime.now(timezone.utc).isoformat(), 'session_dir': str(session_dir), 'argv': args}
                gc.write_json(session_dir / 'run.json', manifest)
                new_runs = set(gc.list_runs(out_dir)) - existing_runs
                if len(new_runs) == 1:
                    gc.write_json(next(iter(new_runs)) / 'desktop-run.json', manifest)
            except OSError as exc:
                emit(f'Could not save run status: {exc}\n')
        done_q.put(result)


def start_infer_run(cfg: gc.InferenceConfig) -> RunHandle:
    """Launch infer.py for `cfg` (the Test page's counterpart to
    start_run/deeptune.py) and return progress queues the GUI polls."""
    issues = gc.validate_infer_config(cfg)
    if issues:
        raise ValueError('\n'.join(issues))
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    log_q: "queue.Queue[str]" = queue.Queue(maxsize=2000)
    done_q: "queue.Queue[DoneResult]" = queue.Queue()
    args = gc.build_infer_argv(cfg)
    state = _ProcessState()
    thread = threading.Thread(
        target=_infer_process,
        args=(args, log_q, done_q, state),
        daemon=True,
        name="deeptune-infer",
    )
    handle = RunHandle(thread, log_q, done_q, state, None)
    thread.start()
    return handle


def _infer_process(
    args: list[str],
    log_q: "queue.Queue[str]",
    done_q: "queue.Queue[DoneResult]",
    state: _ProcessState,
) -> None:
    """Same subprocess/streaming/cancellation shape as _run_process, minus
    the desktop-sessions log file and deeptune-*-run-folder bookkeeping
    that only apply to deeptune.py's training output layout -- infer.py
    writes directly into its --out folder (predictions.csv,
    inference_metrics.json), with no per-run subfolder of its own."""
    process: Optional[subprocess.Popen[str]] = None
    result: DoneResult = ('error', 'Run did not finish.')

    def emit(chunk: str) -> None:
        try:
            log_q.put_nowait(chunk)
        except queue.Full:
            try:
                log_q.get_nowait()
            except queue.Empty:
                pass
            log_q.put_nowait(chunk)

    try:
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        emit("Starting inference. The first import (torch/transformers) can take a moment...\n")
        cmd, run_cwd = _worker_command(INFER_SCRIPT, "--infer-worker", args)
        process = subprocess.Popen(
            cmd,
            cwd=run_cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=-1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        with state.lock:
            state.process = process
        state.stop_if_requested()
        if process.stdout is not None:
            with process.stdout:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                while chunk := process.stdout.read1(4096):
                    emit(decoder.decode(chunk))
                emit(decoder.decode(b'', final=True))
        code = process.wait()
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
        elif code == 0:
            result = ('success', '')
        else:
            result = ('error', f'Inference exited with code {code}. See the run log for details.')
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        if state.cancel_requested.is_set():
            result = ('cancelled', 'Run cancelled.')
        else:
            result = ('error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
    finally:
        done_q.put(result)
