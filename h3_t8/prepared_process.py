"""Owned Windows process tree for prepared jobs; no framework or GPU imports.

Reuse the existing kill-on-close Job Object and pre-execution ownership gate.
Unlike enumerating children periodically, job ownership also covers children
created just before a worker exits or between telemetry observations.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .dlss_fi_backend.process import BOOTSTRAP, WindowsJob


def note_original(original, message):
    """Attach cleanup evidence without replacing an exception on Python 3.10.

    Python 3.10 keeps the notes as an inspectable attribute; only Python 3.11+
    renders them automatically in tracebacks. Durable process receipts remain
    the primary diagnostic record. Even custom broken exception methods must
    not turn a cancellation into a different failure.
    """
    try:
        method = getattr(original, 'add_note', None)
        if callable(method):
            method(message)
        else:
            original.__notes__ = [*getattr(original, '__notes__', []), message]
    except BaseException:
        pass


def stop_owned(job, process, *, assigned):
    """Best-effort all cleanup steps, returning uncertainty instead of masking errors."""
    errors = []

    def attempt(label, action):
        try:
            return action()
        except BaseException as error:
            errors.append(f'{label}: {type(error).__name__}: {error}')
            return None

    if job is not None:
        # Even when accounting fails, attempt termination of this owned job.
        active = attempt('query before stop', job.active)
        if active != 0:
            attempt('terminate job', job.terminate)
    if process is not None:
        if not assigned and process.poll() is None:
            attempt('stop gated worker', process.kill)
        attempt('wait worker', lambda: process.wait(timeout=5))
    remaining = 0 if job is None else None
    if job is not None:
        deadline = time.monotonic() + 5
        while True:
            remaining = attempt('query after stop', job.active)
            if remaining in (None, 0) or time.monotonic() >= deadline:
                break
            time.sleep(.02)
        attempt('close job', job.close)
    if process is not None and process.stdin is not None:
        attempt('close gate', process.stdin.close)
    return {'active_after_cleanup': remaining, 'errors': errors}


def execute_owned(script, request_path, *, timeout, observe, interrupt, write_json):
    """Execute only caller-selected packaged scripts, never bundle-selected code."""
    if type(timeout) not in (int, float) or not 0 < timeout <= 7200:
        raise ValueError('Prepared deadline must be finite and at most7200seconds')
    if sys.flags.optimize:
        raise RuntimeError('Prepared worker validation requires Python without -O')
    script, request_path = Path(script).resolve(strict=True), Path(request_path).resolve(strict=True)
    stage = request_path.parent
    job = process = None
    assigned = False
    receipt = {'status': 'incomplete', 'job_assigned_before_task': False}
    started = time.monotonic()
    try:
        interrupt()
        environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2')
        if environment.get('CUDA_VISIBLE_DEVICES') not in (None, '0', '-1'):
            raise ValueError('Ambiguous inherited CUDA mask')
        # -I ignores PYTHONOPTIMIZE/PYTHONPATH; -S gates startup hooks until owned.
        job = WindowsJob()
        with (stage / 'stdout.log').open('xb') as stdout, (stage / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen([sys.executable, '-I', '-S', '-u', '-X', 'utf8', '-c',
                BOOTSTRAP, str(script), '--request', str(request_path)], cwd=script.parent,
                env=environment, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False,
                close_fds=True)
            receipt['pid'] = process.pid
            job.assign(process)
            assigned = True
            receipt['job_assigned_before_task'] = True
            write_json(stage / 'live.json', {'pid': process.pid, 'controller_pid': os.getpid(),
                'job_assigned_before_task': True})
            process.stdin.write(b'T8FI-GO\n')
            process.stdin.flush()
            process.stdin.close()
            while process.poll() is None:
                interrupt()
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('Prepared stage exceeded its bounded timeout')
                observe()
                try:
                    process.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            interrupt()
            receipt['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'Prepared worker failed ({process.returncode}); inspect {stage / "stderr.log"}')
            # Handle-signalled and Job accounting can differ by a scheduler tick.
            deadline = time.monotonic() + .25
            while job.active() and time.monotonic() < deadline:
                interrupt()
                time.sleep(.01)
            if job.active():
                raise RuntimeError('Prepared worker left owned descendants; refusing next stage')
            receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        original = sys.exc_info()[1]
        cleanup = stop_owned(job, process, assigned=assigned)
        receipt.update(cleanup=cleanup, wall_seconds=time.monotonic() - started)
        try:
            write_json(stage / 'process.json', receipt)
        except BaseException as error:
            cleanup['errors'].append(f'write process receipt: {type(error).__name__}: {error}')
        if cleanup['active_after_cleanup'] != 0 or cleanup['errors']:
            if original is not None:
                note_original(original, 'Prepared cleanup: ' + json.dumps(cleanup))
            else:
                raise RuntimeError('Owned prepared cleanup incomplete: ' + json.dumps(cleanup))
    return receipt
