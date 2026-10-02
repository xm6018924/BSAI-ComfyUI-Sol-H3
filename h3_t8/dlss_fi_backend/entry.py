"""Codec-free parent entry. No ComfyUI import, GPU import or UI side effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .process import IsolatedTaskError, run_isolated
from .resources import SerialProbeLease, file_identity


def file_video_path(video, native_type):
    # Avoid get_active_trim_window(): a negative trim can decode to find duration.
    # Exact type and storage contract prevent silently ignoring a subclass's edits.
    if type(video) is not native_type:
        raise ValueError('FI requires native file-backed VIDEO; save frame-batch VIDEO to a file first')
    fields = vars(video)
    required = ('_VideoFromFile__file', '_VideoFromFile__start_time', '_VideoFromFile__duration')
    if any(key not in fields for key in required):
        raise ValueError('Unsupported native VIDEO storage contract; source was not decoded')
    if (any(type(fields[k]) not in (int, float) or fields[k] != 0 for k in required[1:]) or
            fields.get('_VideoFromFile__crop') is not None):
        raise ValueError('FI cannot silently ignore VIDEO trim/crop; save the edited VIDEO to a new file first')
    source = fields[required[0]]
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError('FI requires a local file, not BytesIO')
    path = Path(source).resolve(strict=True)
    if not path.is_file():
        raise ValueError('VIDEO source is not a regular file')
    return path


def parse_cuts(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError('Cut list must be a bounded comma-separated string')
    if not text.strip():
        return []
    tokens = text.split(',')
    if any(not re.fullmatch(r'[1-9][0-9]{0,5}', token.strip()) for token in tokens):
        raise ValueError('Cuts are zero-based right-source frame indices, e.g. 24,48; use positive integers')
    values = [int(token) for token in tokens]
    if len(set(values)) != len(values):
        raise ValueError('Duplicate cut indices')
    return sorted(values)


def process_file(source, runtime, output_root, *, cuts='', timeout=600, check=lambda: None):
    if os.name != 'nt':
        raise RuntimeError('DLSS FI EXP currently requires Windows and one NVIDIA RTX GPU')
    check()
    markers = parse_cuts(cuts)
    source, runtime = Path(source).resolve(strict=True), Path(runtime).resolve(strict=True)
    output_root = Path(output_root).resolve(strict=True)
    if not source.is_file() or not runtime.is_dir() or not output_root.is_dir():
        raise ValueError('Expected an existing source file, runtime directory and output directory')
    # One OS-owned FI lease across copies/Comfy instances; never touches another process.
    with SerialProbeLease(Path(tempfile.gettempdir())/'T8-DLSS-FI-serial.lock'):
        tasks = output_root/'.dlss-fi-tasks'
        tasks.mkdir(exist_ok=True)
        if tasks.is_symlink() or tasks.resolve().parent != output_root:
            raise ValueError('FI task directory must stay inside the output directory')
        task = Path(tempfile.mkdtemp(prefix='task-', dir=tasks))
        try:
            receipt = run_isolated(Path(__file__).with_name('file_task.py'),
                ['--source', str(source), '--runtime', str(runtime), '--task-dir', str(task), '--cuts', json.dumps(markers)],
                timeout=timeout, check=check)
        except IsolatedTaskError as error:
            with (task/'failure.json').open('x', encoding='utf8') as stream:
                json.dump(error.receipt, stream, ensure_ascii=False, indent=2)
            raise
        check()
        result_file = task/'result.json'
        if result_file.stat().st_size > 8*1024**2:
            raise ValueError('Unexpectedly large FI result')
        result = json.loads(result_file.read_text(encoding='utf8'))
        candidate = task/'candidate.mp4'
        if (result['status'] != 'file_validated_pending_atomic_publish' or
                result['media']['file'] != file_identity(candidate) or result['source'] != file_identity(source)):
            raise ValueError('FI result/source identity changed before publication')
        if result['worker_stop']['owned_children_remaining'] or result['worker_stop']['unfinished_threads']:
            raise ValueError('FI worker cleanup is incomplete')
        destination = output_root/f'H3_DLSS_FI_2x_{uuid.uuid4().hex}.mp4'
        check()
        # Windows rename fails if destination exists; no replacement of source or another result.
        os.rename(candidate, destination)
        result.update(status='completed', saved_path=str(destination), isolation=receipt,
                      diagnostics=str(task), published_to_github=False)
        return destination, result
