"""Trusted isolated FI task; never imported/executed automatically by ComfyUI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

# Direct execution occurs only behind process.run_isolated's ownership gate.
# Build a private package namespace without executing the project's node imports.
if not __package__:
    import types
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('_t8_fi_worker')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    __package__ = '_t8_fi_worker.dlss_fi_backend'

from .device_binding import bind_probe, cuda_device_inventory
from .frame_stream import FrameStream
from .media import decode_frames, encode_video, inspect_source, mux_and_validate
from .resources import NvmlResourceReader, ResourceGuard, file_identity
from .transport import BinarySession, runtime_identity


class Monitor:
    """Sticky resource failure also cancels blocked worker requests."""
    def __init__(self, reader):
        self.reader, self.guard = reader, ResourceGuard()
        self.stop, self.cancel = threading.Event(), threading.Event()
        self.error = None
        self.startup = reader.sample()
        if self.guard.observe(self.startup, startup=True):
            raise RuntimeError(self.guard.reason)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(.25):
            try:
                if self.guard.observe(self.reader.sample()):
                    raise RuntimeError(self.guard.reason)
            except Exception as error:
                self.error = error
                self.cancel.set()
                return

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'FI resource protection: {self.error}') from self.error

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('FI resource monitor did not stop')
        self.check()


def write_json(path, data):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def process_file(source_path, runtime, task_dir, cuts=()):
    import psutil
    task_dir = Path(task_dir).resolve(strict=True)
    if not task_dir.is_dir() or any(task_dir.iterdir()):
        raise ValueError('FI task requires its own empty directory')
    source = inspect_source(source_path, cuts=cuts)
    if source['plan'].source_rate > 60:
        raise ValueError('First FI route accepts at most 60 source fps (up to 120 output fps)')
    if source['plan'].generated_count == 0:
        raise ValueError('All intervals are cuts; no interpolated interval remains')
    identities = runtime_identity(runtime)
    implementation = [*Path(__file__).parent.glob('*.py'), Path(__file__).resolve().parents[1]/'dlss_fi_contract.py',
                      Path(__file__).resolve().parents[1]/'dlss_nr_advanced.py']
    code = {str(p): file_identity(p) for p in implementation}
    monitor = session = None
    mapped_paths = set()
    with NvmlResourceReader() as reader:
        monitor = Monitor(reader)
        monitor.thread.start()
        try:
            inventory = cuda_device_inventory()
            monitor.check()
            session = BinarySession([identities['dlssg-worker.exe']['path'], '--serve'],
                cwd=Path(runtime), width=source['width'], height=source['height'],
                frame_count=source['plan'].source_count, cancel=monitor.cancel)
            with (task_dir/'frames.jsonl').open('x', encoding='utf8') as frames:
                def observe(row):
                    frames.write(json.dumps(row)+'\n')
                    if row['slot'] in (0, source['plan'].output_count-2):
                        mapped_paths.update(m.path for m in psutil.Process(session.process.pid).memory_maps(grouped=True)
                                            if 'dlss' in m.path.lower())
                stream = FrameStream(source['plan'], width=source['width'], height=source['height'], session=session,
                                     observer=observe, check=monitor.check)
                encoded = encode_video(task_dir/'video-only.mp4', source,
                    stream.outputs(decode_frames(source, check=monitor.check)), check=monitor.check)
            if stream.report is None or session.next_index != source['plan'].source_count:
                raise ValueError('FI stream did not finish all declared inputs')
            session.close()
            logs = b''.join(session.logs).decode('utf8', 'replace')
            binding = bind_probe({'supports_native_2x_by_report': session.setup['maximum_generated_per_interval'] >= 1,
                                  'stderr_tail': logs}, inventory, monitor.startup['gpu_uuid'])
            mapped = {Path(p).name: file_identity(p) for p in mapped_paths}
            for name, identity in identities.items():
                if name not in mapped or any(mapped[name][k] != identity[k] for k in ('path', 'sha256', 'bytes')):
                    raise ValueError('Actual mapped runtime does not match the pinned pair')
            media = mux_and_validate(task_dir/'video-only.mp4', source, task_dir/'candidate.mp4', check=monitor.check)
            if runtime_identity(runtime) != identities or any(file_identity(p) != value for p, value in code.items()):
                raise ValueError('FI implementation/runtime changed during execution')
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                monitor.close()
    result = {'status': 'file_validated_pending_atomic_publish', 'source': source['file'], 'media': media,
              'ledger': stream.report, 'encoder': encoded, 'runtime': identities, 'code': code,
              'binding': binding, 'mapped': mapped, 'resources': monitor.guard.report(),
              'worker_stop': session.stop_receipt, 'worker_log': logs,
              'quality_qualified': False, 'source_pixels_lossless': False}
    write_json(task_dir/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--cuts', default='[]')
    args = parser.parse_args()
    cut_list = json.loads(args.cuts)
    if not isinstance(cut_list, list) or any(type(i) is not int for i in cut_list):
        raise ValueError('Cut indices must be an integer list')
    process_file(args.source, args.runtime, args.task_dir, cut_list)
