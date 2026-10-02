"""Opt-in file VIDEO adapter with a private, owned CPU codec process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def _owned_worker(worker, request, report, check):
    if os.name == 'nt':
        from .dlss_fi_backend.process import run_isolated, IsolatedTaskError
        interrupted = []

        def guarded_check():
            try:
                check()
            except BaseException as error:
                interrupted.append(error)
                raise

        try:
            receipt = run_isolated(worker, [str(request), str(report)], timeout=1800, check=guarded_check)
        except IsolatedTaskError:
            if interrupted:
                raise interrupted[0]
            raise
        return receipt
    # A pipe that nobody drains until exit can block a noisy codec forever.
    # File-backed bounded tail reading avoids that deadlock without a thread.
    stderr_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen([sys.executable, '-I', str(worker), str(request), str(report)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr_file)
    except BaseException:
        stderr_file.close()
        raise
    try:
        deadline = time.monotonic() + 1800
        while process.poll() is None:
            check()
            if time.monotonic() > deadline:
                raise TimeoutError('Opening fade codec deadline exceeded')
            try:
                process.wait(timeout=.05)
            except subprocess.TimeoutExpired:
                pass
        stderr_file.seek(0, 2)
        stderr_file.seek(max(0, stderr_file.tell()-8192))
        detail = stderr_file.read(8192).decode('utf8', errors='replace')
        if process.returncode:
            raise RuntimeError('Opening fade codec failed: ' + detail[-4000:])
        return {'status': 'complete', 'exit_code': 0, 'active_after_cleanup': 0}
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stderr_file.close()


def mute_fade_file_video(video, mute_first_frames, fade_in_ms, *, interrupt_check=None, output_directory=None):
    import av
    import folder_paths
    from comfy_api.latest import InputImpl, Types
    if interrupt_check is None:
        from comfy.model_management import throw_exception_if_processing_interrupted
        interrupt_check = throw_exception_if_processing_interrupted
    interrupt_check()
    source = video.get_stream_source()
    if hasattr(source, 'seek'):
        source.seek(0)
    with av.open(source) as container:
        if not container.streams.audio:
            return video, json.dumps({'status': 'no_audio_passthrough', 'video_changed': False})
        raw_dimensions = (container.streams.video[0].width, container.streams.video[0].height)
        native_mp4 = 'mp4' in container.format.name.split(',')
    trim = video.get_active_trim_window()
    directory = Path(output_directory or folder_paths.get_temp_directory())
    directory.mkdir(parents=True, exist_ok=True)
    # Intermediate lifetime is a Core temp file, never a shared cache/output clip.
    owned = Path(tempfile.mkdtemp(prefix='t8-opening-fade-', dir=directory))
    active_materialized = (not isinstance(source, str) or trim != (0., 0.)
                           or video.get_dimensions() != raw_dimensions or not native_mp4)
    try:
        if active_materialized:
            source = str(owned / 'active_view.mp4')
            # Core's existing streaming exporter applies the requested view;
            # do not get_components() or flatten a VFR file to a tensor batch.
            video.save_to(source, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO)
        target, request, receipt_path = owned / 'opening_fade.mp4', owned / 'request.json', owned / 'report.json'
        request.write_text(json.dumps(dict(source=source, destination=str(target), n=mute_first_frames,
                                           fade_ms=str(fade_in_ms))), encoding='utf8')
        process = _owned_worker(Path(__file__).with_name('opening_fade_file_worker.py'), request, receipt_path, interrupt_check)
        interrupt_check()
        report = json.loads(receipt_path.read_text(encoding='utf8'))
        report.update(process=process, core_active_view_materialized=active_materialized,
            frame_batch_materialized=False,
            video_preservation_basis='Core active-view MP4 export' if active_materialized else 'original file video stream',
            warning='Long mute/fade can suppress the first word. Native trim/crop/non-MP4 export may transcode the existing view; subsequent fade copies that video stream unchanged.')
        return InputImpl.VideoFromFile(str(target)), json.dumps(report, ensure_ascii=False, indent=2)
    except BaseException:
        # Keep private codec evidence for diagnosis; never return a failed candidate.
        raise
