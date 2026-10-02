"""File-backed 32kHz stereo conditioning with source-timestamp placement.

This is a temporary, resampled model input, never the delivered audio track.
Decoder/resampler frames are written by timestamp into a zero-filled disk file;
the complete waveform is not materialized in RAM. All final source tracks remain
the responsibility of the original-packet-copy finalizer.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_plan import _integer


class OutpaintAudioIntegrityError(ValueError):
    """A temporary conditioning input changed; dependent latent commits are invalid."""


class OutpaintAudioFile:
    def __init__(self, path, sample_count, plan, *, stream_position, pcm_sha256):
        self.path = Path(path)
        self.sample_count = sample_count
        self.plan = plan
        self.stream_position = stream_position
        self.pcm_sha256 = pcm_sha256

    def read(self, start, stop):
        _integer(start, "sample_start", 0, self.sample_count-1)
        _integer(stop, "sample_stop", start+1, self.sample_count)
        if stop-start > (128+26)*800:
            raise ValueError("audio read exceeds the bounded CNN waveform budget")
        with self.path.open("rb") as stream:
            stream.seek(start*8)
            data = stream.read((stop-start)*8)
        if len(data) != (stop-start)*8:
            raise ValueError("canonical audio file was truncated")
        result = torch.from_numpy(np.frombuffer(data, dtype="<f4").copy().reshape(-1, 2).T).unsqueeze(0)
        if not torch.isfinite(result).all():
            raise ValueError("canonical audio contains nonfinite samples")
        return result

    def shot(self, shot_index):
        _integer(shot_index, "shot_index", 0, len(self.plan["shots"])-1)
        shot = self.plan["shots"][shot_index]
        start = round(Fraction(shot["start"]*32000, 24))
        stop = round(Fraction(shot["stop"]*32000, 24))

        def reader(a, b):
            _integer(a, "shot_sample_start", 0, stop-start-1)
            _integer(b, "shot_sample_stop", a+1, stop-start)
            return self.read(start+a, start+b)

        return stop-start, reader


@contextmanager
def prepare_outpaint_audio_file(inspection, plan, *, stream_position=0, scratch_parent=None, interrupt_check=None):
    """Yield a bounded reader or None for a genuinely audio-free input.

    Explicit stream selection defaults to the first track; the pinned upstream
    chooses the last track. Multi-track parity probes must choose the same track.
    Missing intervals of an existing track are observed silence, matching the
    reference's zero-filled source timeline. No audio stream means unobserved
    model audio instead, not an observed all-zero posterior.
    """
    source, checked = validate_outpaint_source(inspection, plan)
    tracks = len(inspection["audio_pcm"])
    if tracks == 0:
        if stream_position != 0:
            raise ValueError("audio-free source has no selectable track")
        try:
            yield None
        finally:
            validate_outpaint_source(inspection, checked)
        return
    _integer(stream_position, "audio_stream_position", 0, tracks-1)
    origin = Fraction(inspection["cfr"]["first_pts"]) * Fraction(inspection["cfr"]["time_base"])
    sample_count = round(Fraction(checked["source"]["frames"]*32000, 24))
    with tempfile.TemporaryDirectory(prefix="t8-outpaint-source-audio-", dir=scratch_parent) as directory:
        path = Path(directory) / "conditioning-stereo-32000.f32le"
        _decode_isolated(source, path, origin, sample_count, stream_position, interrupt_check)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024*1024), b""):
                if interrupt_check:
                    interrupt_check()
                digest.update(data)
        validate_outpaint_source(inspection, checked)
        try:
            yield OutpaintAudioFile(path, sample_count, checked, stream_position=stream_position,
                                   pcm_sha256=digest.hexdigest())
        finally:
            validate_outpaint_source(inspection, checked)
            after = hashlib.sha256()
            with path.open("rb") as stream:
                for data in iter(lambda: stream.read(1024*1024), b""):
                    after.update(data)
            if after.hexdigest() != digest.hexdigest():
                raise OutpaintAudioIntegrityError("canonical conditioning audio changed during use")


def _decode_isolated(source, target, origin, samples, track, interrupt_check):
    from .video_outpaint_compose import _stop_owned

    process = None
    worker = Path(__file__).with_name("video_outpaint_audio_decode.py")
    with tempfile.TemporaryFile() as errors:
        try:
            if interrupt_check:
                interrupt_check()
            process = subprocess.Popen([
                sys.executable, "-I", str(worker), "--source", str(source), "--output", str(target),
                "--origin="+str(origin), "--samples", str(samples), "--track", str(track),
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), shell=False)
            deadline = time.monotonic()+max(120, samples/32000*4)
            while process.poll() is None:
                if interrupt_check:
                    interrupt_check()
                if time.monotonic() > deadline:
                    raise TimeoutError("isolated conditioning audio decode timed out")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell()-2000))
                detail = errors.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"isolated conditioning audio decode failed ({process.returncode}): {detail}")
            if not target.is_file() or target.stat().st_size != samples*8:
                raise RuntimeError("isolated conditioning audio decode returned an incomplete PCM file")
        finally:
            _stop_owned(process)
